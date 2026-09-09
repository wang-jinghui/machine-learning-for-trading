# -*- coding: utf-8 -*-
"""WalkForward + CombinatorialPurgedCV 嵌套自适应参数搜索模块。

架构定位：**cpcv_parameter_search 是 walkforward_parameter_search 的嵌套变体**
——pipeline、fitness 注册表、参数空间 schema 与 walkforward 同源同构，但
不再 import walkforward 本体：共享部分 fork 至 cpcv_search_base（两侧独立
演进、互不影响；同步策略见该模块 docstring），唯一增量是两点：

* 多一个 `train_size` 搜索维度（内层训练子窗口，walkforward 中是固定 cv 参数）；
* 搜索结构：外层 WalkForward 每折在内层 CPCV 上独立重搜参数（策略随市场
  状态自适应），外层 test 段永不进入任何搜索（绝对样本外）。

工作流（与 walkforward_parameter_search.py 配套）：
    WF 全局搜索易过拟合，只用于探测想法、收敛参数范围；最终必须把**同一份
    参数空间**通过本模块做 WF+CPCV 嵌套验证。因此两侧配置文件 [param_space]
    逐键同构（本模块配置 = wf 配置的 [param_space] + [nested_space] 的
    train_size）。

参数空间契约（与 walkforward 完全一致）：
    space = {键: range节点{low, high, step} | choice列表}
    键为管道前缀体系（extremes__k / nondomin__* / correlate__threshold），
    nondomin__fitness_measures 的 choice 候选为注册表名称字符串；
    train_size 节点（嵌套独有，无前缀）也并入同一 space，由
    cpcv_search_base.suggest_from_space 统一采样。
    每折最优参数 fitness_measures 为名称字符串 —— 与 wf 搜索结果同形、
    可直接对比/序列化（notebook 中为 measure 组合列表，此处已升级）。

用法::

    # 配置驱动（推荐）：与 wf 同构的空间 + 嵌套参数
    cfg = load_nested_config("cpcv_parameter_search_config.toml")
    folds = nested_adaptive_search(X, **cfg["search_kwargs"])      # space 缺省自动载入
    paths = adaptive_multi_paths(X, folds, **cfg["paths_kwargs"])

    # 或直接传 space 字典（键与 walkforward 配置 [param_space] 相同 + train_size）
    space = {
        "extremes__k": {"low": 0.3, "high": 0.5, "step": 0.1},
        "nondomin__min_n_assets": {"low": 5, "high": 15, "step": 5},
        "nondomin__threshold": {"low": -0.5, "high": -0.4, "step": 0.1},
        "correlate__threshold": {"low": 0.1, "high": 0.3, "step": 0.1},
        "nondomin__fitness_measures": ["mean-variance-avgdd", "mean-semideviation-avgdd"],
        "train_size": {"low": 252, "high": 504, "step": 126},
    }
    folds = nested_adaptive_search(X, test_size=126, train_size=630,
                                   space=space, n_trials=100, n_jobs=12,
                                   cv_n_jobs=4)   # n_jobs: optuna trial级并行;
                                                   # cv_n_jobs: 单trial内CPCV并行
"""

from __future__ import annotations

import tomllib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from skfolio.model_selection import WalkForward, CombinatorialPurgedCV, cross_val_predict
from skfolio.optimization import EqualWeighted

# cpcv 体系共享底座（fork 自 walkforward_parameter_search，两侧独立演进、
# 互不影响——不 import walkforward 本体，避免连带加载其顶层代码与配置）：
#   build_pipeline     筛选+等权 pipeline（complete→variance→extremes→nondomin→correlate→optimization）
#   suggest_from_space 参数空间节点 → optuna trial 采样（range→int/float, choice→categorical）
#   StopWhenNoImprovement / FITNESS_MEASURES
from cpcv_search_base import (
    FITNESS_MEASURES,
    StopWhenNoImprovement,
    build_pipeline,
    suggest_from_space,
)

# 逐折 IS deflate：study trial 分数(mean path ASR)经 calc_dsr H0 噪声基准
# 校正选择偏差, 产出 dsr/margin 等判据量与 OOS test ASR 对照(过拟合判定);
# top_trials 用于 Top-K 候选提取(与 calc_dsr 同去重口径)
from cpcv_analysis import calc_dsr, top_trials

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "cpcv_parameter_search_config.toml"

# build_model 与 cpcv_search_base 的 build_pipeline 同义（含 extremes 步骤），
# 保留别名以便与 notebook 调用点名称一致
build_model = build_pipeline


# ---------------------------------------------------------------------------
# 配置加载：与 walkforward 同构的参数空间 + [cpcv] 嵌套段
# ---------------------------------------------------------------------------
def load_space(config_path=DEFAULT_CONFIG) -> dict:
    """读取配置文件的 [param_space] + [nested_space]，合并为单一搜索空间。

    键名与节点结构原样保留（zero 转换）：[param_space] 与
    walkforward_parameter_search 配置逐键同构（含 extremes__k，CPCV 嵌套
    同样搜索该维度）；[nested_space] 提供 train_size（嵌套独有增量维度）。
    """
    cfg = tomllib.load(open(config_path, "rb"))
    space = dict(cfg["param_space"])
    space.update(cfg.get("nested_space", {}))
    return space


def load_nested_config(config_path=DEFAULT_CONFIG) -> dict:
    """读取 [cpcv] 段配置，展开为可直接 ** 传给搜索/压测函数的 kwargs。

    Returns
    -------
    dict : {"search_kwargs": {...}, "paths_kwargs": {...}, "space": {...},
            "outer_wf": {test_size, train_size, purged_size}}，两阶段外层
        WalkForward 窗口（test/train/purged）一致；reduce_test 各自独立：
        搜索默认 True（不足 test_size 的尾段缩短保留并照常搜参）、
        压测默认 False（不产出不完整尾段，尾折参数自然不被使用）。
    """
    config_path = Path(config_path)
    cfg = tomllib.load(open(config_path, "rb"))
    cpcv = cfg["cpcv"]
    outer = dict(
        test_size=cpcv["outer_test_size"],
        train_size=cpcv["outer_train_size"],
        purged_size=cpcv.get("outer_purged_size", 1),
    )
    search_kwargs = dict(
        test_size=outer["test_size"],
        train_size=outer["train_size"],
        outer_purged_size=outer["purged_size"],
        outer_reduce_test=cpcv.get("outer_reduce_test", True),
        n_test_folds=cpcv.get("n_test_folds", 2),
        inner_purged_size=cpcv.get("inner_purged_size", 2),    # 内层 CPCV purge（checklist 1.3 显式化）
        inner_embargo_size=cpcv.get("inner_embargo_size", 2),  # 内层 CPCV embargo
        n_jobs=cpcv.get("n_jobs", 4),          # 内层 optuna trial 级并行
        cv_n_jobs=cpcv.get("cv_n_jobs", 4),    # 单 trial 内 CPCV 并行
        n_trials=cpcv.get("n_trials", 40),
        sampler=cpcv.get("sampler", "tpe"),    # 内层采样器: "tpe"(自适应) | "random"(独立采样, DSR deflate 前提)
        patience=cpcv.get("patience", 50),     # 早停：连续无有效改善 trial 数
        min_delta=cpcv.get("min_delta", 1e-4), # 早停：有效改善的最小增量
        seed=cpcv.get("seed", 42),
        verbose=cpcv.get("verbose", True),
    )
    paths_kwargs = dict(
        test_size=outer["test_size"],
        train_size=outer["train_size"],
        outer_purged_size=outer["purged_size"],
        outer_reduce_test=cpcv.get("paths_reduce_test", False),
        n_test_folds=cpcv.get("n_test_folds", 2),
        inner_purged_size=cpcv.get("inner_purged_size", 2),
        inner_embargo_size=cpcv.get("inner_embargo_size", 2),
        # 压测无 optuna trial 层：n_jobs 为单次 cross_val_predict 并行（单层，可直接拉满）
        n_jobs=cpcv.get("n_jobs", 4),
        verbose=cpcv.get("verbose", True),
    )
    return {"search_kwargs": search_kwargs, "paths_kwargs": paths_kwargs,
            "space": load_space(config_path), "outer_wf": outer}


# ---------------------------------------------------------------------------
# 内层评分：CPCV 路径级指标分布均值组合
# ---------------------------------------------------------------------------
def inner_cpcv_score(X_tr, params, test_size, n_jobs=4, n_test_folds=2,
                     inner_purged_size=2, inner_embargo_size=2):
    """在 train+test 合并窗口上做 CPCV, 返回路径级分数。

    窗口 = 内层训练 ts 天 + 内层测试 test_size 天(与外层 test 同长)
    块长 = test_size // n_test_folds, n_folds = 窗口天数 // 块长
    -> 最后 n_test_folds 块恰好覆盖 test_size 天, 前 n_folds-n_test_folds 块覆盖 ts 天

    分数 = mean(路径 annualized_mean) - mean(路径 max_drawdown)
         + mean(路径 skew)：逐路径取指标再平均 —— 对应 walkforward 目标
        asr_mdd_skew 的"路径分布均值"版本（CPCV 输出每条路径一个 MPP）。
    params 为管道前缀键（含 nondomin__fitness_measures 名称或组合列表），
    与 walkforward build_pipeline 契约一致。
    inner_purged_size / inner_embargo_size : 内层 CPCV 的 Purge/Embargo
        （标签重叠净化 + 验证段后自相关隔离，checklist 1.3；默认 2 与原
        硬编码行为一致，由 nested_adaptive_search / adaptive_multi_paths
        参数化透传，来源可配置并写入实验日志）。
    """
    # CPCV 组合路径无时序约束: test 块可整体早于 train 块, SelectComplete
    # 的"首行检查"只覆盖 train 块首行(窗口深处) → 窗口起点后的上市前导
    # NaN 会在早期 test 块漏出(optuna worker 无 pandas set_output 时
    # DropZeroVariance 校验崩溃)。与 WF 顺序切分(结构性免疫)对齐的做法:
    # 进 CPCV 前按窗口起点剔除未上市列——上市日为起点前的历史事实, 无泄漏;
    # SelectComplete 仍逐路径执行, 幂等保留。
    X_tr = X_tr.loc[:, X_tr.iloc[0].notna()]
    model = build_pipeline(params)
    block = max(1, test_size // n_test_folds)
    n_folds = max(n_test_folds + 1, len(X_tr) // block)
    inner_cv = CombinatorialPurgedCV(n_folds=n_folds, n_test_folds=n_test_folds,
                                     purged_size=inner_purged_size,
                                     embargo_size=inner_embargo_size)
    cvp = cross_val_predict(model, X_tr, cv=inner_cv, n_jobs=n_jobs)
    #ann_mean = np.array([mptf.annualized_mean for mptf in cvp]).mean()
    #ann_std = np.array([mptf.annualized_standard_deviation for mptf in cvp]).mean()
    #ann_semidev = np.array([mptf.annualized_semi_deviation for mptf in cvp]).mean()
    #mad = np.array([mptf.mean_absolute_deviation for mptf in cvp]).mean()
    #cvar = np.array([mptf.cvar for mptf in cvp]).mean()
    #cdar = np.array([mptf.cdar for mptf in cvp]).mean()
    #edar = np.array([mptf.edar for mptf in cvp]).mean()
    #maxdd = np.array([mptf.max_drawdown for mptf in cvp]).mean()
    #avgdd = np.array([mptf.average_drawdown for mptf in cvp]).mean()
    #skew = np.array([mptf.skew for mptf in cvp]).mean()
    #kurt = np.array([mptf.kurtosis for mptf in cvp]).mean()
    # RatioMeasure 全量 18 项：超额 Mean 除以各风险度量的比率（逐路径取均值）
    #sr = np.array([mptf.sharpe_ratio for mptf in cvp]).mean()
    asr = np.array([mptf.annualized_sharpe_ratio for mptf in cvp]).mean()
    #sor = np.array([mptf.sortino_ratio for mptf in cvp]).mean()
    #asor = np.array([mptf.annualized_sortino_ratio for mptf in cvp]).mean()
    #madr = np.array([mptf.mean_absolute_deviation_ratio for mptf in cvp]).mean()
    #flpmr = np.array([mptf.first_lower_partial_moment_ratio for mptf in cvp]).mean()
    #varr = np.array([mptf.value_at_risk_ratio for mptf in cvp]).mean()
    #cvarr = np.array([mptf.cvar_ratio for mptf in cvp]).mean()
    #ermr = np.array([mptf.entropic_risk_measure_ratio for mptf in cvp]).mean()
    #evarr = np.array([mptf.evar_ratio for mptf in cvp]).mean()
    #wrr = np.array([mptf.worst_realization_ratio for mptf in cvp]).mean()
    #darr = np.array([mptf.drawdown_at_risk_ratio for mptf in cvp]).mean()
    #cdarr = np.array([mptf.cdar_ratio for mptf in cvp]).mean()
    #calmar = np.array([mptf.calmar_ratio for mptf in cvp]).mean()
    #avgddr = np.array([mptf.average_drawdown_ratio for mptf in cvp]).mean()
    #edarr = np.array([mptf.edar_ratio for mptf in cvp]).mean()
    #uir = np.array([mptf.ulcer_index_ratio for mptf in cvp]).mean()
    #ginir = np.array([mptf.gini_mean_difference_ratio for mptf in cvp]).mean()
    return asr


# ---------------------------------------------------------------------------
# 内层 Optuna 搜索（TPE/随机采样可切换，空间由 suggest_from_space 统一驱动）
# ---------------------------------------------------------------------------
def search_inner_params(X_tr, space, test_size, n_test_folds=2, n_jobs=4,
                        cv_n_jobs=4, n_trials=40, patience=50, min_delta=1e-4,
                        verbose=True, seed=42, sampler="tpe",
                        inner_purged_size=2, inner_embargo_size=2):
    """Optuna 搜索: 在 train_i 上优化 CPCV 路径分数, 返回该折最优参数与 study。

    space : walkforward 同构参数空间（管道前缀键 + train_size 增量键）。
    sampler : "tpe"（自适应引导采样）| "random"（参数空间独立随机采样）。
        random 模式下各 trial 相互独立——这是 DSR deflate 的 iid 前提，
        故早停（StopWhenNoImprovement）自动禁用，跑满 n_trials。
    并行度分两层（与 walkforward run_optuna 对齐）：
        n_jobs : optuna trial 级并行（study.optimize）
        cv_n_jobs : 单 trial 内 CPCV cross_val_predict 并行
    峰值并发 ≈ n_jobs × cv_n_jobs，预算需按机器核数控制。
    patience / min_delta : StopWhenNoImprovement 早停参数（仅 tpe 模式生效；
        与 walkforward run_optuna 同语义；内层 trial 预算小，默认 patience
        比 wf 的 100 更敏感）
    Returns
    -------
    (best_params, best_value, study) : best_params 为 {键: 值}，
        fitness_measures 为注册表名称字符串 —— 与 walkforward 搜索结果同形、
        可序列化；study 保留每次 trial 的 params/value 全记录（供逐折 DSR
        deflate：N=unique trials、trial 分数序列取自 study）
    """
    def objective(trial):
        # 与 walkforward 相同的采样逻辑（range→int/float, choice→categorical）
        params = suggest_from_space(trial, space)
        ts = params["train_size"]
        # 内层 CPCV 输入 = 训练 ts 天 + 测试 test_size 天(与外层 WF 结构一致);
        # ts 上限受约束(≤ 外层train-test_size), 保证 ts+test_size 不超出 X_tr
        w = X_tr.iloc[-(ts + test_size):]
        return inner_cpcv_score(w, params, test_size, n_jobs=cv_n_jobs,
                                n_test_folds=n_test_folds,
                                inner_purged_size=inner_purged_size,
                                inner_embargo_size=inner_embargo_size)

    study = optuna_study(seed=seed, sampler=sampler)
    # random 无历史引导，早停只会随机截断搜索 → 仅 tpe 模式挂早停回调
    callbacks = ([StopWhenNoImprovement(patience=patience, min_delta=min_delta)]
                 if sampler == "tpe" else [])
    study.optimize(objective,
                   n_trials=n_trials,
                   n_jobs=n_jobs,
                   show_progress_bar=verbose,
                   callbacks=callbacks)
    return dict(study.best_params), study.best_value, study


# ---------------------------------------------------------------------------
# 外层 WF 嵌套主循环 / 多路径压测
# ---------------------------------------------------------------------------
def _outer_wf(X, test_size, train_size, purged_size, reduce_test):
    return WalkForward(test_size=test_size, train_size=train_size,
                       purged_size=purged_size, reduce_test=reduce_test,
                       expand_train=False)


def run_params_on_fold(X, fold_idx, params, test_size=126, train_size=756,
                       purged_size=1, reduce_test=True):
    """固定参数在外层 WF 第 fold_idx 折上的普通 OOS 应用（无内层 CPCV 展开）。

    语义与 nested_adaptive_search 折末完全一致：模型 fit 在训练段最近
    params["train_size"] 天（内层训练子窗口，注意与外层参数 train_size
    同名不同义），随后预测全训练段与纯净 test 段 —— 供参数扰动 /
    敏感性 / Top-K 候选评估等"固定参数直接应用"场景复用（只评估、不
    搜索、不选参，不构成 OOS 消耗）。
    fold_idx 与嵌套搜索折位对齐：调用方须使用与 nested_adaptive_search
    相同的 test_size / train_size / purged_size / reduce_test，否则折位
    错位（扰动/敏感性模块的逐折循环天然满足）。

    Returns
    -------
    (train_ptf, test_ptf) : 均为 skfolio Portfolio（日收益序列），
        train_ptf 覆盖整个外层训练段、test_ptf 覆盖该折纯净 OOS 段
    """
    outer_cv = _outer_wf(X, test_size, train_size, purged_size, reduce_test)
    tr_idx, te_idx = list(outer_cv.split(X))[fold_idx]
    X_tr = X.iloc[tr_idx]
    # 窗口与嵌套搜索同口径：最近 params["train_size"] 天
    w = X_tr.iloc[-params["train_size"]:]
    m = build_pipeline(params)
    m.fit(w)
    return m.predict(X_tr), m.predict(X.iloc[te_idx])


def nested_adaptive_search(X, test_size=126, train_size=756, space=None,
                           n_test_folds=2, n_jobs=4, cv_n_jobs=4, n_trials=40,
                           patience=50, min_delta=1e-4, verbose=True, seed=42,
                           sampler="tpe", outer_purged_size=1,
                           outer_reduce_test=True, inner_purged_size=2,
                           inner_embargo_size=2):
    """方案B主循环: 外层WF滚动, 每折内层Optuna独立搜参, 该折参数预测纯净test段。

    参数说明（注意两个 train_size 同名不同义）：
        train_size : 外层 WalkForward 训练窗口（固定，天）
        space["train_size"] : 内层训练子窗口搜索维度（天，≤ 外层train-test_size）
    并行度分两层（与 walkforward run_optuna 对齐）：
        n_jobs : 内层 optuna trial 级并行
        cv_n_jobs : 单 trial 内 CPCV cross_val_predict 并行
    外层折间串行，峰值并发 ≈ n_jobs × cv_n_jobs。
    inner_cv 按训练子窗口动态构建（不外部传入）。
    sampler : "tpe"（自适应引导采样）| "random"（参数空间独立随机采样）；
        random 模式是 DSR deflate 的 iid 前提（早停自动禁用，跑满 n_trials，
        见 search_inner_params），两模式返回结构一致。

    NOTE: 搜索（默认 reduce_test=True）会把不足 test_size 的尾段缩短保留并
    照常搜参；压测阶段若 reduce_test=False 则不产出该尾段 → 尾折参数自然
    不被压测使用（两阶段设计如此，非错位）。fold 索引对齐只要求两阶段外层
    test_size / train_size / purged_size 一致。

    Returns
    -------
    list : [{fold, params, score, test, train, "train ASR", "test ASR",
             dsr, sr_obs, margin, max_p95, n_trials, study}]；params 为
        walkforward 同形参数（fitness_measures 为名称字符串，可序列化）；
        dsr/sr_obs/margin/max_p95/n_trials 为该折 study 的 calc_dsr(H0 噪声
        deflate) 摘要——dsr = 最优分数超过纯运气基准的概率, sr_obs = best
        trial 的 val 分数, margin = sr_obs − E[max_N], max_p95 = 运气最大
        分布 95% 分位, n_trials = params 去重后的真实试验次数；与 "test
        ASR"(OOS) 对照即 "DSR 低 + OOS 显著下降 → 过拟合" 判据；study 为
        该折 optuna study（trial 级 params/value 全记录，可复算 deflate）
    """
    if space is None:
        space = load_space()
    outer_cv = _outer_wf(X, test_size, train_size, outer_purged_size, outer_reduce_test)
    folds = []
    for i, (tr_idx, te_idx) in enumerate(outer_cv.split(X)):
        X_tr = X.iloc[tr_idx]
        if len(X_tr) < 252:
            continue
        # 1) 内层搜索（只触碰训练块, test段保持纯净）
        best, score, study = search_inner_params(X_tr, space, test_size,
                                                 n_test_folds=n_test_folds,
                                                 n_jobs=n_jobs, cv_n_jobs=cv_n_jobs,
                                                 n_trials=n_trials,
                                                 patience=patience, min_delta=min_delta,
                                                 verbose=verbose, seed=seed,
                                                 sampler=sampler,
                                                 inner_purged_size=inner_purged_size,
                                                 inner_embargo_size=inner_embargo_size)
        # 2) 用该折最优参数做普通 OOS 应用：fit最近train_size天 → 预测纯净
        #    test段（run_params_on_fold 为公共应用函数，与扰动/敏感性/Top-K
        #    候选评估同一入口；同窗口同口径，行为与原折末内联逻辑一致）
        train_ptf, test_ptf = run_params_on_fold(
            X, i, best, test_size=test_size, train_size=train_size,
            purged_size=outer_purged_size, reduce_test=outer_reduce_test)
        test_ptf.name = f"Fold{i}"   # predict 不接受 portfolio_params(0.20.x), 预测后设置名称
        # 3) 该折 IS deflate: study trial 分数(H0 噪声基准) → 判据量,
        #    与下面 test ASR(OOS) 对照判定过拟合
        d = calc_dsr(study)
        folds.append({"fold": i, "params": best, "score": score,
                      "test": test_ptf, "train": train_ptf,
                      "train ASR": train_ptf.annualized_sharpe_ratio,
                      "test ASR": test_ptf.annualized_sharpe_ratio,
                      "dsr": d["dsr"], "sr_obs": d["sr_obs"],
                      "margin": d["margin"], "max_p95": d["max_p95"],
                      "n_trials": d["n_trials"],
                      "study": study})   # trial 级全记录(params/value), 供复算 deflate
        if verbose:
            print(f"Fold {i}: mean score={score:.4f} | "
                  f"test_ann={test_ptf.annualized_mean:.4f} | test_days={len(test_ptf.returns)}")
    return folds


def summarize_fold_params(fold_results):
    """解析 nested_adaptive_search 的每折参数结果，格式化为一行一折的表格。

    Parameters
    ----------
    fold_results : list
        nested_adaptive_search 返回的 folds（每折 dict：fold / params / score /
        train ASR / test ASR 与 calc_dsr 摘要 dsr / sr_obs / margin / n_trials）

    Returns
    -------
    pd.DataFrame
        列 = fold | score | dsr | sr_obs | margin | train ASR | test ASR
        | 各参数键 | n_trials；fitness_measures 展平为 fitness 字符串列
        （便于一眼对比各折选了哪个 measure 组合）
    """
    rows = []
    for f in fold_results:
        p = dict(f["params"])
        p["fitness"] = str(p["nondomin__fitness_measures"])
        del p["nondomin__fitness_measures"]
        rows.append({"fold": f["fold"], "score": round(f["score"], 4),
                     "dsr": round(f["dsr"], 4), "sr_obs": round(f["sr_obs"], 4),
                     "margin": round(f["margin"], 4),
                     "train ASR": round(f["train ASR"], 4),
                     "test ASR": round(f["test ASR"], 4), **p,
                     "n_trials": int(f["n_trials"])})
    return pd.DataFrame(rows)


def summarize_top_k_params(fold_results, k=5):
    """每折 Top-K 候选审计表（IS 侧，纯记录不触 OOS）。

    从每折 study 的 trial 记录按分数降序、参数去重取前 k 个候选（去重
    口径与 calc_dsr 的 N 统计一致，见 cpcv_analysis.top_trials）。
    top1 即生产参数（nested_adaptive_search 折末所用）；top2~k 候选供
    参数高原/孤峰的 OOS 侧压测对比（评估不选参，不消耗 OOS）。
    gap2top1 = top1 分数 − 当前候选分数：落差小 = 参数高原信号，悬崖 =
    孤峰信号（可与 OOS 侧候选压测结果相互印证）。

    Parameters
    ----------
    fold_results : list，nested_adaptive_search 返回的 folds（每折含 study）
    k : int，每折候选数

    Returns
    -------
    pd.DataFrame : 一行一折一候选，列 = fold | rank | score | gap2top1
        | 各参数键（fitness 展平为字符串）| n_trials
    """
    rows = []
    for f in fold_results:
        cands = top_trials(f["study"].trials, k=k)
        top_score = float(cands[0].values[0]) if cands else float("nan")
        for rank, t in enumerate(cands, start=1):
            p = dict(t.params)
            p["fitness"] = str(p["nondomin__fitness_measures"])
            del p["nondomin__fitness_measures"]
            rows.append({"fold": f["fold"], "rank": rank,
                         "score": round(float(t.values[0]), 4),
                         "gap2top1": round(top_score - float(t.values[0]), 4),
                         **p, "n_trials": int(f["n_trials"])})
    return pd.DataFrame(rows)


def derive_outer_window(folds, purged_size=1, reduce_test=False):
    """从 fold_results 推导外层 WF 窗口参数（消费侧折位对齐用）。

    adaptive_multi_paths / 消融 / 扰动等应用函数若窗口参数与嵌套搜索不一致
    （如漏传 test_size/train_size），外层 split 会静默错位甚至产出空折集
    ——比报错更危险。本函数从每折 train/test Portfolio 的实际段长度取众数
    还原窗口（尾折因 reduce_test 缩短，用众数而非末折），产出可直接 ** 传
    给 adaptive_multi_paths 的窗口四键。
    purged_size / reduce_test 无法从结果反推：purged_size 默认 1（与嵌套
    搜索默认一致，务必核对搜索配置），reduce_test 默认 False（压测语义：
    不产出不完整尾段，与 notebook 原语义一致）。
    """
    te_lens = Counter(len(f["test"].returns) for f in folds)
    tr_lens = Counter(len(f["train"].returns) for f in folds)
    return {"test_size": int(te_lens.most_common(1)[0][0]),
            "train_size": int(tr_lens.most_common(1)[0][0]),
            "outer_purged_size": purged_size,
            "outer_reduce_test": reduce_test}


def adaptive_multi_paths(X, folds, test_size=126, train_size=756,
                         n_test_folds=2, n_jobs=4, verbose=True,
                         outer_purged_size=1, outer_reduce_test=False,
                         inner_purged_size=2, inner_embargo_size=2,
                         pipeline_builder=None):
    """每折: 参数p_i + (train_i最近train_size天 + test_i)窗口内CPCV 多路径压测。

    -> {fold: {path_id: [test块1, test块2, ...]}}; 窗口按该折train_size截断
    test两折为独立段(独立组合单元), 不拼成MPTF

    并行度：无 optuna trial 层，n_jobs 为单次 cross_val_predict 并行
    （与 inner_cpcv_score 同层，可按机器核数直接拉满）。

    inner_purged_size / inner_embargo_size : 内层 CPCV Purge/Embargo，
        与 nested_adaptive_search 搜索侧同源参数（checklist 1.3）。
    pipeline_builder : 可选的 pipeline 构建器（默认 build_pipeline），
        供消融（步骤开关变体）等场景替换；签名与 build_pipeline 一致
        （接收每折 params dict → Pipeline）。

    NOTE: 默认 reduce_test=False → 不产出不足 test_size 的尾段，nested 搜索
    多出的尾折参数在此不会被使用（notebook 原语义）；fold 索引对齐只要求
    两阶段外层 test_size / train_size / purged_size 一致。
    """
    outer_cv = _outer_wf(X, test_size, train_size, outer_purged_size, outer_reduce_test)
    params_by_fold = {f["fold"]: f["params"] for f in folds}
    fold_paths = {}
    for i, (tr_idx, te_idx) in enumerate(outer_cv.split(X)):
        if i not in params_by_fold:
            continue
        p = params_by_fold[i]
        # 窗口 = train_i 最近 train_size 天 + test_i (搜索时模型fit口径)
        ts_len = p["train_size"]                        # N=最近N天
        w_idx = np.concatenate([tr_idx[-ts_len:], te_idx])
        window = X.iloc[w_idx]
        # 与 inner_cpcv_score 同因: CPCV 组合路径下 SelectComplete 只查 train
        # 块首行, 窗口起点的上市前导 NaN 会在早期 test 块漏出 → 按窗口起点
        # 剔除未上市列(历史事实, 无泄漏; SelectComplete 幂等保留)。
        window = window.loc[:, window.iloc[0].notna()]
        # 内层CPCV: 块长=test_size//n_test_folds, n_folds=窗口天数//块长
        # 使最后 n_test_folds 块恰好覆盖 test_size 天, 与内层搜索口径一致
        inner_cv = CombinatorialPurgedCV(n_folds=len(window) // (test_size // n_test_folds),
                                         n_test_folds=n_test_folds,
                                         purged_size=inner_purged_size,
                                         embargo_size=inner_embargo_size)
        builder = pipeline_builder if pipeline_builder is not None else build_pipeline
        model = builder(p)
        cvp = cross_val_predict(model, window, cv=inner_cv,
                                n_jobs=n_jobs, portfolio_params=dict(name=f"F{i}"))
        # test 段 = 每条路径最后 n_test_folds 个独立块(段), 不拼接, 作为独立组合单元
        fold_paths[i] = {pid: mptf[-n_test_folds:] for pid, mptf in enumerate(cvp)}
        if verbose:
            print(f"Fold {i}: paths={len(cvp)} | window={len(window)} | "
                  f"inner_folds={inner_cv.n_folds} | test_days={len(te_idx)}")
    return fold_paths


def build_test_parts(fold_paths, n_test_folds):
    """构建段单元列表: 每个元素 = 某(折, 块) 上所有路径的 Portfolio 列表。

    fold_paths : {fold: {path_id: [test块0, test块1, ...]}}
    n_test_folds : 每路径 test 段块数（= n_blocks）
    返回: all_test_parts, 共 len(folds)*n_test_folds 个段单元, 可直接喂给
    discrete_lhs_safe（cpcv_analysis 模块）
    """
    all_test_parts = []
    for i in sorted(fold_paths.keys()):
        for b in range(n_test_folds):
            all_test_parts.append([fold_paths[i][pid][b] for pid in sorted(fold_paths[i])])
    return all_test_parts


# ---------------------------------------------------------------------------
# Optuna study 工厂（延迟 import optuna，保持模块轻量）
# ---------------------------------------------------------------------------
def optuna_study(seed=42, direction="maximize", sampler="tpe"):
    """新建 optuna study（seed 固定 + 日志降噪）。

    sampler : "tpe" → TPESampler（自适应引导采样）；
              "random" → RandomSampler（独立随机采样，DSR deflate 的 iid 前提）
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if sampler == "random":
        optuna_sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        optuna_sampler = optuna.samplers.TPESampler(seed=seed)
    return optuna.create_study(direction=direction, sampler=optuna_sampler)
