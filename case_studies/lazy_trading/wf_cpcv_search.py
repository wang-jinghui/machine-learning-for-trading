# -*- coding: utf-8 -*-
"""WalkForward + CombinatorialPurgedCV 嵌套自适应参数搜索模块。

架构定位：**cpcv_parameter_search 是 walkforward_parameter_search 的嵌套变体**
——pipeline、fitness 注册表、参数空间 schema 全部与 walkforward 同源复用
（import 自 walkforward_parameter_search），唯一增量是两点：

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
    walkforward_parameter_search.suggest_from_space 统一采样。
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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from skfolio.model_selection import WalkForward, CombinatorialPurgedCV, cross_val_predict
from skfolio.optimization import EqualWeighted

# 与 walkforward_parameter_search 同源复用（结构对齐、永不漂移）：
#   build_pipeline     筛选+等权 pipeline（complete→variance→extremes→nondomin→correlate→optimization）
#   suggest_from_space 参数空间节点 → optuna trial 采样（range→int/float, choice→categorical）
#   StopWhenNoImprovement / FITNESS_MEASURES / expand_range 等
from walkforward_parameter_search import (
    FITNESS_MEASURES,
    StopWhenNoImprovement,
    build_pipeline,
    suggest_from_space,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "cpcv_parameter_search_config.toml"

# build_model 与 walkforward 的 build_pipeline 对齐（含 extremes 步骤），
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
        n_jobs=cpcv.get("n_jobs", 4),          # 内层 optuna trial 级并行
        cv_n_jobs=cpcv.get("cv_n_jobs", 4),    # 单 trial 内 CPCV 并行
        n_trials=cpcv.get("n_trials", 40),
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
        # 压测无 optuna trial 层：n_jobs 为单次 cross_val_predict 并行（单层，可直接拉满）
        n_jobs=cpcv.get("n_jobs", 4),
        verbose=cpcv.get("verbose", True),
    )
    return {"search_kwargs": search_kwargs, "paths_kwargs": paths_kwargs,
            "space": load_space(config_path), "outer_wf": outer}


# ---------------------------------------------------------------------------
# 内层评分：CPCV 路径级指标分布均值组合
# ---------------------------------------------------------------------------
def inner_cpcv_score(X_tr, params, test_size, n_jobs=4, n_test_folds=2):
    """在 train+test 合并窗口上做 CPCV, 返回路径级分数。

    窗口 = 内层训练 ts 天 + 内层测试 test_size 天(与外层 test 同长)
    块长 = test_size // n_test_folds, n_folds = 窗口天数 // 块长
    -> 最后 n_test_folds 块恰好覆盖 test_size 天, 前 n_folds-n_test_folds 块覆盖 ts 天

    分数 = mean(路径 annualized_mean) - mean(路径 max_drawdown)
         + mean(路径 skew)：逐路径取指标再平均 —— 对应 walkforward 目标
        asr_mdd_skew 的"路径分布均值"版本（CPCV 输出每条路径一个 MPP）。
    params 为管道前缀键（含 nondomin__fitness_measures 名称或组合列表），
    与 walkforward build_pipeline 契约一致。
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
                                     purged_size=2, embargo_size=2)
    cvp = cross_val_predict(model, X_tr, cv=inner_cv, n_jobs=n_jobs)
    #ann_mean = np.array([mptf.annualized_mean for mptf in cvp])
    ann_sr = np.array([mptf.annualized_sharpe_ratio for mptf in cvp])
    max_dd = np.array([mptf.max_drawdown for mptf in cvp])
    #avg_dd = np.array([mptf.average_drawdown for mptf in cvp])
    skew = np.array([mptf.skew for mptf in cvp])
    #kurt = np.array([mptf.kurtosis for mptf in cvp])
    return float(np.median(ann_sr) - np.median(max_dd) + np.median(skew)) #- np.mean(kurt))//100


# ---------------------------------------------------------------------------
# 内层 Optuna 搜索（TPE 采样，空间由 suggest_from_space 统一驱动）
# ---------------------------------------------------------------------------
def search_inner_params(X_tr, space, test_size, n_test_folds=2, n_jobs=4,
                        cv_n_jobs=4, n_trials=40, patience=50, min_delta=1e-4,
                        verbose=True, seed=42):
    """Optuna 搜索: 在 train_i 上用 TPE 优化 CPCV 路径分数, 返回该折最优参数。

    space : walkforward 同构参数空间（管道前缀键 + train_size 增量键）。
    并行度分两层（与 walkforward run_optuna 对齐）：
        n_jobs : optuna trial 级并行（study.optimize）
        cv_n_jobs : 单 trial 内 CPCV cross_val_predict 并行
    峰值并发 ≈ n_jobs × cv_n_jobs，预算需按机器核数控制。
    patience / min_delta : StopWhenNoImprovement 早停参数（与 walkforward
        run_optuna 同语义；内层 trial 预算小，默认 patience 比 wf 的 100 更敏感）
    Returns
    -------
    (best_params, best_value) : best_params 为 {键: 值}，fitness_measures
        为注册表名称字符串 —— 与 walkforward 搜索结果同形、可序列化
    """
    def objective(trial):
        # 与 walkforward 相同的采样逻辑（range→int/float, choice→categorical）
        params = suggest_from_space(trial, space)
        ts = params["train_size"]
        # 内层 CPCV 输入 = 训练 ts 天 + 测试 test_size 天(与外层 WF 结构一致);
        # ts 上限受约束(≤ 外层train-test_size), 保证 ts+test_size 不超出 X_tr
        w = X_tr.iloc[-(ts + test_size):]
        return inner_cpcv_score(w, params, test_size, n_jobs=cv_n_jobs, n_test_folds=n_test_folds)

    study = optuna_study(seed=seed)
    study.optimize(objective,
                   n_trials=n_trials,
                   n_jobs=n_jobs,
                   show_progress_bar=verbose,
                   callbacks=[StopWhenNoImprovement(patience=patience, min_delta=min_delta)])
    return dict(study.best_params), study.best_value


# ---------------------------------------------------------------------------
# 外层 WF 嵌套主循环 / 多路径压测
# ---------------------------------------------------------------------------
def _outer_wf(X, test_size, train_size, purged_size, reduce_test):
    return WalkForward(test_size=test_size, train_size=train_size,
                       purged_size=purged_size, reduce_test=reduce_test,
                       expand_train=False)


def nested_adaptive_search(X, test_size=126, train_size=756, space=None,
                           n_test_folds=2, n_jobs=4, cv_n_jobs=4, n_trials=40,
                           patience=50, min_delta=1e-4, verbose=True, seed=42,
                           outer_purged_size=1, outer_reduce_test=True):
    """方案B主循环: 外层WF滚动, 每折内层Optuna独立搜参, 该折参数预测纯净test段。

    参数说明（注意两个 train_size 同名不同义）：
        train_size : 外层 WalkForward 训练窗口（固定，天）
        space["train_size"] : 内层训练子窗口搜索维度（天，≤ 外层train-test_size）
    并行度分两层（与 walkforward run_optuna 对齐）：
        n_jobs : 内层 optuna trial 级并行
        cv_n_jobs : 单 trial 内 CPCV cross_val_predict 并行
    外层折间串行，峰值并发 ≈ n_jobs × cv_n_jobs。
    inner_cv 按训练子窗口动态构建（不外部传入）。

    NOTE: 搜索（默认 reduce_test=True）会把不足 test_size 的尾段缩短保留并
    照常搜参；压测阶段若 reduce_test=False 则不产出该尾段 → 尾折参数自然
    不被压测使用（两阶段设计如此，非错位）。fold 索引对齐只要求两阶段外层
    test_size / train_size / purged_size 一致。

    Returns
    -------
    list : [{fold, params, inner_median, test(portfolio)}]，params 为
        walkforward 同形参数（fitness_measures 为名称字符串，可序列化）
    """
    if space is None:
        space = load_space()
    outer_cv = _outer_wf(X, test_size, train_size, outer_purged_size, outer_reduce_test)
    folds = []
    for i, (tr_idx, te_idx) in enumerate(outer_cv.split(X)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        if len(X_tr) < 252:
            continue
        # 1) 内层搜索（只触碰训练块, test段保持纯净）
        best, score = search_inner_params(X_tr, space, test_size,
                                          n_test_folds=n_test_folds,
                                          n_jobs=n_jobs, cv_n_jobs=cv_n_jobs,
                                          n_trials=n_trials,
                                          patience=patience, min_delta=min_delta,
                                          verbose=verbose, seed=seed)
        # 2) 用该折最优参数重建模型, fit在训练子窗口, 直接predict纯净test段
        #    (Pipeline已含EqualWeighted最后一步, predict内部完成筛选+优化)
        w = X_tr.iloc[-best["train_size"]:]
        m = build_pipeline(best)
        m.fit(w)
        test_ptf = m.predict(X_te)
        test_ptf.name = f"Fold{i}"   # predict 不接受 portfolio_params(0.20.x), 预测后设置名称
        folds.append({"fold": i, "params": best, "inner_median": score, "test": test_ptf})
        if verbose:
            print(f"Fold {i}: inner_score={score:.4f} | "
                  f"test_ann={test_ptf.annualized_mean:.4f} | test_days={len(test_ptf.returns)}")
    return folds


def summarize_fold_params(fold_results):
    """解析 nested_adaptive_search 的每折参数结果，格式化为一行一折的表格。

    Parameters
    ----------
    fold_results : list
        nested_adaptive_search 返回的 folds（[{fold, params, inner_median, test}]）

    Returns
    -------
    pd.DataFrame
        列 = fold | inner_median | 各参数键；fitness_measures 展平为
        fitness 字符串列（便于一眼对比各折选了哪个 measure 组合）
    """
    rows = []
    for f in fold_results:
        p = dict(f["params"])
        p["fitness"] = str(p["nondomin__fitness_measures"])
        del p["nondomin__fitness_measures"]
        rows.append({"fold": f["fold"], "inner_median": round(f["inner_median"], 4), **p})
    return pd.DataFrame(rows)


def adaptive_multi_paths(X, folds, test_size=126, train_size=756,
                         n_test_folds=2, n_jobs=4, verbose=True,
                         outer_purged_size=1, outer_reduce_test=False):
    """每折: 参数p_i + (train_i最近train_size天 + test_i)窗口内CPCV 多路径压测。

    -> {fold: {path_id: [test块1, test块2, ...]}}; 窗口按该折train_size截断
    test两折为独立段(独立组合单元), 不拼成MPTF

    并行度：无 optuna trial 层，n_jobs 为单次 cross_val_predict 并行
    （与 inner_cpcv_score 同层，可按机器核数直接拉满）。

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
                                         purged_size=2, embargo_size=2)
        model = build_pipeline(p)
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
def optuna_study(seed=42, direction="maximize"):
    """新建 optuna study（TPESampler + seed，日志降噪）。"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
