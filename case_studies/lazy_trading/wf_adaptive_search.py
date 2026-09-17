# -*- coding: utf-8 -*-
"""单层 WF 每折 IS 自适应参数搜索模块（DSR/FDR/PBO 原生场景版）。

架构定位：与 wf_cpcv_search（WF+CPCV 嵌套，内层用"CPCV 验证块分数均值"
选参）并列的搜索方案——外层 WalkForward 窗口、参数空间、pipeline 与
wf_cpcv 完全同构；唯一区别是**选择分数**：每折在训练段上直接以"IS 拟合
分数"（fit 最近 ts 天 → 同窗预测的绩效）为目标做 Optuna 自适应搜索，
选中参数随后在该折纯净 test 段盲测。

为什么是 IS 拟合分数：DSR/FDR/PBO 原文的定义域是"在同一数据上试 N 次
选最优 → 选择偏差校正 → 独立 OOS 验证"。CPCV 验证块分数均值既非拟合
IS 也非时间外推 OOS，与校正工具口径失配；本模块把选择分数退回拟合 IS，
每折 (IS, OOS) 对与校正工具定义一一对应：

* 逐折 deflate：study trial 分数（IS 拟合分数序列）经 empirical_deflate
  经验零分布（GPD 上尾）校正；
* 判据："p_luck 高 + OOS 显著下降 → 过拟合"（与 wf_cpcv 同形）；
* 代价：验证集正则化被移除（IS 选参天生更过拟合），由上述判据暴露——
  检测体系的价值正是让"该杀"变得可执行。

复用（只读，冻结模块）：cpcv_search_base 与 cpcv_analysis 底座、以及
wf_cpcv_search 的公共件（_outer_wf / run_params_on_fold / optuna_study /
load_space）——本模块不修改任何现有文件；产出的 fold_results 与
nested_adaptive_search 同构，检测流（summarize_fold_params /
summarize_top_k_params / wf_cpcv_robustness / wf_cpcv_ablation）与
wf_cpcv 用同一份实现直接消费。

检测流调用形态（与 wf_cpcv 逐行对镜像）::

    cfg = load_adaptive_config("wf_adaptive_search_config.toml")
    folds = adaptive_wf_search(X, **cfg["search_kwargs"])   # space 缺省自动载入
    summarize_fold_params(folds)                            # wf_cpcv_search
    perturbation_mc(X, folds, space=cfg["space"])           # wf_cpcv_robustness
    topk_paths_eval(X, folds, **cfg["paths_kwargs"])        # 压测段见 [paths]
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from cpcv_search_base import (
    StopWhenNoImprovement,
    build_pipeline,
    suggest_from_space,
)

# 逐折 IS deflate：study trial 分数(IS 拟合 ASR)经 empirical_deflate 经验
# 零分布(GPD 上尾)校正选择偏差, 产出 p_luck/margin 等判据量与 OOS test
# ASR 对照(过拟合判定)
from cpcv_analysis import empirical_deflate

# wf_cpcv 冻结模块的公共件（只读引用——折位对齐/应用入口与 wf_cpcv
# 是同一份实现, 保证两套框架的 fold 索引与口径严格对齐）:
#   _outer_wf         外层 WalkForward 构建（同一窗口切分）
#   run_params_on_fold 固定参数折上应用（与扰动/敏感性/Top-K 同一入口）
#   optuna_study      采样器/seed 构建（TPE 自适应 | random 独立采样）
#   load_space        [param_space]+[nested_space] 配置读取
from wf_cpcv_search import (
    _outer_wf,
    load_space,
    optuna_study,
    run_params_on_fold,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "wf_adaptive_search_config.toml"


# ---------------------------------------------------------------------------
# 配置加载：[search] 单层搜索段 + [paths] 压测段（检测流共用）
# ---------------------------------------------------------------------------
def load_adaptive_config(config_path=DEFAULT_CONFIG) -> dict:
    """读取配置文件的 [search]/[paths] 段，展开为可直接 ** 传参的 kwargs。

    返回结构与 load_nested_config 同形（search_kwargs / paths_kwargs /
    space / outer_wf），便于两套框架的调用代码逐行镜像：

    * [param_space] + [nested_space] 与 cpcv 版逐键同构（同一份收敛空间）；
    * [search]：外层 WF 窗口 + 每折 Optuna 参数（单层 trial 并行，无内层
      CPCV，故无 cv_n_jobs / inner_purged_size / inner_embargo_size）；
    * [paths]：压测段参数（topk_paths_eval / ablate 的 CPCV 多路径压测——
      评估不选参，与选择结构解耦，与 cpcv 版 paths_* 语义一致）。
    """
    config_path = Path(config_path)
    cfg = tomllib.load(open(config_path, "rb"))
    s = cfg.get("search", {})
    p = cfg.get("paths", {})
    outer = dict(
        test_size=s.get("outer_test_size", 126),
        train_size=s.get("outer_train_size", 630),
        purged_size=s.get("outer_purged_size", 1),
    )
    search_kwargs = dict(
        test_size=outer["test_size"],
        train_size=outer["train_size"],
        outer_purged_size=outer["purged_size"],
        outer_reduce_test=s.get("outer_reduce_test", True),
        n_trials=s.get("n_trials", 100),
        sampler=s.get("sampler", "tpe"),
        patience=s.get("patience", 50),
        min_delta=s.get("min_delta", 1e-4),
        n_jobs=s.get("n_jobs", 12),
        seed=s.get("seed", 42),
        verbose=s.get("verbose", True),
    )
    paths_kwargs = dict(
        test_size=outer["test_size"],
        train_size=outer["train_size"],
        outer_purged_size=outer["purged_size"],
        outer_reduce_test=p.get("paths_reduce_test", False),
        n_test_folds=p.get("n_test_folds", 2),
        inner_purged_size=p.get("inner_purged_size", 2),
        inner_embargo_size=p.get("inner_embargo_size", 2),
        n_jobs=p.get("n_jobs", 12),
        verbose=s.get("verbose", True),
    )
    return {"search_kwargs": search_kwargs, "paths_kwargs": paths_kwargs,
            "space": load_space(config_path), "outer_wf": outer}


# ---------------------------------------------------------------------------
# 每折 IS 评分：拟合窗内绩效（DSR 的"同一数据上 N 次试验"分数）
# ---------------------------------------------------------------------------
def inner_is_score(X_tr, params):
    """单层 WF 每折 IS 拟合分数（选择依据）。

    = fit 最近 params["train_size"] 天 → 同窗 predict 的
    annualized_sharpe_ratio。纯拟合口径（无任何外推段/验证块），与
    wf_cpcv 的 inner_cpcv_score（CPCV 验证块分数均值）相对——拟合分数
    是 DSR/FDR/PBO 原文定义域内的选择分数。顺序切分下窗口起点的上市
    前导 NaN 由 pipeline 首行检查结构性免疫（无需像 CPCV 组合路径那样
    预剔除未上市列）。
    """
    ts = int(params["train_size"])
    w = X_tr.iloc[-ts:]
    model = build_pipeline(params)
    model.fit(w)
    return model.predict(w).annualized_sharpe_ratio


# ---------------------------------------------------------------------------
# 每折 Optuna 搜索（IS 拟合分数目标；TPE/random 采样可切换）
# ---------------------------------------------------------------------------
def search_is_params(X_tr, space, n_trials=100, n_jobs=12, patience=50,
                     min_delta=1e-4, verbose=True, seed=42, sampler="tpe"):
    """Optuna 在训练段上优化 IS 拟合分数, 返回 (best_params, best_value, study)。

    与 wf_cpcv 的 search_inner_params 平行, 区别仅在评估: trial 分数 =
    inner_is_score（IS 拟合, 无内层 CPCV 展开）——单层并行, 峰值并发 =
    n_jobs（无 cv_n_jobs 第二层）。
    sampler : "tpe"（自适应引导采样, 挂早停）| "random"（参数空间独立随机
        采样——empirical_deflate 的 iid 前提, 早停自动禁用, 跑满 n_trials）。
    study 保留每次 trial 的 params/value 全记录（供逐折 empirical_deflate：
    N=unique trials、trial 分数序列取自 study）。
    """
    def objective(trial):
        params = suggest_from_space(trial, space)
        return inner_is_score(X_tr, params)

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
# 单层 WF 主循环：每折 IS 自适应搜参 → 该折参数预测纯净 OOS 段
# ---------------------------------------------------------------------------
def adaptive_wf_search(X, test_size=126, train_size=630, space=None,
                       n_jobs=12, n_trials=100, patience=50, min_delta=1e-4,
                       verbose=True, seed=42, sampler="tpe",
                       outer_purged_size=1, outer_reduce_test=True):
    """外层 WF 滚动, 每折在训练段上 IS 自适应搜参, 该折参数盲测纯净 test 段。

    参数说明（与 wf_cpcv 的 nested_adaptive_search 同形，注意两个
    train_size 同名不同义）：
        train_size : 外层 WalkForward 训练窗口（固定，天）
        space["train_size"] : 每折内 IS 拟合窗口搜索维度（天，≤ 外层
            train-test_size，保证拟合窗不越出训练段）
    外层折间串行，每折 Optuna trial 级并行（峰值并发 = n_jobs，单层）。
    sampler : "tpe" | "random"（random 为 empirical_deflate 的 iid 前提，
    早停自动禁用，见 search_is_params；两模式返回结构一致）。

    Returns
    -------
    list : 与 nested_adaptive_search 同构的 fold_results
        [{fold, params, score, test, train, "train ASR", "test ASR",
          p_luck, margin, max_p95, n_trials, study}]
        ——score = 该折 IS 拟合分数（Optuna best value，选择依据；WFE
        判读用 test ASR / score）。p_luck/margin/max_p95/n_trials
        为该折 study 的 empirical_deflate（经验零分布 GPD 上尾）摘要
        ——与 "test ASR"(OOS) 对照即 "p_luck 高 + OOS 显著下降 → 过拟合"
        判据；study 为该折 optuna study（trial 级全记录，可复算 deflate）。
        检测流与 wf_cpcv 同形直接消费。
    """
    if space is None:
        space = load_space(DEFAULT_CONFIG)
    outer_cv = _outer_wf(X, test_size, train_size, outer_purged_size, outer_reduce_test)
    folds = []
    for i, (tr_idx, te_idx) in enumerate(outer_cv.split(X)):
        X_tr = X.iloc[tr_idx]
        if len(X_tr) < 252:
            continue
        # 1) 每折 IS 搜索（只触碰训练块, test 段保持纯净）
        best, score, study = search_is_params(X_tr, space,
                                              n_jobs=n_jobs, n_trials=n_trials,
                                              patience=patience, min_delta=min_delta,
                                              verbose=verbose, seed=seed,
                                              sampler=sampler)
        # 2) 该折最优参数普通 OOS 应用：fit 最近 train_size 天 → 预测纯净
        #    test 段（run_params_on_fold 为公共应用函数，与扰动/敏感性/
        #    Top-K 候选评估同一入口；同窗口同口径，折位对齐保证一致）
        train_ptf, test_ptf = run_params_on_fold(
            X, i, best, test_size=test_size, train_size=train_size,
            purged_size=outer_purged_size, reduce_test=outer_reduce_test)
        test_ptf.name = f"Fold{i}"   # predict 不接受 portfolio_params(0.20.x), 预测后设置名称
        # 3) 该折 IS deflate：study trial 分数(经验零分布 GPD 上尾) → 判据量,
        #    与下面 test ASR(OOS) 对照判定过拟合
        d = empirical_deflate(study)
        folds.append({"fold": i, "params": best, "score": score,
                      "test": test_ptf, "train": train_ptf,
                      "train ASR": train_ptf.annualized_sharpe_ratio,
                      "test ASR": test_ptf.annualized_sharpe_ratio,
                      "p_luck": d["p_luck"],
                      "margin": d["margin"], "max_p95": d["max_p95"],
                      "n_trials": d["n_trials"],
                      "study": study})   # trial 级全记录(params/value), 供复算 deflate
        if verbose:
            print(f"Fold {i}: IS score={score:.4f} | "
                  f"test_ann={test_ptf.annualized_mean:.4f} | test_days={len(test_ptf.returns)}")
    return folds
