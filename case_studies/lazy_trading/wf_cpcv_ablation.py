# -*- coding: utf-8 -*-
"""外层WF+内层CPCV嵌套流程的固定参数消融模块（checklist 4.3）。

定位：ablation_extremes.py 面向"简单 WF 单层 + 全局搜索"流程，与本目录
嵌套自适应流程（wf_cpcv_search.nested_adaptive_search）结构不兼容，弃用；
本模块在嵌套搜索结果之上做**管线步骤级**消融：

    基准 : 复用 nested_adaptive_search 已产出的每折最优参数（**不重搜**，
        消融的是"组件存在与否的边际贡献"，口径干净、成本 ≈ 臂数 × 一次
        压测；重搜会混入参数自适应噪声，且计算量不可行）。
    臂   : 固定每折参数不变，按步骤名开关 pipeline（剔除 extremes /
        nondomin / correlate 等），复用 adaptive_multi_paths 的 CPCV
        多路径压测（通过其 pipeline_builder 参数注入变体构建器）。
    对比 : 各臂与基准臂的 OOS 多路径绩效分布对照表（mean/median/5%/95%），
        (折, 路径) 级样本的差异显著性可用 cpcv_analysis.boot_diff_ci 自取。

与 cpcv_search_base.build_pipeline 的关系：本模块自带 build_variant_pipeline
（同构副本 + 步骤开关），**不修改** cpcv_search_base / wf_cpcv_search 的
pipeline 本体 —— 搜索语义不受消融影响，变更亦不波及其他消费方。

用法（notebook 会话内，fold_results 来自 nested_adaptive_search）::

    from wf_cpcv_ablation import ARMS, ablate
    paths_kwargs = dict(test_size=252, train_size=1008, n_test_folds=4,
                        n_jobs=8, inner_purged_size=2, inner_embargo_size=2)
    frames, summary = ablate(X, fold_results, paths_kwargs=paths_kwargs)
    print(summary)                      # 臂 × 指标汇总表
    # frames["no_extremes"] 等为 (折,路径) 级绩效表，可绘图/显著性检验

CLI（冒烟/小预算验证用，完整消融建议在 notebook 会话内复用已完成的搜索）::

    G:\\Anaconda3\\envs\\ml4t\\python.exe wf_cpcv_ablation.py --n-trials 3 --tail-days 1200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from skfolio import ExtraRiskMeasure
from skfolio.optimization import EqualWeighted
from skfolio.pre_selection import (
    DropCorrelated,
    DropZeroVariance,
    SelectComplete,
    SelectNonDominated,
)

import cpcv_search_base as base
from Pre_selection import SelectKExtremes
from wf_cpcv_search import (
    adaptive_multi_paths,
    derive_outer_window,
    load_nested_config,
    nested_adaptive_search,
)

# Windows GBK 控制台打印中文安全兜底（与同目录脚本一致）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# 消融臂定义：drops = 从 pipeline 剔除的步骤名（tuple 顺序无意义）
ARMS = {
    "full": {"drops": (), "label": "全管线（基准）"},
    "no_extremes": {"drops": ("extremes",),
                    "label": "无峰度预筛（剔除 extremes 步骤）"},
    "no_nondomin": {"drops": ("nondomin",),
                    "label": "无非支配筛选（剔除 nondomin 步骤）"},
    "no_correlate": {"drops": ("correlate",),
                     "label": "无去相关（剔除 correlate 步骤）"},
    "no_extremes_correlate": {"drops": ("extremes", "correlate"),
                              "label": "剔除 extremes + correlate"},
}

DEFAULT_METRICS = ["annualized_mean", "annualized_sharpe_ratio",
                   "max_drawdown", "skew", "kurtosis"]


def build_variant_pipeline(params: dict, drops=()) -> Pipeline:
    """按一组参数构建筛选+等权 pipeline，支持按步骤名剔除（消融臂）。

    与 cpcv_search_base.build_pipeline / wf_cpcv_search.build_pipeline 同构
    （complete→variance→extremes→nondomin→correlate→optimization），仅
    多一个 drops 开关；fitness 名称解析复用 cpcv_search_base 注册表。
    被剔除步骤的对应参数（如 extremes__k）随步骤失效，原样忽略。
    """
    fm = params["nondomin__fitness_measures"]
    if isinstance(fm, str):
        fm = base.FITNESS_MEASURES[fm]
    steps = [
        ("complete", SelectComplete(drop_assets_with_internal_nan=False)),
        ("variance", DropZeroVariance(threshold=1e-8)),
        ("extremes", SelectKExtremes(
            k=params["extremes__k"], measure=ExtraRiskMeasure.KURTOSIS,
            highest=False)),
        ("nondomin", SelectNonDominated(
            min_n_assets=params["nondomin__min_n_assets"],
            threshold=params["nondomin__threshold"],
            fitness_measures=fm)),
        ("correlate", DropCorrelated(threshold=params["correlate__threshold"])),
        ("optimization", EqualWeighted()),
    ]
    return Pipeline([s for s in steps if s[0] not in drops])


def arm_fold_paths(X, folds, drops=(), paths_kwargs=None):
    """单臂多路径压测：固定每折最优参数 + 步骤开关。

    复用 wf_cpcv_search.adaptive_multi_paths（其 pipeline_builder 注入
    变体构建器，本体与折对齐逻辑不变）；drops=() 时为基准臂。
    paths_kwargs 未给或缺少窗口键时，从 folds 推导外层窗口（与嵌套搜索
    折位 1:1 对齐，防静默错位）；显式键优先。
    """
    w = derive_outer_window(folds)       # test/train/purged/reduce_test=搜索口径
    w.update(dict(paths_kwargs or {}))   # 用户显式键优先
    return adaptive_multi_paths(
        X, folds,
        **{**w, "pipeline_builder": (lambda p: build_variant_pipeline(p, drops))})


def ablate(X, folds, arms=None, paths_kwargs=None, metrics=DEFAULT_METRICS):
    """执行固定参数消融并汇总（checklist 4.3：逐模块边际贡献对照）。

    Parameters
    ----------
    X : pd.DataFrame，与 nested_adaptive_search 同口径的收益数据
    folds : list，nested_adaptive_search 返回结果（每折 params 复用，不重搜）
    arms : list[str] | None，臂名（ARMS 键）；None = 全部臂
    paths_kwargs : dict | None，透传给 adaptive_multi_paths 的其余参数
        （n_test_folds / n_jobs 等）；窗口四键（test_size / train_size /
        outer_purged_size / outer_reduce_test）未给时自动从 folds 推导
        （折位对齐，防静默错位），显式键优先
    metrics : list[str]，绩效指标（skfolio Portfolio/MPP 属性名）

    Returns
    -------
    (frames, summary) : frames = {臂名: DataFrame(行=(折,路径), 列=metrics)}；
        summary = DataFrame(行=臂, 列=各指标 mean/median/5%/95%)，臂序=full 优先
    """
    from cpcv_analysis import fold_paths_frame

    arms = list(arms) if arms is not None else list(ARMS)
    frames, rows = {}, []
    for name in arms:
        fp = arm_fold_paths(X, folds, drops=ARMS[name]["drops"],
                            paths_kwargs=paths_kwargs)
        df = fold_paths_frame(fp, metrics)
        frames[name] = df
        s = {"臂": name, "说明": ARMS[name]["label"], "样本数": len(df)}
        for m in metrics:
            col = df[m]
            s[f"{m}_mean"] = col.mean()
            s[f"{m}_median"] = col.median()
            s[f"{m}_p5"] = col.quantile(0.05)
            s[f"{m}_p95"] = col.quantile(0.95)
        rows.append(s)
    summary = pd.DataFrame(rows).set_index("臂")
    return frames, summary


# ---------------------------------------------------------------------------
# CLI（冒烟/小预算；完整消融建议在 notebook 会话复用已完成的 fold_results）
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="WF+CPCV 嵌套流程固定参数消融（冒烟级 CLI）")
    parser.add_argument("--n-trials", type=int, default=3,
                        help="每折内层 optuna trial 数（小预算冒烟；完整流程请先在 notebook 跑 nested_adaptive_search 再调 ablate()）")
    parser.add_argument("--tail-days", type=int, default=None,
                        help="只取数据尾部 N 天控制折数（None=全量）")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--data", choices=("X", "X_net"), default="X",
                        help="搜索与消融的数据口径（默认 X，与 WF+CPCV+PS_252 一致）")
    parser.add_argument("--n-jobs", type=int, default=2, help="trial 级并行度")
    parser.add_argument("--cv-n-jobs", type=int, default=1, help="单 trial 内 CPCV 并行度")
    args = parser.parse_args(argv)

    import walkforward_parameter_search as wfps
    from log_result import init_logger, log_print, log_result

    init_logger("wf_cpcv_ablation_smoke")
    X_all, X_net, info = wfps.load_data()
    X_use = X_net if args.data == "X_net" else X_all
    if args.tail_days:
        X_use = X_use.tail(args.tail_days)
    log_print(f"数据: {args.data} | 行数={len(X_use)} | inf={info['inf_cols']} | 异常剔除 X={len(info['bad_raw'])} X_net={len(info['bad_net'])}",
              section="数据加载", echo=True)

    cfg = load_nested_config()
    search_kwargs = {**cfg["search_kwargs"], "n_trials": args.n_trials,
                     "n_jobs": args.n_jobs, "cv_n_jobs": args.cv_n_jobs}
    log_print(search_kwargs, section="搜索配置（冒烟）", echo=False)
    folds = nested_adaptive_search(X_use, space=cfg["space"], **search_kwargs)

    frames, summary = ablate(X_use, folds, arms=args.arms)
    print("\n==== 固定参数消融汇总 ====")
    print(summary.to_string())
    log_result(summary, section="固定参数消融汇总")


if __name__ == "__main__":
    main()
