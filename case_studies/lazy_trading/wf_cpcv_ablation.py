# -*- coding: utf-8 -*-
"""外层WF+内层CPCV嵌套流程的固定参数消融模块（checklist 4.3）。

定位：ablation_extremes.py 面向"简单 WF 单层 + 全局搜索"流程，与本目录
嵌套自适应流程（wf_cpcv_search.nested_adaptive_search）结构不兼容，弃用；
本模块在嵌套搜索结果之上做**管线步骤级**消融：

    基准 : 复用 nested_adaptive_search 已产出的每折最优参数（**不重搜**，
        消融的是"组件存在与否的边际贡献"，口径干净、成本 ≈ 臂数 × 一次
        压测；重搜会混入参数自适应噪声，且计算量不可行）。
    臂   : 固定每折参数不变，按步骤名开关 pipeline（剔除 extremes /
        nondomin / correlate 等），逐折普通 OOS 单测（无内层 CPCV 展开，
        与部署语义一致），所有折 test 段拼接为一条完整 OOS 路径。
    对比 : 各臂与基准臂的完整 OOS 路径绩效对照表（臂 × 指标）。

与 cpcv_search_base.build_pipeline 的关系：本模块自带 build_variant_pipeline
（同构副本 + 步骤开关），**不修改** cpcv_search_base / wf_cpcv_search 的
pipeline 本体 —— 搜索语义不受消融影响，变更亦不波及其他消费方。

用法（notebook 会话内，fold_results 来自 nested_adaptive_search）::

    from wf_cpcv_ablation import ARMS, ablate
    paths, summary = ablate(X, fold_results)   # 每臂一条完整 OOS 路径（全折拼接）
    print(summary)                      # 臂 × 指标汇总表
    # paths["no_extremes"] 等为 MultiPeriodPortfolio，可画累计收益/分年统计

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
from skfolio import ExtraRiskMeasure, MultiPeriodPortfolio
from skfolio.optimization import EqualWeighted
from skfolio.pre_selection import (
    DropCorrelated,
    DropZeroVariance,
    SelectComplete,
    SelectNonDominated,
)

import cpcv_search_base as base
from Pre_selection import SelectKExtremes
from wf_cpcv_robustness import derive_wf_kwargs
from wf_cpcv_search import (
    load_nested_config,
    nested_adaptive_search,
    run_params_on_fold,
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


def arm_oos_path(X, folds, drops=(), wf_kwargs=None, purged_size=1,
                 reduce_test=True):
    """单臂完整 OOS 路径：固定每折最优参数 + 步骤开关（drops），逐折普通
    OOS 单测（run_params_on_fold，无内层 CPCV 展开，与部署语义一致），
    所有折的 test 段拼接为一条 MultiPeriodPortfolio；drops=() 时为基准臂。

    wf_kwargs 未给时从 folds 推导折位参数（与嵌套搜索 1:1 对齐）；
    purged_size / reduce_test 默认 1 / True（与搜索侧一致，含缩短尾折），
    须与搜索配置核对。
    """
    if wf_kwargs is None:
        wf_kwargs = derive_wf_kwargs(folds, purged_size=purged_size,
                                     reduce_test=reduce_test)

    def builder(p):
        return build_variant_pipeline(p, drops)

    parts = [run_params_on_fold(X, f["fold"], f["params"],
                                **wf_kwargs, pipeline_builder=builder)[1]
             for f in folds]
    return MultiPeriodPortfolio(parts)


def ablate(X, folds, arms=None, wf_kwargs=None, purged_size=1,
           reduce_test=True, metrics=DEFAULT_METRICS):
    """执行固定参数消融并汇总（checklist 4.3：逐模块边际贡献对照）。

    每臂 = 固定每折最优参数 + 步骤开关，逐折普通 OOS 单测，所有折 test
    段拼接为一条完整 OOS 路径，在完整路径上取 skfolio 指标；臂间对比 =
    "若换用该臂管线，完整 OOS 路径会是什么样"。

    Parameters
    ----------
    X : pd.DataFrame，与 nested_adaptive_search 同口径的收益数据
    folds : list，nested_adaptive_search 返回结果（每折 params 复用，不重搜）
    arms : list[str] | None，臂名（ARMS 键）；None = 全部臂
    wf_kwargs : dict | None，run_params_on_fold 的折位参数（None = 从
        folds 实测段长推导众数窗口；purged/reduce 用下方显式参数）
    metrics : list[str]，绩效指标（skfolio Portfolio/MPP 属性名）

    Returns
    -------
    (paths, summary) : paths = {臂名: MultiPeriodPortfolio}（完整 OOS
        路径本体，供画累计收益/分年统计）；summary = DataFrame（行=臂，
        列 = 折数/天数 + 各指标），臂序 = full 优先
    """
    arms = list(arms) if arms is not None else list(ARMS)
    rows, paths = [], {}
    for name in arms:
        mpp = arm_oos_path(X, folds, drops=ARMS[name]["drops"],
                           wf_kwargs=wf_kwargs, purged_size=purged_size,
                           reduce_test=reduce_test)
        paths[name] = mpp
        s = {"臂": name, "说明": ARMS[name]["label"],
             "折数": len(folds), "天数": len(mpp.returns)}
        for m in metrics:
            s[m] = getattr(mpp, m)
        rows.append(s)
    summary = pd.DataFrame(rows).set_index("臂")
    return paths, summary


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

    paths, summary = ablate(X_use, folds, arms=args.arms,
                            purged_size=cfg["outer_wf"]["purged_size"])
    print("\n==== 固定参数消融汇总 ====")
    print(summary.to_string())
    log_result(summary, section="固定参数消融汇总")


if __name__ == "__main__":
    main()
