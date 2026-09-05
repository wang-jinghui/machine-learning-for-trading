# -*- coding: utf-8 -*-
"""Extremes 消融实验：126/504 切分下只切 SelectKExtremes 一档的严格对照。

目的：回答"峰度预筛（保留 KURTOSIS 最低的 k 档资产）是否有净效应"。
此前两份日志（WalkForward+ParameterSearch.log / WalkForward+ParameterSearch+Extremes.log）
的轮次混杂了 corr 空间（无 extremes 组 0.1~0.3 vs k 组 0.1~0.4）、fitness 集、
k 口径（整数 200 vs 比例 0.1~0.5）等差异，无法直接读出 extremes 的净效应。
本脚本固定除 extremes 外的全部条件，仅切换 5 个档位：

    none : 管线中无 extremes 步骤（不剔除，对照档）
    k200 : SelectKExtremes(k=200, KURTOSIS, highest=False)  整数保留 200 只
    k0.1 : 同上但 k=0.1（保留峰度最低 10%）
    k0.3 : 同上但 k=0.3（保留 30%）
    k0.5 : 同上但 k=0.5（保留 50%）

固定条件（全部档位一致）：
    * 数据：X（原始收益口径），hs_funds_prices.parquet（load_data 同 walkforward_parameter_search）
    * CV：WalkForward(test_size=126, train_size=504, purged_size=1,
      reduce_test=True, expand_train=False)（默认；--test-size/--train-size 可换切分）
    * 目标函数：--objective 指定（默认 annualized_sharpe_ratio；可选 OBJECTIVES 注册表内任意名）
    * 搜索空间：nondomin min_n_assets 5~25 step5 / threshold -0.5~-0.3 step0.1 /
      correlate threshold 0.1~0.4 step0.1 / fitness_measures 10 种 mean-* 组合
    * 搜索：optuna TPE seed=42，patience 连续无改善提前停止
    * pipeline 其余步骤：complete → variance → [extremes] → nondomin →
      correlate → EqualWeighted

多组矩阵（切分 × 目标 × 档位）批量运行见 ablation_matrix.py：自动按优先级排队、
总时限截断、已完成的组自动跳过（断点续跑），每组一个独立日志文件。

每档输出：最优参数、搜索内得分、OOS 滚动预测关键指标；
最后输出 5 档汇总对照表。日志段落风格与 walkforward_parameter_search.py 一致
（数据加载 / 实验配置 / WalkForward 配置 / 参数空间 / 目标函数源码 / 消融档位定义 /
OOS MPP Summary / 汇总对照表），落盘 logs/<log-name>.log（默认 Extremes+Ablation.log）。

用法::

    G:\\Anaconda3\\envs\\ml4t\\python.exe ablation_extremes.py                      # 全量 5 档
    G:\\Anaconda3\\envs\\ml4t\\python.exe ablation_extremes.py --n-trials 3         # 冒烟验证
    G:\\Anaconda3\\envs\\ml4t\\python.exe ablation_extremes.py --arms none k0.5
    G:\\Anaconda3\\envs\\ml4t\\python.exe ablation_extremes.py --test-size 63 --train-size 504 --objective asr_mdd_skew \\
        --log-name Extremes+Ablation+63_504_skew                                     # 换切分 + 目标 + 日志名
    G:\\Anaconda3\\envs\\ml4t\\python.exe ablation_matrix.py --max-hours 15         # 矩阵批量（见该脚本）
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from skfolio import ExtraRiskMeasure
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import EqualWeighted
from skfolio.pre_selection import (
    DropCorrelated,
    DropZeroVariance,
    SelectComplete,
    SelectNonDominated,
)

from log_result import init_logger, log_print, log_result
from Pre_selection import SelectKExtremes

import walkforward_parameter_search as wfps  # 复用数据加载 / 搜索工具 / 注册表
from walkforward_parameter_search import (
    FITNESS_MEASURES,
    OBJECTIVES,
    StopWhenNoImprovement,
    format_params,
    load_data,
    suggest_from_space,
)

# Windows GBK 控制台打印中文安全兜底
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_NAME = "Extremes+Ablation"
OBJECTIVE_NAME = "annualized_sharpe_ratio"  # 与 8/31 日志 k=200 组目标一致

# 与 8/31 日志 FITNESS_MEASURES 打印一致的 10 种 mean-* 组合（不含 ratio 组合）
ARM_FITNESS_CHOICES = [
    "mean-variance",
    "mean-variance-maxdd",
    "mean-variance-avgdd",
    "mean-semideviation-avgdd",
    "mean-mad-maxdd",
    "mean-mad-avgdd",
    "mean-mad-maxdd-cvar",
    "mean-mad-maxdd-cvar-sharpe",
    "mean-mad-avgdd-cvar",
    "mean-mad-avgdd-cvar-sharpe",
]

# 固定搜索空间（4 个自由参数，extremes 不进搜索、按档位固定）
SPACE = {
    "nondomin__min_n_assets": {"low": 5, "high": 25, "step": 5},
    "nondomin__threshold": {"low": -0.5, "high": -0.3, "step": 0.1},
    "correlate__threshold": {"low": 0.1, "high": 0.4, "step": 0.1},
    "nondomin__fitness_measures": ARM_FITNESS_CHOICES,
}

# 消融档位定义：k=None 表示管线中无 extremes 步骤
ARMS = {
    "none": {"label": "无 extremes（不剔除，对照档）", "k": None},
    "k200": {"label": "k=200 整数（保留峰度最低 200 只）", "k": 200},
    "k0.1": {"label": "k=0.1 比例（保留峰度最低 10%）", "k": 0.1},
    "k0.3": {"label": "k=0.3 比例（保留峰度最低 30%）", "k": 0.3},
    "k0.5": {"label": "k=0.5 比例（保留峰度最低 50%）", "k": 0.5},
}


def build_arm_pipeline(params: dict, k: int | float | None) -> Pipeline:
    """按一组具体参数 + 档位 k 构建筛选 + 等权组合 pipeline。

    k=None 时省略 extremes 步骤（no-extremes 对照档）；
    其余与 walkforward_parameter_search.build_pipeline 完全一致。
    """
    fm = params["nondomin__fitness_measures"]
    if isinstance(fm, str):
        fm = FITNESS_MEASURES[fm]
    steps = [
        ("complete", SelectComplete(drop_assets_with_internal_nan=False)),
        ("variance", DropZeroVariance(threshold=1e-8)),
    ]
    if k is not None:
        steps.append(
            ("extremes", SelectKExtremes(
                k=k, measure=ExtraRiskMeasure.KURTOSIS, highest=False))
        )
    steps += [
        ("nondomin", SelectNonDominated(
            min_n_assets=params["nondomin__min_n_assets"],
            threshold=params["nondomin__threshold"],
            fitness_measures=fm,
        )),
        ("correlate", DropCorrelated(threshold=params["correlate__threshold"])),
        ("optimization", EqualWeighted()),
    ]
    return Pipeline(steps)


def run_arm_search(X, cv, k, n_trials=500, n_jobs=12, cv_n_jobs=4,
                   patience=100, min_delta=1e-4, seed=42,
                   objective_name: str = OBJECTIVE_NAME):
    """单档位 optuna 搜索（与 wfps.run_optuna 同参数语义，仅 pipeline 构建不同）。"""
    objective_fn = OBJECTIVES[objective_name]

    def objective(trial):
        params = suggest_from_space(trial, SPACE)
        model = build_arm_pipeline(params, k)
        pred = cross_val_predict(model, X, cv=cv, n_jobs=cv_n_jobs)
        return objective_fn(pred)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        callbacks=[StopWhenNoImprovement(patience=patience, min_delta=min_delta)],
    )
    return format_params(study.best_params), study.best_value


def arm_evaluate_oos(X, params: dict, k, cv, n_jobs: int = 4):
    """基于最优参数做 OOS 滚动预测，返回 MultiPeriodPortfolio。"""
    model = build_arm_pipeline(params, k)
    return cross_val_predict(model, X, cv=cv, n_jobs=n_jobs)


def extract_oos_metrics(pred) -> dict:
    """从 OOS MultiPeriodPortfolio 提取对照表所需的关键指标（小数格式）。"""
    n_assets = np.mean([p.n_assets for p in pred.portfolios])
    return {
        "年化收益": pred.annualized_mean,
        "年化波动": pred.annualized_standard_deviation,
        "年化SR": pred.annualized_sharpe_ratio,
        "最大回撤": pred.max_drawdown,
        "偏度": pred.skew,
        "峰度": pred.kurtosis,
        "平均持仓数": n_assets,
        "折数": len(pred.portfolios),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extremes 消融实验（同切分严格对照）")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS),
                        help="要跑的档位（默认全部）")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="每档 optuna trial 数（默认 = 参数空间网格规模的一半）")
    parser.add_argument("--n-jobs", type=int, default=12, help="trial 级并行度")
    parser.add_argument("--cv-n-jobs", type=int, default=4, help="单 trial 内 cross_val_predict 并行度")
    parser.add_argument("--patience", type=int, default=100, help="optuna 连续无改善提前停止阈值")
    parser.add_argument("--seed", type=int, default=42, help="optuna TPE 随机种子")
    parser.add_argument("--objective", choices=list(OBJECTIVES), default=OBJECTIVE_NAME,
                        help="目标函数注册名（默认 annualized_sharpe_ratio）")
    parser.add_argument("--test-size", type=int, default=126, help="WalkForward 测试段长度（交易日）")
    parser.add_argument("--train-size", type=int, default=504, help="WalkForward 训练段长度（交易日）")
    parser.add_argument("--data-file", type=Path, default=wfps.DEFAULT_DATA, help="基金价格 parquet 路径")
    parser.add_argument("--log-name", default=None,
                        help="日志文件名（logs/ 下，不含扩展名；默认 Extremes+Ablation）")
    parser.add_argument("--no-oos", action="store_true", help="跳过最优参数后的 OOS 滚动预测评估")
    args = parser.parse_args(argv)

    init_logger(args.log_name or DEFAULT_LOG_NAME)

    X, X_net, info = load_data(args.data_file)
    log_print(
        f"数据加载完成: {args.data_file.name} | 行数={len(X)} 资产数={len(X.columns)} | "
        f"inf列={info['inf_cols']} | 剔除异常列 X={len(info['bad_raw'])} X_net={len(info['bad_net'])}",
        section="数据加载",
        echo=True,
    )

    grid_size = wfps.compute_grid_size(SPACE)
    n_trials = args.n_trials or grid_size // 2
    log_print(
        f"数据: X（原始收益） | 搜索方式: optuna（TPE seed={args.seed}, n_trials={n_trials}）"
        f" | 目标函数: {args.objective}",
        section="实验配置",
        echo=True,
    )
    cv = WalkForward(
        test_size=args.test_size, train_size=args.train_size,
        purged_size=1, reduce_test=True, expand_train=False,
    )
    log_print(cv, section="WalkForward 配置", echo=False)
    log_print(
        f"\n{wfps.format_param_space(SPACE)}\n网格规模 = {grid_size}",
        section="参数空间",
        echo=True,
    )
    log_print(
        inspect.getsource(OBJECTIVES[args.objective]),
        section=f"目标函数: {args.objective}",
        echo=False,
    )
    arm_lines = "\n".join(
        f"  {name:6s}: k={arm['k']!s:5s} | {arm['label']}" for name, arm in ARMS.items()
    )
    log_print(f"本次档位（--arms 过滤）:\n{arm_lines}", section="消融档位定义", echo=True)

    summary_rows = {}
    for arm_name in args.arms:
        arm = ARMS[arm_name]
        print(f"\n>>> 开始档位 {arm_name} 的 optuna 搜索"
              f"（{arm['label']}；网格规模 {grid_size}，n_trials={n_trials}, "
              f"n_jobs={args.n_jobs}, seed={args.seed}）...")
        log_print(
            f"k = {arm['k']} | {arm['label']}",
            section=f"消融档位: {arm_name}", echo=False,
        )

        best_params, best_score = run_arm_search(
            X, cv, arm["k"],
            n_trials=n_trials, n_jobs=args.n_jobs, cv_n_jobs=args.cv_n_jobs,
            patience=args.patience, seed=args.seed, objective_name=args.objective,
        )
        print(f">>> 最优参数: {best_params}")
        print(f">>> 最优得分: {best_score:.6f}")
        log_print(best_params, section=f"[{arm_name}] 最优参数", echo=False)
        log_print(f"{best_score:.6f}", section=f"[{arm_name}] 最优得分", echo=False)

        row = {"档位": arm_name, "最优参数": best_params, "搜索内得分": best_score}
        if not args.no_oos:
            print(f">>> 基于最优参数进行 OOS 滚动预测（X 口径，档位 {arm_name}）...")
            oos_pred = arm_evaluate_oos(X, best_params, arm["k"], cv,
                                        n_jobs=args.cv_n_jobs)
            mpt_summary = oos_pred.summary()
            print(mpt_summary.to_string())
            log_print(mpt_summary, section=f"[{arm_name}] OOS MPP Summary", echo=False)
            row.update(extract_oos_metrics(oos_pred))
        summary_rows[arm_name] = row

    # 汇总对照表
    if len(summary_rows) > 1:
        df = []
        for name, row in summary_rows.items():
            rec = {"档位": name, "说明": ARMS[name]["label"]}
            rec.update(row)
            df.append(rec)
        table = _to_table(df)
        print("\n==== 汇总对照 ====")
        print(table.to_string(index=False))
        log_result(table, section="汇总对照表")


def _to_table(rows: list[dict]) -> "pd.DataFrame":
    """汇总行 → 百分数字段格式化的 DataFrame（读数用）。"""
    import pandas as pd

    df = pd.DataFrame(rows)
    for col in ("年化收益", "年化波动", "最大回撤", "偏度", "峰度"):
        if col in df.columns:
            df[col] = (df[col] * 100).map(lambda v: f"{v:.2f}%")
    if "年化SR" in df.columns:
        df["年化SR"] = df["年化SR"].map(lambda v: f"{v:.3f}")
    if "平均持仓数" in df.columns:
        df["平均持仓数"] = df["平均持仓数"].map(lambda v: f"{v:.1f}")
    if "搜索内得分" in df.columns:
        df["搜索内得分"] = df["搜索内得分"].map(lambda v: f"{v:.6f}")
    return df


if __name__ == "__main__":
    main()
