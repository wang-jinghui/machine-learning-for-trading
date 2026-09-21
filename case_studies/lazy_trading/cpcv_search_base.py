# -*- coding: utf-8 -*-
"""CPCV 嵌套搜索体系共享基础：pipeline / fitness 注册表 / 参数空间采样 / 早停。

本模块是 wf_cpcv_search.py（及 cpcv 体系后续模块）的共享底座，内容
**fork 自 walkforward_parameter_search.py**（同名符号原样搬运、零改动）：

* `FITNESS_MEASURES`      fitness 名称 → skfolio measure 组合注册表
* `build_pipeline`        筛选+等权 pipeline（complete→variance→extremes→
                         nondomin→correlate→optimization）
* `suggest_from_space`    参数空间节点 → optuna trial 采样
* `StopWhenNoImprovement` optuna 提前停止回调

解耦背景：wf_cpcv_search 原本直接 import walkforward_parameter_search，
导致每次加载连带执行该脚本的全部顶层代码（polars/optuna/log_result 等）
且两侧修改互相牵制。现改为引用本模块——walkforward_parameter_search.py
保持不动，cpcv 体系侧改动不再波及 WF 搜索脚本，反之亦然。

同步策略（重要）：本模块内容与 walkforward_parameter_search 同名部分是
**各自独立的两份副本**（非单源）。修改任一側的注册表/pipeline 时，如
期望两侧语义一致，需手动同步另一侧；cpcv 配置 [param_space] 的 choice
名称集合同样与 WF 侧 TOML 逐键同构（嵌套验证契约），新增/改名需两侧
同步。刻意分叉（仅 cpcv 侧使用新名称）时无同步义务，但该名称不得出现
在 WF 侧 TOML 的 choice 中。
"""

from __future__ import annotations

import ast
import inspect
import sys

from sklearn import set_config
from sklearn.pipeline import Pipeline
from skfolio import PerfMeasure, RiskMeasure, RatioMeasure, ExtraRiskMeasure
from skfolio.optimization import EqualWeighted
from skfolio.pre_selection import (
    DropCorrelated,
    DropZeroVariance,
    SelectComplete,
    SelectNonDominated,
)

from Pre_selection import SelectKExtremes

# Windows GBK 控制台打印中文安全兜底（与 walkforward_parameter_search 顶层一致）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# skfolio pandas 输出全局配置：与 walkforward_parameter_search 顶层一致。
# 依赖方勿移除——并行 worker 内 DropZeroVariance 校验依赖 pandas set_output。
set_config(transform_output="pandas")

# ---------------------------------------------------------------------------
# 注册表：fitness_measures 名称 → skfolio measure 组合（SelectNonDominated 用）
# 配置文件中 "nondomin__fitness_measures" 的候选值只写名称，构建 pipeline 时在此解析。
# ---------------------------------------------------------------------------
FITNESS_MEASURES = {
    # MEAN + RiskMeasure 组合
    "mean-variance": [PerfMeasure.MEAN, RiskMeasure.VARIANCE],
    "mean-variance-kurt": [PerfMeasure.MEAN, RiskMeasure.VARIANCE, ExtraRiskMeasure.KURTOSIS],
    "mean-variance-cvar": [PerfMeasure.MEAN, RiskMeasure.VARIANCE, RiskMeasure.CVAR],
    "mean-semivariance": [PerfMeasure.MEAN, RiskMeasure.SEMI_VARIANCE],
    "mean-semivariance-cvar": [PerfMeasure.MEAN, RiskMeasure.SEMI_VARIANCE, RiskMeasure.CVAR],
    "mean-semivariance-avgdd": [PerfMeasure.MEAN, RiskMeasure.SEMI_VARIANCE, RiskMeasure.AVERAGE_DRAWDOWN],
    "mean-variance-maxdd": [PerfMeasure.MEAN, RiskMeasure.VARIANCE, RiskMeasure.MAX_DRAWDOWN],
    "mean-variance-avgdd": [PerfMeasure.MEAN, RiskMeasure.VARIANCE, RiskMeasure.AVERAGE_DRAWDOWN],
    "mean-variance-maxdd-kurt": [PerfMeasure.MEAN, RiskMeasure.VARIANCE, RiskMeasure.MAX_DRAWDOWN, ExtraRiskMeasure.KURTOSIS],
    "mean-variance-avgdd-kurt": [PerfMeasure.MEAN, RiskMeasure.VARIANCE, RiskMeasure.AVERAGE_DRAWDOWN, ExtraRiskMeasure.KURTOSIS],
    "mean-semideviation-avgdd": [PerfMeasure.MEAN, RiskMeasure.SEMI_DEVIATION, RiskMeasure.AVERAGE_DRAWDOWN],
    "mean-mad": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION],
    "mean-mad-cvar": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.CVAR],
    "mean-mad-maxdd": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.MAX_DRAWDOWN],
    "mean-mad-avgdd": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.AVERAGE_DRAWDOWN],
    "mean-mad-avgdd-kurt": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.AVERAGE_DRAWDOWN, ExtraRiskMeasure.KURTOSIS],
    "mean-mad-maxdd-cvar": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.MAX_DRAWDOWN, RiskMeasure.CVAR],
    "mean-mad-maxdd-cvar-kurt": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.MAX_DRAWDOWN, RiskMeasure.CVAR, ExtraRiskMeasure.KURTOSIS],
    "mean-mad-maxdd-kurt": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.MAX_DRAWDOWN, ExtraRiskMeasure.KURTOSIS],
    "mean-mad-maxdd-cvar-sharpe": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.MAX_DRAWDOWN, RiskMeasure.CVAR, RatioMeasure.SHARPE_RATIO],
    "mean-mad-avgdd-cvar": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.AVERAGE_DRAWDOWN, RiskMeasure.CVAR],
    "mean-mad-avgdd-cvar-kurt": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.AVERAGE_DRAWDOWN, RiskMeasure.CVAR, ExtraRiskMeasure.KURTOSIS],
    "mean-mad-avgdd-cvar-sharpe": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.AVERAGE_DRAWDOWN, RiskMeasure.CVAR, RatioMeasure.SHARPE_RATIO],
    "mean-mad-maxdd-avgdd-cvar": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.MAX_DRAWDOWN, RiskMeasure.AVERAGE_DRAWDOWN, RiskMeasure.CVAR],
    "mean-mad-kurt-avgdd-cvar": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, ExtraRiskMeasure.KURTOSIS, RiskMeasure.AVERAGE_DRAWDOWN, RiskMeasure.CVAR],
    # Ratio 组合
    "calmar-cvar": [RatioMeasure.CALMAR_RATIO, RiskMeasure.CVAR],
    "calmar-variance": [RatioMeasure.CALMAR_RATIO, RiskMeasure.VARIANCE],
    "calmar-semivariance": [RatioMeasure.CALMAR_RATIO, RiskMeasure.SEMI_VARIANCE],
    "sharpe-cvar": [RatioMeasure.SHARPE_RATIO, RiskMeasure.CVAR],
    "sharpe-maxdd": [RatioMeasure.SHARPE_RATIO, RiskMeasure.MAX_DRAWDOWN],
    "sharpe-avgdd": [RatioMeasure.SHARPE_RATIO, RiskMeasure.AVERAGE_DRAWDOWN],
    "sortino-cvar": [RatioMeasure.SORTINO_RATIO, RiskMeasure.CVAR],
    "sortino-variance": [RatioMeasure.SORTINO_RATIO, RiskMeasure.VARIANCE],
    "sortino-semivariance": [RatioMeasure.SORTINO_RATIO, RiskMeasure.SEMI_VARIANCE],
}


# ---------------------------------------------------------------------------
# 参数空间 schema 转换：参数空间节点 → optuna trial 采样
# ---------------------------------------------------------------------------
def suggest_from_space(trial, space: dict) -> dict:
    """参数空间 → optuna trial 采样（range→suggest_int/float，choice→suggest_categorical）。"""
    params = {}
    for name, node in space.items():
        if isinstance(node, list):
            params[name] = trial.suggest_categorical(name, list(node))
        elif all(isinstance(node[k], int) for k in ("low", "high", "step")):
            params[name] = trial.suggest_int(name, node["low"], node["high"], step=node["step"])
        else:
            params[name] = trial.suggest_float(name, node["low"], node["high"], step=node["step"])
    return params


def build_pipeline(params: dict) -> Pipeline:
    """用一组具体参数构建筛选 + 等权组合 pipeline（与 walkforward 侧同构）。

    params 中的 "nondomin__fitness_measures" 支持两种形式：注册表名称
    （optuna 采样结果）或 measure 组合列表（sklearn 搜索器 set_params 值）。
    """
    fm = params["nondomin__fitness_measures"]
    if isinstance(fm, str):
        fm = FITNESS_MEASURES[fm]
    return Pipeline([
        # 严格模式：剔除所有包含缺失值的资产（前导/末端/内部一律剔除）。
        # 搜索（CPCV 各路径 fit）与评估/应用（各折 fit）的池口径由
        # SelectComplete 统一保证：fit 可见列 = 完整资产集，fit/predict
        # 同一掩码，无需外层人工池过滤。
        ("complete", SelectComplete(drop_assets_with_internal_nan=True)),
        ("variance", DropZeroVariance(threshold=1e-8)),
        ("extremes", SelectKExtremes(
            k=params["extremes__k"], measure=ExtraRiskMeasure.KURTOSIS, highest=False)),
        ("nondomin", SelectNonDominated(
            min_n_assets=params["nondomin__min_n_assets"],
            threshold=params["nondomin__threshold"],
            fitness_measures=fm,
        )),
        ("correlate", DropCorrelated(threshold=params["correlate__threshold"])),
        ("optimization", EqualWeighted()),
    ])


# ---------------------------------------------------------------------------
# pipeline 构建源码快照（日志用）：自动跟随 build_pipeline 的实际定义
# ---------------------------------------------------------------------------
def build_pipeline_src() -> str:
    """返回 build_pipeline 中 return 之后的 Pipeline 定义源码（pipeline 状态快照）。

    用途：日志记录当前 pipeline 的步骤结构与参数配置（如实验日志的
    "pipeline 状态"段）。通过 ast 定位 build_pipeline 源码中位置最靠后的
    return 语句，截取其后的 Pipeline([...]) 文本 —— 调整 pipeline 的步骤
    或参数后本函数自动跟随新定义，无需手动维护日志文本。

    Notes
    -----
    依赖源码文件可读（inspect.getsource）：交互式环境（REPL / notebook
    单元格内联定义、pyc 分发）下会抛错，仅适用于模块文件内的常规定义。

    Returns
    -------
    str : "Pipeline([...])" 源码片段（保留原缩进），如::

        Pipeline([
            ("complete", SelectComplete(drop_assets_with_internal_nan=True)),
            ...
            ("optimization", EqualWeighted()),
        ])
    """
    src = inspect.getsource(build_pipeline)
    tree = ast.parse(src)
    func = tree.body[0]
    # 取源码位置最靠后的 return：pipeline 构建语句在函数末尾，即便未来插入
    # 提前 return 的校验分支，也不会截错目标
    ret = max((node for node in ast.walk(func) if isinstance(node, ast.Return)),
              key=lambda node: node.lineno)
    return ast.get_source_segment(src, ret.value)


# ---------------------------------------------------------------------------
# Optuna 提前停止
# ---------------------------------------------------------------------------
class StopWhenNoImprovement:
    """Optuna 提前停止：连续 patience 次 trial 无有效改善（> min_delta）即停止，静默不输出。"""

    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = None
        self.no_improve_count = 0

    def __call__(self, study, trial):
        current_value = study.best_value
        if self.best_value is None:
            self.best_value = current_value
            return
        if current_value - self.best_value > self.min_delta:
            self.best_value = current_value
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
        if self.no_improve_count >= self.patience:
            study.stop()
