# -*- coding: utf-8 -*-
"""WalkForward + 参数搜索脚本：统一 GridSearch / RandomSearch / Optuna 三种搜索方式。

从 WalkForward+ParameterSearch.ipynb 提取并重构，与 notebook 的区别：

* 三种搜索方式共用同一份参数空间定义（TOML 配置文件），不再各自维护
  一套 param_grid / param_distributions / suggest_* 描述；
* --method 指定搜索方式，auto（默认）按参数空间规模自动选择：
  网格规模 <=100 → grid，100<规模<=300 → random，>300 → optuna；
* 目标函数独立注册（OBJECTIVES），grid/random 用 make_scorer 包装，
  optuna 直接调用，目标函数源码与参数空间分开写入实验日志；
* 数据口径可选：X（原始收益）或 X_net（去中性化超额收益），通过
  search.data 配置或 --data 指定。

用法::

    G:\\Anaconda3\\envs\\ml4t\\python.exe walkforward_parameter_search.py
    G:\\Anaconda3\\envs\\ml4t\\python.exe walkforward_parameter_search.py --method optuna --n-trials 200
    G:\\Anaconda3\\envs\\ml4t\\python.exe walkforward_parameter_search.py --config my_space.toml --data X
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import optuna
import polars as pl
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn import set_config
from skfolio import PerfMeasure, RiskMeasure, RatioMeasure, ExtraRiskMeasure
from skfolio.metrics import make_scorer
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import EqualWeighted
from skfolio.pre_selection import (
    DropCorrelated,
    DropZeroVariance,
    SelectComplete,
    SelectNonDominated,
)
from skfolio.preprocessing import prices_to_returns

from log_result import init_logger, log_print
from Pre_selection import SelectKExtremes

# Windows GBK 控制台打印中文安全兜底
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

set_config(transform_output="pandas")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "hs_funds_prices.parquet"
DEFAULT_CONFIG = BASE_DIR / "parameter_search_config.toml"
DEFAULT_LOG_NAME = "WalkForward+ParameterSearch+Extremes"

# 自动选择阈值：网格规模 <= 100 → grid；100 < 规模 <= 300 → random；> 300 → optuna
AUTO_GRID_MAX = 100
AUTO_RANDOM_MAX = 300

# ---------------------------------------------------------------------------
# 注册表：fitness_measures 名称 → skfolio measure 组合（SelectNonDominated 用）
# 配置文件中 "nondomin__fitness_measures" 的候选值只写名称，构建 pipeline 时在此解析。
# ---------------------------------------------------------------------------
FITNESS_MEASURES = {
    # MEAN + RiskMeasure 组合
    "mean-variance": [PerfMeasure.MEAN, RiskMeasure.VARIANCE],
    "mean-variance-maxdd": [PerfMeasure.MEAN, RiskMeasure.VARIANCE, RiskMeasure.MAX_DRAWDOWN],
    "mean-variance-avgdd": [PerfMeasure.MEAN, RiskMeasure.VARIANCE, RiskMeasure.AVERAGE_DRAWDOWN],
    "mean-semideviation-avgdd": [PerfMeasure.MEAN, RiskMeasure.SEMI_DEVIATION, RiskMeasure.AVERAGE_DRAWDOWN],
    "mean-mad-maxdd": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.MAX_DRAWDOWN],
    "mean-mad-avgdd": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.AVERAGE_DRAWDOWN],
    "mean-mad-maxdd-cvar": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.MAX_DRAWDOWN, RiskMeasure.CVAR],
    "mean-mad-maxdd-cvar-sharpe": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.MAX_DRAWDOWN, RiskMeasure.CVAR, RatioMeasure.SHARPE_RATIO],
    "mean-mad-avgdd-cvar": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.AVERAGE_DRAWDOWN, RiskMeasure.CVAR],
    "mean-mad-avgdd-cvar-sharpe": [PerfMeasure.MEAN, RiskMeasure.MEAN_ABSOLUTE_DEVIATION, RiskMeasure.AVERAGE_DRAWDOWN, RiskMeasure.CVAR, RatioMeasure.SHARPE_RATIO],
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
# 注册表：目标函数名称 → 函数（统一签名为 objective(pred) -> float）
# pred 为 cross_val_predict 返回的 MultiPeriodPortfolio；grid/random 通过
# make_scorer 包装，optuna 直接调用。新增目标函数只需在此追加。
# ---------------------------------------------------------------------------
def objective_asr_mdd(pred):
    """年化夏普 − 最大回撤（实验证实 MDD 权重取 2 倍无显著提升，改为一倍）"""
    return pred.annualized_sharpe_ratio - pred.max_drawdown


def objective_asr_mdd_skew(pred):
    """年化夏普 − 最大回撤 + 偏度（偏度自带符号：负偏惩罚、正偏奖励）"""
    return pred.annualized_sharpe_ratio - pred.max_drawdown + pred.skew


def objective_combined_score(pred):
    """年化收益 + 偏度 − 最大回撤 − 峰度/100（负偏度惩罚、正偏度奖励）"""
    return pred.annualized_mean + pred.skew - pred.max_drawdown - pred.kurtosis / 100


def objective_annualized_mean(pred):
    """年化收益"""
    return pred.annualized_mean


def objective_annualized_sharpe_ratio(pred):
    """年化夏普比率"""
    return pred.annualized_sharpe_ratio


OBJECTIVES = {
    "asr_mdd": objective_asr_mdd,
    "asr_mdd_skew": objective_asr_mdd_skew,
    "combined_score": objective_combined_score,
    "annualized_mean": objective_annualized_mean,
    "annualized_sharpe_ratio": objective_annualized_sharpe_ratio,
}


# ---------------------------------------------------------------------------
# 参数空间 schema 与三种搜索方式的转换
# ---------------------------------------------------------------------------
def _decimals(x) -> int:
    """数值的小数位数（用于浮点结果舍入，消除累积误差）。"""
    s = format(x, ".15g")
    return len(s.split(".")[1]) if "." in s else 0


def expand_range(node: dict) -> list:
    """把 range 节点展开为候选值列表（按步数生成，避免浮点端点与累积误差）。"""
    low, high, step = node["low"], node["high"], node["step"]
    if isinstance(low, int) and isinstance(high, int) and isinstance(step, int):
        n = (high - low) // step + 1
        return [low + i * step for i in range(n)]
    n = round((high - low) / step) + 1
    decimals = max(_decimals(x) for x in (low, high, step))
    return [round(low + i * step, decimals) for i in range(n)]


def _resolve_choice(param_name: str, value):
    """choice 候选值解析：fitness_measures 名称 → measure 组合，其余原样。"""
    if param_name == "nondomin__fitness_measures":
        return FITNESS_MEASURES[value]
    return value


def to_grid(space: dict) -> dict:
    """参数空间 → GridSearchCV.param_grid（range 展开为列表，choice 原样）。"""
    grid = {}
    for name, node in space.items():
        if isinstance(node, list):
            grid[name] = [_resolve_choice(name, v) for v in node]
        else:
            grid[name] = expand_range(node)
    return grid


def to_random(space: dict) -> dict:
    """参数空间 → RandomizedSearchCV.param_distributions。

    与 to_grid 展开出完全相同的候选集（sklearn 对列表做离散均匀采样），
    保证三种搜索方式共享同一候选空间、结果可直接对照。
    """
    return to_grid(space)


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


def compute_grid_size(space: dict) -> int:
    """网格规模 = 各参数候选数之积（用于 auto 判定与日志）。"""
    size = 1
    for name, node in space.items():
        size *= len(node) if isinstance(node, list) else len(expand_range(node))
    return size


def format_param_space(space: dict) -> str:
    """参数空间的可读表示（fitness_measures 显示名称），用于日志。"""
    lines = []
    for name, node in space.items():
        if isinstance(node, list):
            lines.append(f"  {name} = choice{node}")
        else:
            lines.append(f"  {name} = range[{node['low']}, {node['high']}] step={node['step']}")
    return "\n".join(lines)


def _fm_name(measures) -> str:
    """measure 组合 → 注册表名称（best_params 反查用，未匹配则原样输出）。"""
    for name, combo in FITNESS_MEASURES.items():
        if combo == measures:
            return name
    return str(measures)


def format_params(params: dict) -> dict:
    """搜索结果参数的可读表示：fitness_measures 的 measure 列表转回名称。"""
    out = {}
    for k, v in params.items():
        out[k] = _fm_name(v) if (k == "nondomin__fitness_measures" and isinstance(v, list)) else v
    return out


def build_pipeline(params: dict) -> Pipeline:
    """用一组具体参数构建筛选 + 等权组合 pipeline（三种搜索方式共用）。

    params 中的 "nondomin__fitness_measures" 支持两种形式：注册表名称
    （optuna 采样结果）或 measure 组合列表（sklearn 搜索器 set_params 值）。
    """
    fm = params["nondomin__fitness_measures"]
    if isinstance(fm, str):
        fm = FITNESS_MEASURES[fm]
    return Pipeline([
        ("complete", SelectComplete(drop_assets_with_internal_nan=False)),
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
# 数据加载（与 notebook 口径一致：基准移出、中性化、异常收益剔除）
# ---------------------------------------------------------------------------
def load_data(data_file: Path = DEFAULT_DATA):
    """加载基金价格并构造 X（原始收益）与 X_net（去中性化超额收益）。"""
    prices = pl.read_parquet(data_file).to_pandas()
    prices = prices.set_index("timestamp").ffill()
    prices = prices[prices.index.year > 2015]
    bench_symbol = "510300.SH"
    bench = prices[bench_symbol]
    prices = prices.drop(columns=[bench_symbol])

    X = prices_to_returns(prices, drop_inceptions_nan=False)
    X_net = X.sub(prices_to_returns(bench.to_frame("bench"), drop_inceptions_nan=False)["bench"], axis=0)

    # Inf 值检查与异常收益剔除（与 notebook 口径一致）
    inf_cols = X_net.columns[np.isinf(X_net).any(axis=0)].tolist()
    bad_net = X_net.columns[(X_net.abs() > 0.25).any()].tolist()
    bad_raw = X.columns[(X.abs() > 0.25).any()].tolist()
    X_net = X_net.drop(columns=bad_net)
    X = X.drop(columns=bad_raw)
    return X, X_net, dict(inf_cols=inf_cols, bad_net=bad_net, bad_raw=bad_raw)


# ---------------------------------------------------------------------------
# 搜索执行器
# ---------------------------------------------------------------------------
def _template_pipeline(space: dict) -> Pipeline:
    """用每个参数的首个候选值构建模板（供 sklearn 搜索器 clone + set_params）。"""
    defaults = {}
    for name, node in space.items():
        if isinstance(node, list):
            defaults[name] = _resolve_choice(name, node[0])
        else:
            defaults[name] = expand_range(node)[0]
    return build_pipeline(defaults)


def run_grid(X, space, cv, objective_fn, n_jobs=8, verbose=1):
    """网格搜索：枚举参数空间全组合（sklearn GridSearchCV）。"""
    searcher = GridSearchCV(
        estimator=_template_pipeline(space),
        cv=cv,
        param_grid=to_grid(space),
        scoring=make_scorer(objective_fn),
        n_jobs=n_jobs,
        verbose=verbose,
        refit=False,
    )
    searcher.fit(X)
    return format_params(searcher.best_params_), searcher.best_score_


def run_random(X, space, cv, objective_fn, n_iter=30, n_jobs=12, seed=42, verbose=1):
    """随机搜索：在网格候选集中均匀采样 n_iter 组（sklearn RandomizedSearchCV）。"""
    searcher = RandomizedSearchCV(
        estimator=_template_pipeline(space),
        cv=cv,
        param_distributions=to_random(space),
        scoring=make_scorer(objective_fn),
        n_jobs=n_jobs,
        verbose=verbose,
        n_iter=n_iter,
        random_state=seed,
        refit=False,
    )
    searcher.fit(X)
    return format_params(searcher.best_params_), searcher.best_score_


class StopWhenNoImprovement:
    """Optuna 提前停止：连续 patience 次 trial 无有效改善（> min_delta）即停止。"""

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
            print(f"\n连续 {self.patience} 次 trial 无有效改善，提前停止。")
            study.stop()


def run_optuna(X, space, cv, objective_fn, n_trials=500, n_jobs=8, cv_n_jobs=4,
               patience=100, min_delta=1e-4, seed=42):
    """贝叶斯优化：TPE 采样（optuna，trial 级并行）。"""
    def objective(trial):
        params = suggest_from_space(trial, space)
        model = build_pipeline(params)
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


def evaluate_oos(X, params: dict, cv, n_jobs: int = 4):
    """基于一组参数（最优参数）构建 pipeline，cross_val_predict 做 OOS 滚动预测。

    Returns
    -------
    MultiPeriodPortfolio : 各折 test 段的滚动组合，调用 .summary() 可得到
        与 notebook "OOS MPP Summary" 一致的指标表。
    """
    model = build_pipeline(params)
    return cross_val_predict(model, X, cv=cv, n_jobs=n_jobs)


# ---------------------------------------------------------------------------
# 配置加载与日志
# ---------------------------------------------------------------------------
def load_config(config_file: Path) -> dict:
    """读取 TOML 配置并校验关键字段。"""
    import tomllib

    with open(config_file, "rb") as f:
        cfg = tomllib.load(f)

    search = cfg["search"]
    assert search["method"] in ("auto", "grid", "random", "optuna"), f"非法 method: {search['method']}"
    assert search["data"] in ("X", "X_net"), f"非法 data: {search['data']}"
    assert search["objective"] in OBJECTIVES, f"未注册的目标函数: {search['objective']}"
    assert cfg["param_space"], "param_space 不能为空"
    return cfg


def auto_method(grid_size: int) -> str:
    """按网格规模自动选择搜索方式：<=100 grid，100<<=300 random，>300 optuna。"""
    if grid_size <= AUTO_GRID_MAX:
        return "grid"
    if grid_size <= AUTO_RANDOM_MAX:
        return "random"
    return "optuna"


def log_experiment_setup(cfg: dict, method: str, grid_size: int, X, X_net, cv, objective_name: str):
    """把实验设置（数据口径、CV、参数空间、目标函数）写入日志。"""
    search = cfg["search"]
    data_name = search["data"]
    log_print(
        f"数据: {data_name}（行数={len(X)}） | 搜索方式: {method}"
        f"（auto 依据: 网格规模 {grid_size}） | 目标函数: {objective_name}",
        section="实验配置",
        echo=True,
    )
    log_print(cv, section="WalkForward 配置", echo=False)
    log_print(f"\n{format_param_space(cfg['param_space'])}\n网格规模 = {grid_size}", section="参数空间", echo=True)
    log_print(inspect.getsource(OBJECTIVES[objective_name]), section=f"目标函数: {objective_name}", echo=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="WalkForward + 参数搜索（grid/random/optuna）")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TOML 配置文件路径")
    parser.add_argument("--method", choices=("auto", "grid", "random", "optuna"), default=None,
                        help="搜索方式（默认 auto：按网格规模自动选择）")
    parser.add_argument("--data", choices=("X", "X_net"), default=None, help="数据口径")
    parser.add_argument("--objective", default=None, help="目标函数注册名")
    parser.add_argument("--n-trials", type=int, default=None, help="optuna trial 数")
    parser.add_argument("--n-iter", type=int, default=None, help="random 采样次数")
    parser.add_argument("--n-jobs", type=int, default=None, help="搜索并行度")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA, help="基金价格 parquet 路径")
    parser.add_argument("--log-name", default=None,
                        help="日志文件名（logs/ 下，不含扩展名；默认取配置 [search].log_name）")
    parser.add_argument("--no-oos", action="store_true", help="跳过最优参数后的 OOS 滚动预测评估")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    search = cfg["search"]
    # CLI 参数覆盖配置文件
    for cli_key, cfg_key in (("method", "method"), ("data", "data"), ("objective", "objective"),
                             ("n_trials", "n_trials"), ("n_iter", "n_iter"), ("n_jobs", "n_jobs"),
                             ("log_name", "log_name")):
        cli_val = getattr(args, cli_key)
        if cli_val is not None:
            search[cfg_key] = cli_val

    init_logger(search.get("log_name") or DEFAULT_LOG_NAME)
    X, X_net, info = load_data(args.data_file)
    log_print(
        f"数据加载完成: {args.data_file.name} | 行数={len(X)} 资产数={len(X.columns)} | "
        f"inf列={info['inf_cols']} | 剔除异常列 X={len(info['bad_raw'])} X_net={len(info['bad_net'])}",
        section="数据加载",
        echo=True,
    )

    space = cfg["param_space"]
    grid_size = compute_grid_size(space)
    method = search["method"] if search["method"] != "auto" else auto_method(grid_size)
    objective_name = search["objective"]

    X_use = X_net if search["data"] == "X_net" else X
    cv = WalkForward(
        test_size=cfg["cv"]["test_size"],
        train_size=cfg["cv"]["train_size"],
        purged_size=cfg["cv"]["purged_size"],
        reduce_test=cfg["cv"]["reduce_test"],
        expand_train=cfg["cv"]["expand_train"],
    )
    log_experiment_setup(cfg, method, grid_size, X, X_net, cv, objective_name)

    objective_fn = OBJECTIVES[objective_name]
    print(f"\n>>> 开始 {method} 搜索（网格规模 {grid_size}，数据 {search['data']}）...")
    if method == "grid":
        best_params, best_score = run_grid(X_use, space, cv, objective_fn, n_jobs=search["n_jobs"], verbose=search["verbose"])
    elif method == "random":
        best_params, best_score = run_random(X_use, space, cv, objective_fn, n_iter=search["n_iter"],
                                             n_jobs=search["n_jobs"], seed=search["seed"], verbose=search["verbose"])
    else:  # optuna
        best_params, best_score = run_optuna(X_use, space, cv, objective_fn, n_trials=search["n_trials"],
                                             n_jobs=search["n_jobs"], cv_n_jobs=search["cv_n_jobs"],
                                             patience=search["patience"], min_delta=search["min_delta"],
                                             seed=search["seed"])

    print(f"\n>>> 最优参数: {best_params}")
    print(f">>> 最优得分: {best_score:.6f}")
    log_print(best_params, section="最优参数", echo=False)
    log_print(f"{best_score:.6f}", section="最优得分", echo=False)

    if not args.no_oos:
        print(f"\n>>> 基于最优参数进行 OOS 滚动预测（{search['data']} 口径）...")
        oos_pred = evaluate_oos(X_use, best_params, cv, n_jobs=search["cv_n_jobs"])
        mpt_summary = oos_pred.summary()
        print(mpt_summary.to_string())
        log_print(mpt_summary, section="OOS MPP Summary", echo=False)


if __name__ == "__main__":
    main()
