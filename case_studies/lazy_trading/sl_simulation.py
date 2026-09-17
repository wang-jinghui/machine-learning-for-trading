# -*- coding: utf-8 -*-
"""止损事件驱动模拟：WF+CPCV 体系上的每日净值监控 + 事件触发重搜。

定位：既有 WF+CPCV 嵌套搜索（wf_cpcv_search.nested_adaptive_search）已
产出每折自适应最优参数（folds），每折 test 段为固定样本外。本模块在其上
做"实盘语义"的模拟测试——**不再套 WalkForward 固定切分**（止损触发时点
由数据事件决定、epoch 长度异构，固定切分无法表达），而是逐日推进：

* 每日收盘按当前 epoch 的冻结头寸计算组合收益，增量监控净值 / 回撤 /
  滚动波动（口径与 skfolio 属性逐点一致，见下）；
* epoch 上限 = test_size 天：走满 -> "到期"换仓；期间任一止损条件命中 ->
  当日收盘提前重搜（搜索逻辑与 wf-cpcv 相同：外层窗口 = 截至决策日最近
  outer_train_size 天，内层 CPCV + Optuna 与嵌套搜索同构），次日建仓新
  参数（计时重启语义）；两类 epoch 结束时都进入下一轮(重)搜；
* 决策日与预计算 folds 折位对齐（= 该折 train 段末日）时零成本复用其
  参数与审计（无止损事件的模拟即"预计算参数的重放"）；否则现场 CPCV
  重搜并写入运行内缓存（同决策日重复求解直接命中）。

口径（热路径 O(1)/天，不构建对象；与 skfolio 0.20.x 实测逐点一致）：
* 净值     net_t = net_{t-1} * (1 + r_t)，r_t = 当日收益行 x 冻结权重
           （NaN->0：停牌 / 缺数按 0 收益处理）；
* 回撤     dd_t = net_t / cummax(net_{0..t}) - 1（峰值 = epoch 内净值的运行
           最大，入场首日即首个峰值观测、**不含 1.0 虚拟基线**——与
           compounded=True 的 Portfolio.drawdowns 逐点一致；入场后
           先跌再回升但未破 1.0 期间 dd=0，该场景由 cum_return 条件覆盖）；
* 滚动波动 窗口内日收益样本 std（ddof=1，非年化；与 rolling_measure(
           STANDARD_DEVIATION, W) 一致；条件 annualize 选项按 x sqrt(252)）。
冷路径（每个 epoch 结束一次）：构建该 epoch 的 Portfolio(block, weights,
compounded=True)，并断言热路径序列与其属性（returns / drawdowns /
cumulative_returns / rolling_measure）allclose；收尾用 MultiPeriodPortfolio
(legs, compounded=True) 出全套官方指标（净值 / 几何回撤 / summary）。

注意（与原 WF 折位的 1 天差异）：WF 折在决策日与 test 起点之间留
purged_size 个间隔日；本模拟为"决策日次交易日建仓"（触发止损 = 下一日
执行新策略），无间隔日 -> 无止损重放相对原折 test 段整体前移 1 天
（purged_size=1 时）。对照基线请使用"同一模拟器 conditions=[]"的结果。

用法::

    import sys; sys.path.insert(0, r"case_studies/lazy_trading")
    from sl_simulation import load_sl_config, simulate_sl, summarize_sl

    cfg = load_sl_config()                    # 止损条件 + 模拟调度（搜索配置读嵌套 toml）
    res  = simulate_sl(X, folds=folds, cfg=cfg)                  # SL 模拟
    base = simulate_sl(X, folds=folds, cfg=cfg, conditions=[])   # 无止损基线
    print(summarize_sl(res, base))            # 对比表
    print(res["epochs"]); print(res["triggers"])

冒烟 CLI（小预算自检；--dd-threshold 覆盖 drawdown 阈值以便自动触发止损路径）::

    D:\\Anaconda3\\envs\\ml4t\\python.exe sl_simulation.py --n-trials 3 --tail-days 1500 --dd-threshold -0.01
"""

from __future__ import annotations

import argparse
import math
import operator
import sys
import tomllib
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from skfolio import MultiPeriodPortfolio, Portfolio, RiskMeasure

from cpcv_analysis import empirical_deflate
from cpcv_search_base import build_pipeline
from wf_cpcv_search import load_nested_config, search_inner_params

# Windows GBK 控制台打印中文安全兜底（与同目录脚本一致）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "sl_simulation_config.toml"
DEFAULT_NESTED_CONFIG = BASE_DIR / "cpcv_parameter_search_config.toml"

MIN_TRAIN_DAYS = 252      # 决策日重搜的最短训练窗口（与嵌套搜索 X_tr<252 跳过同口径）
ANNUAL_FACTOR = 252       # 年化因子（Portfolio 默认 annualized_factor，滚动波动年化用）

# search_inner_params 接受的键（从 load_nested_config 的 search_kwargs 过滤）
_SEARCH_KEYS = ("n_test_folds", "inner_purged_size", "inner_embargo_size",
                "n_jobs", "cv_n_jobs", "n_trials", "sampler", "patience",
                "min_delta", "seed", "verbose")


# ---------------------------------------------------------------------------
# 配置加载：止损条件 + 模拟调度（搜索配置与 wf-cpcv 同源，不重复定义）
# ---------------------------------------------------------------------------
def load_sl_config(config_path=DEFAULT_CONFIG,
                   nested_config_path=DEFAULT_NESTED_CONFIG) -> dict:
    """读取止损模拟配置，并合并嵌套搜索配置（空间 / 窗口 / 搜索参数同源）。

    Returns
    -------
    dict : {"sim": {start_date/end_date/max_hold_days},
            "stop_loss": {min_epoch_days, conditions: [条件配置 dict...]},
            "space": 参数空间（含 train_size），
            "search_kwargs": 嵌套搜索参数（test_size/train_size/n_trials/...），
            "outer_wf": 外层窗口快照}
    """
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    nested = load_nested_config(nested_config_path)
    sim = dict(cfg.get("sim", {}))
    sl = dict(cfg.get("stop_loss", {}))
    return {"sim": sim, "stop_loss": sl,
            "space": nested["space"], "search_kwargs": nested["search_kwargs"],
            "outer_wf": nested["outer_wf"]}


# ---------------------------------------------------------------------------
# 止损条件（可插拔）+ 增量滚动标准差
# ---------------------------------------------------------------------------
class StopLossCondition:
    """单条止损条件（配置驱动，可插拔）。

    metric : "drawdown"（入场以来净值运行最大回撤，负值；入场首日
        dd=0，与 skfolio Portfolio.drawdowns 同口径）/ "cum_return"
        （epoch 累计收益）/ "rolling_std"（epoch 内 W 日滚动日波动，条件自持窗口）。
    op : "<=" | ">="（如 drawdown <= -0.10 触发、rolling_std >= 阈值触发）。
    annualize : 仅 rolling_std 生效——按 x sqrt(252) 年化后与阈值比较。
    check(dd, cum, vols) : vols 为 {window: 滚动std}（窗口未满 None -> 不判）。
    """

    VALID_METRICS = ("drawdown", "cum_return", "rolling_std")

    def __init__(self, name, metric, op, threshold, window=0, annualize=False):
        if metric not in self.VALID_METRICS:
            raise ValueError(f"未知指标 {metric!r}（可选 {self.VALID_METRICS}）")
        if op not in ("<=", ">="):
            raise ValueError(f"未知比较符 {op!r}（可选 '<=', '>='）")
        self.name = str(name)
        self.metric = metric
        self.threshold = float(threshold)
        self.window = int(window or 0)
        self.annualize = bool(annualize)
        self._cmp = operator.le if op == "<=" else operator.ge
        if metric == "rolling_std" and self.window < 2:
            raise ValueError("rolling_std 条件需 window >= 2")

    def check(self, dd, cum, vols) -> bool:
        """单日收盘判定：命中返回 True。"""
        if self.metric == "drawdown":
            v = dd
        elif self.metric == "cum_return":
            v = cum
        else:
            v = vols.get(self.window)
            if v is None:                      # 窗口未满：不判（避免建仓初期误触发）
                return False
            if self.annualize:
                v = v * math.sqrt(ANNUAL_FACTOR)
        return bool(self._cmp(v, self.threshold))

    def __repr__(self):
        extra = f", window={self.window}, annualize={self.annualize}" \
            if self.metric == "rolling_std" else ""
        return f"StopLossCondition({self.name}: {self.metric}{extra})"


class _RollingStd:
    """O(1) 增量滚动标准差（ddof=1，与 skfolio rolling_measure 同口径）。

    push(x) 追加观测并自动滑出超窗样本；std() 窗口满才返回值（否则 None）。
    """

    def __init__(self, window):
        self.window = int(window)
        self.buf = deque()
        self._s = 0.0
        self._s2 = 0.0

    def push(self, x):
        self.buf.append(x)
        self._s += x
        self._s2 += x * x
        if len(self.buf) > self.window:
            y = self.buf.popleft()
            self._s -= y
            self._s2 -= y * y

    def std(self):
        n = len(self.buf)
        if n < self.window or n < 2:
            return None
        var = (self._s2 - self._s * self._s / n) / (n - 1)
        return math.sqrt(max(var, 0.0))


def make_conditions(conds_cfg) -> list:
    """配置列表（dict）或已构建对象列表 -> StopLossCondition 列表。"""
    out = []
    for i, c in enumerate(conds_cfg or []):
        if isinstance(c, StopLossCondition):
            out.append(c)
            continue
        out.append(StopLossCondition(name=str(c.get("name") or f"cond{i + 1}"),
                                     metric=c["metric"], op=c["op"],
                                     threshold=c["threshold"],
                                     window=c.get("window", 0),
                                     annualize=c.get("annualize", False)))
    return out


# ---------------------------------------------------------------------------
# 决策日取参：folds 对齐复用（零成本）/ 现场 CPCV 重搜（写回缓存）
# ---------------------------------------------------------------------------
def seed_cache_from_folds(folds) -> dict:
    """预计算 folds -> 决策日缓存（决策日 = 该折 train 段末日 = 其搜索 X_tr 末日）。

    折位对齐即零成本复用：决策日 d 的搜索窗口（截至 d 最近 outer_train_size
    天）与 fold 内层搜索的 X_tr 完全一致（外层层展窗同长、同数据）。
    """
    cache = {}
    for f in folds or []:
        d_day = pd.Timestamp(f["train"].returns_df.index[-1])
        cache[d_day] = {"params": dict(f["params"]), "score": float(f["score"]),
                        "n_trials": int(f["n_trials"]), "p_luck": float(f["p_luck"]),
                        "source": "fold_reuse", "study": f.get("study"),
                        "decision_day": d_day}
    return cache


def _decide_params(X, decision_pos, cache, space, test_size, outer_train_size,
                   search_kwargs, verbose, warns):
    """决策日 d 取参：缓存命中直接返回；否则现场 CPCV 重搜。数据不足返回 None。

    现场搜索与 wf-cpcv 同构：X_tr = 截至决策日（含）最近 outer_train_size 天，
    search_inner_params 内部按 trial 采样 train_size 子窗口 + 内层 CPCV 评分。
    """
    d_day = pd.Timestamp(X.index[decision_pos])
    rec = cache.get(d_day)
    if rec is not None:
        return rec
    X_tr = X.iloc[:decision_pos + 1].tail(outer_train_size)
    if len(X_tr) < MIN_TRAIN_DAYS:
        warns.append(f"决策日 {d_day.date()}: 训练窗口 {len(X_tr)} 天 < "
                     f"{MIN_TRAIN_DAYS}，跳过重搜（保留原参数）")
        return None
    if verbose:
        print(f"决策日 {d_day.date()}: fresh CPCV 重搜 | 外层窗口 {len(X_tr)} 天 | "
              f"test_size={test_size} | n_trials={search_kwargs.get('n_trials')}")
    best, score, study = search_inner_params(X_tr, space, test_size, **search_kwargs)
    audit = empirical_deflate(study)
    rec = {"params": best, "score": float(score), "n_trials": int(audit["n_trials"]),
           "p_luck": float(audit["p_luck"]), "source": "fresh", "study": study,
           "decision_day": d_day}
    cache[d_day] = rec
    return rec


def _fit_epoch_targets(X, decision_pos, params):
    """决策日拟合：最近 params['train_size'] 天 -> 冻结头寸 (assets, weights)。

    与 run_params_on_fold 折末应用同口径：fit 用内层训练子窗口，
    predict 只应用已拟合权重（不再重训）。
    """
    ts = int(params["train_size"])
    win = X.iloc[decision_pos + 1 - ts:decision_pos + 1]
    m = build_pipeline(params)
    m.fit(win)
    probe = m.predict(win)                 # 冻结权重视图（predict 不重训）
    assets = list(probe.assets)
    w = np.asarray(probe.weights, dtype=float)
    return assets, w


# ---------------------------------------------------------------------------
# epoch 冷路径：leg 对象构建 + 热路径口径自检
# ---------------------------------------------------------------------------
def _verify_epoch(leg, epoch_id, hot_r, hot_dd, hot_net, vol_windows):
    """断言热路径逐日序列与 Portfolio 属性一致（防口径静默漂移）。"""
    lr = np.asarray(leg.returns, dtype=float)
    np.testing.assert_allclose(lr, np.asarray(hot_r, dtype=float),
                               rtol=1e-9, atol=1e-12,
                               err_msg=f"epoch {epoch_id}: returns 与热路径不一致")
    np.testing.assert_allclose(np.asarray(leg.drawdowns, dtype=float),
                               np.asarray(hot_dd, dtype=float), rtol=1e-9, atol=1e-12,
                               err_msg=f"epoch {epoch_id}: drawdowns 与热路径不一致")
    np.testing.assert_allclose(np.asarray(leg.cumulative_returns, dtype=float),
                               np.asarray(hot_net, dtype=float), rtol=1e-9, atol=1e-12,
                               err_msg=f"epoch {epoch_id}: 净值与热路径不一致")
    for W in vol_windows:
        s = leg.rolling_measure(measure=RiskMeasure.STANDARD_DEVIATION, window=W)
        s = s.to_numpy() if hasattr(s, "to_numpy") else np.asarray(s, dtype=float)
        hot = pd.Series(hot_r, dtype=float).rolling(W).std(ddof=1).to_numpy()
        np.testing.assert_allclose(s, hot, rtol=1e-9, atol=1e-12, equal_nan=True,
                                   err_msg=f"epoch {epoch_id}: 滚动波动(W={W}) 与热路径不一致")


# ---------------------------------------------------------------------------
# 模拟主循环
# ---------------------------------------------------------------------------
def simulate_sl(X, folds=None, cfg=None, conditions=None, start_date=None,
                end_date=None, search_overrides=None, verify=True, verbose=True,
                bench=None) -> dict:
    """止损事件驱动模拟主函数（逐日推进，见模块 docstring 语义说明）。

    Parameters
    ----------
    X : pd.DataFrame，与 WF+CPCV 同口径收益数据（DatetimeIndex 升序）
    folds : list | None（可选），nested_adaptive_search 结果：决策日与之对齐时
        零成本复用其参数；None 则全程现场重搜。起点优先级：start_date > folds
        首折决策日 > 第 outer train_size 天收盘（建仓 = 次一交易日）
    cfg : dict | None，load_sl_config() 结果（None 时现场加载默认配置）
    conditions : list | None，止损条件（StopLossCondition / 配置 dict 均可）；
        None -> 用 cfg["stop_loss"]["conditions"]（唯一生效入口）；
        显式传 [] 即"无止损基线"（同循环同缓存）
    start_date / end_date : 覆盖 cfg["sim"] 的模拟起止（start_date = 首个建仓日）
    search_overrides : dict | None，覆盖现场重搜参数（冒烟用，如 n_trials=3）
    verify : bool，每 epoch 结束断言热路径与 Portfolio 属性一致
    verbose : bool，打印 epoch / 搜索摘要
    bench : pd.Series | None，基准收益序列（可选；仅原样放入返回 dict，
        供 notebook 画净值对比图，不参与任何模拟逻辑）

    Returns
    -------
    dict :
        daily    : DataFrame 逐日监控（net_global/net_epoch/cum_epoch/dd_epoch/
                   vol_<W>/hit_<条件名>/trigger）
        epochs   : DataFrame 一行一 epoch（决策日/建仓/退出/天数/exit_reason/
                   search_source/n_trials/p_luck/参数/fit_ts/epoch 指标/换手率）
        triggers : DataFrame 止损触发明细（触发日/条件/触发时指标值）
        legs     : list[Portfolio]（每 epoch 一个 leg，compounded=True）
        mpt      : MultiPeriodPortfolio（全路径官方指标 / 净值 / 几何回撤）
        searches : dict 决策日 -> 取参记录（参数/来源/审计/study）
        warnings : list[str]（数据不足跳过重搜等）
        conditions/cfg : 条件名列表与配置快照
        bench    : 基准收益序列（调用方传入则原样带回，供画图）
    """
    cfg = load_sl_config() if cfg is None else cfg
    search_kwargs = dict(cfg["search_kwargs"])
    if search_overrides:
        search_kwargs.update(search_overrides)
    test_size = int(search_kwargs["test_size"])
    outer_train_size = int(search_kwargs["train_size"])
    space = cfg["space"]
    sk = {k: search_kwargs[k] for k in _SEARCH_KEYS if k in search_kwargs}

    sim_cfg = cfg.get("sim", {})
    max_hold = int(sim_cfg.get("max_hold_days") or test_size)
    min_epoch_days = int(cfg.get("stop_loss", {}).get("min_epoch_days", 0) or 0)
    conditions = make_conditions(cfg.get("stop_loss", {}).get("conditions")) \
        if conditions is None else make_conditions(conditions)
    if len({c.name for c in conditions}) != len(conditions):
        raise ValueError("止损条件名重复（hit_<name> 列名冲突）")
    start_date = start_date if start_date is not None else sim_cfg.get("start_date")
    end_date = end_date if end_date is not None else sim_cfg.get("end_date")

    X = X.sort_index()
    dates = X.index
    n_obs = len(X)
    Xf = X.fillna(0.0)                 # 热路径 / leg 统一 NaN->0（停牌按 0 收益）

    warns = []
    cache = seed_cache_from_folds(folds)

    # ---- 起止位置 ----
    end_pos = n_obs - 1
    if end_date is not None:
        end_pos = int(dates.searchsorted(pd.Timestamp(end_date), side="right") - 1)
        if end_pos < 0:
            raise ValueError(f"end_date={end_date} 早于数据起点")
    if start_date is not None:
        entry_pos = int(dates.searchsorted(pd.Timestamp(start_date), side="left"))
        if entry_pos >= n_obs:
            raise ValueError(f"start_date={start_date} 晚于数据末端")
    elif folds:
        d0 = pd.Timestamp(folds[0]["train"].returns_df.index[-1])
        entry_pos = int(dates.searchsorted(d0, side="right"))   # 决策日次交易日
    else:
        # 无 folds：起点由 train_size 自然给定——前 train_size 天作首个训练
        # 窗口，决策日 = 第 train_size 天收盘，建仓 = 第 train_size+1 天
        entry_pos = outer_train_size
        if entry_pos >= n_obs:
            raise ValueError(
                f"未传 folds：默认起点 = 第 {outer_train_size + 1} 天"
                f"（train_size+1），数据仅 {n_obs} 天")
    if entry_pos < 1:
        raise ValueError("模拟起点需至少留 1 天作为决策日")
    if entry_pos > end_pos:
        raise ValueError("模拟起点晚于终点（检查 start_date / end_date）")

    vol_windows = sorted({c.window for c in conditions if c.metric == "rolling_std"})

    epochs, daily_rows, trigger_rows = [], [], []
    legs = []
    prev_assets, prev_w = None, None
    last_rec = None
    epoch_id = 0
    net_global = 1.0

    while entry_pos <= end_pos:
        decision_pos = entry_pos - 1
        rec = _decide_params(X, decision_pos, cache, space, test_size,
                             outer_train_size, sk, verbose, warns)
        if rec is None:
            if last_rec is None:
                raise ValueError(
                    f"起始决策日 {dates[decision_pos].date()} 数据不足（<{MIN_TRAIN_DAYS} 天）"
                    f"且缓存未命中：请提供 folds 或指定更晚的 start_date")
            rec = {**last_rec, "source": "no_search"}       # 保留原参数继续
        last_rec = rec
        params = rec["params"]
        assets, w = _fit_epoch_targets(X, decision_pos, params)
        Xa = Xf[assets].to_numpy()

        # ---- 逐日热路径 ----
        epoch_end_limit = min(entry_pos + max_hold - 1, end_pos)
        rolls = {W: _RollingStd(W) for W in vol_windows}
        # peak 从 0 起：入场首日净值即首个峰值观测（与 skfolio
        # get_drawdowns 对净值取 cummax 的口径一致，不含 1.0 虚拟基线）
        net, peak = 1.0, 0.0
        hot_r, hot_dd, hot_net = [], [], []
        exit_pos, trig = epoch_end_limit, None
        exit_reason = "expired" if entry_pos + max_hold - 1 <= end_pos else "end_of_data"

        for pos in range(entry_pos, epoch_end_limit + 1):
            r = float(Xa[pos] @ w)
            net *= (1.0 + r)
            if net > peak:
                peak = net
            dd = net / peak - 1.0
            cum = net - 1.0
            net_global *= (1.0 + r)
            day_in = pos - entry_pos + 1
            vols = {}
            for W, rs in rolls.items():
                rs.push(r)
                vols[W] = rs.std()

            hits = [c for c in conditions
                    if day_in >= min_epoch_days and c.check(dd=dd, cum=cum, vols=vols)]
            hit = hits[0] if hits else None
            row = {"date": dates[pos], "epoch": epoch_id, "day_in_epoch": day_in,
                   "ret": r, "net_global": net_global, "net_epoch": net,
                   "cum_epoch": cum, "dd_epoch": dd}
            for W in vol_windows:
                row[f"vol_{W}"] = vols.get(W)
            for c in conditions:
                row[f"hit_{c.name}"] = c in hits
            row["trigger"] = hit.name if hit else ""
            daily_rows.append(row)
            hot_r.append(r)
            hot_dd.append(dd)
            hot_net.append(net)

            if hit is not None:
                exit_pos, trig, exit_reason = pos, hit, f"sl:{hit.name}"
                trigger_rows.append(
                    {"epoch": epoch_id, "date": dates[pos], "day_in_epoch": day_in,
                     "condition": hit.name, "dd_epoch": dd, "cum_epoch": cum,
                     **{f"vol_{W}": vols.get(W) for W in vol_windows}})
                break

        # ---- 冷路径：leg 构建 + 口径自检 ----
        block = Xf.loc[dates[entry_pos]:dates[exit_pos], assets]
        leg = Portfolio(X=block, weights=w, compounded=True, name=f"Ep{epoch_id}")
        if verify:
            _verify_epoch(leg, epoch_id, hot_r, hot_dd, hot_net, vol_windows)
        legs.append(leg)

        # 换手率（相邻 leg 权重 union 对齐，与 cpcv_analysis.turnover_series 同口径）
        if prev_assets is not None:
            common = np.union1d(prev_assets, assets)
            w1 = np.zeros(len(common))
            w1[np.isin(common, prev_assets)] = prev_w
            w2 = np.zeros(len(common))
            w2[np.isin(common, assets)] = w
            turnover = float(0.5 * np.abs(w1 - w2).sum())
        else:
            turnover = float("nan")
        prev_assets, prev_w = assets, w

        p = dict(params)
        fitness = str(p.pop("nondomin__fitness_measures", ""))
        epoch_ret = float(np.asarray(leg.cumulative_returns, dtype=float)[-1] - 1.0)
        epochs.append({"epoch": epoch_id, "decision_day": dates[decision_pos],
                       "entry": dates[entry_pos], "exit": dates[exit_pos],
                       "n_days": exit_pos - entry_pos + 1, "exit_reason": exit_reason,
                       "search_source": rec["source"], "n_trials": rec["n_trials"],
                       "is_score": round(rec["score"], 4), "p_luck": round(rec["p_luck"], 4),
                       "fit_ts": p.pop("train_size", None), "fitness": fitness, **p,
                       "epoch_cum_return": round(epoch_ret, 4),
                       "epoch_max_drawdown": round(float(leg.max_drawdown), 4),
                       "epoch_annualized_mean": round(float(leg.annualized_mean), 4),
                       "epoch_annualized_sharpe": round(float(leg.annualized_sharpe_ratio), 4),
                       "epoch_annualized_std": round(float(leg.annualized_standard_deviation), 4),
                       "rebalance_turnover": round(turnover, 4) if np.isfinite(turnover) else np.nan})
        if verbose:
            print(f"Epoch {epoch_id}: {dates[entry_pos].date()} -> {dates[exit_pos].date()} "
                  f"({exit_pos - entry_pos + 1}d, {exit_reason}) | src={rec['source']} | "
                  f"ASR={float(leg.annualized_sharpe_ratio):.3f} | 累计={epoch_ret:.2%}")

        epoch_id += 1
        entry_pos = exit_pos + 1

    if not legs:
        raise ValueError("无有效 epoch（检查 start_date / end_date / 数据长度）")

    daily = pd.DataFrame(daily_rows)
    epochs_df = pd.DataFrame(epochs)
    trig_cols = ["epoch", "date", "day_in_epoch", "condition", "dd_epoch",
                 "cum_epoch"] + [f"vol_{W}" for W in vol_windows]
    triggers_df = pd.DataFrame(trigger_rows, columns=trig_cols) if trigger_rows \
        else pd.DataFrame(columns=trig_cols)
    mpt = MultiPeriodPortfolio(legs, compounded=True)
    return {"daily": daily, "epochs": epochs_df, "triggers": triggers_df,
            "legs": legs, "mpt": mpt, "searches": cache, "warnings": warns,
            "conditions": [c.name for c in conditions], "cfg": cfg, "bench": bench}


# ---------------------------------------------------------------------------
# 汇总对比：SL 模拟 vs 基线（无止损，同循环同缓存）
# ---------------------------------------------------------------------------
def summarize_sl(result, baseline=None) -> pd.DataFrame:
    """核心指标对比表：行 = 场景（SL 模拟 / 无止损基线），列 = 绩效 + 调度统计。

    指标全部取自 mpt（MultiPeriodPortfolio, compounded=True）官方属性；
    调度统计取自 epochs 表（触发次数 / 搜索来源分布）。
    """
    def _row(res):
        mpt = res["mpt"]
        ep = res["epochs"]
        reasons = ep["exit_reason"].astype(str)
        src = ep["search_source"].astype(str)
        return {"年化收益": round(float(mpt.annualized_mean), 4),
                "年化夏普": round(float(mpt.annualized_sharpe_ratio), 4),
                "最大回撤": round(float(mpt.max_drawdown), 4),
                "期末净值": round(float(np.asarray(mpt.cumulative_returns, dtype=float)[-1]), 4),
                "epoch数": int(len(ep)),
                "止损次数": int(reasons.str.startswith("sl:").sum()),
                "到期换仓": int((reasons == "expired").sum()),
                "新搜索(fresh)": int((src == "fresh").sum()),
                "复用(fold)": int((src == "fold_reuse").sum()),
                "跳过重搜": int((src == "no_search").sum()),
                "交易起点": str(pd.Timestamp(ep["entry"].iloc[0]).date()),
                "交易终点": str(pd.Timestamp(ep["exit"].iloc[-1]).date())}

    rows = {"SL模拟": _row(result)}
    if baseline is not None:
        rows["无止损基线"] = _row(baseline)
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# CLI（冒烟/小预算；完整模拟请在 notebook 会话内复用已完成的 folds）
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="止损事件驱动模拟（冒烟自检）")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="SL 配置文件路径")
    parser.add_argument("--n-trials", type=int, default=3, help="冒烟预算：内层 optuna trial 数")
    parser.add_argument("--tail-days", type=int, default=1500, help="只取数据尾部 N 天")
    parser.add_argument("--data", choices=("X", "X_net"), default="X", help="数据口径")
    parser.add_argument("--n-jobs", type=int, default=2, help="trial 级并行度")
    parser.add_argument("--cv-n-jobs", type=int, default=1, help="单 trial 内 CPCV 并行度")
    parser.add_argument("--dd-threshold", type=float, default=None,
                        help="覆盖 drawdown 条件阈值（冒烟触发止损路径用，如 -0.01）")
    args = parser.parse_args(argv)

    import walkforward_parameter_search as wfps
    from wf_cpcv_search import nested_adaptive_search

    X_all, X_net, info = wfps.load_data()
    X_use = X_net if args.data == "X_net" else X_all
    if args.tail_days:
        X_use = X_use.tail(args.tail_days)
    print(f"[smoke] 数据 {len(X_use)} 天 x {X_use.shape[1]} 列 | 口径={args.data}")

    cfg = load_sl_config(args.config)
    sk = {**cfg["search_kwargs"], "n_trials": args.n_trials,
          "n_jobs": args.n_jobs, "cv_n_jobs": args.cv_n_jobs}
    print(f"[smoke] 小预算嵌套搜索产 folds（n_trials={args.n_trials}）...")
    folds = nested_adaptive_search(X_use, space=cfg["space"], **sk)

    conds_cfg = [dict(c) for c in cfg["stop_loss"]["conditions"]]
    if args.dd_threshold is not None:
        for c in conds_cfg:
            if c.get("metric") == "drawdown":
                c["threshold"] = args.dd_threshold
    conditions = make_conditions(conds_cfg)

    overrides = {"n_trials": args.n_trials, "n_jobs": args.n_jobs, "cv_n_jobs": args.cv_n_jobs}
    print("\n[smoke] SL 模拟 ...")
    res = simulate_sl(X_use, folds=folds, cfg=cfg, conditions=conditions,
                      search_overrides=overrides)
    print("\n[smoke] 无止损基线（同循环同缓存）...")
    base = simulate_sl(X_use, folds=folds, cfg=cfg, conditions=[],
                       search_overrides=overrides)

    print("\n==== SL 模拟 vs 无止损基线 ====")
    print(summarize_sl(res, base).to_string())
    print("\n==== epochs ====")
    print(res["epochs"].to_string())
    print("\n==== triggers ====")
    print(res["triggers"].to_string())
    if res["warnings"]:
        print("\n[warn]")
        for w in res["warnings"]:
            print("  -", w)
    print("\n[smoke] 完成（热路径 vs Portfolio 属性自检已通过）")


if __name__ == "__main__":
    main()
