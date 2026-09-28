# -*- coding: utf-8 -*-
"""标准 CSCV PBO 诊断 v3：候选集 = 参数空间网格直接枚举（无搜索层）。

定位
----
按 Bailey/Borwein/López de Prado/Zhu（The Probability of Backtest Overfitting）原始 CSCV：N 个配置各执行一次 -> 收益矩阵 M (T x N) -> 对称
重排 -> 以论文第 3 节的四种分析判定 IS->OOS 的排名传递性。v3 去掉 v1/v2 中"先搜再评"的搜索层：候选集不再由 Optuna/采样器/预算产生，而是把参数空间
网格逐一枚举（每个网格节点 = 论文意义上的一个 trial）。参数空间边界即检验的分母（N），枚举无 seed、完全确定。

流程（三段，即论文 Algorithm 2.3 的输入构造）
--------------------------------------------
1) 空间枚举：expand_range x product -> N 个配置（字典序，确定性）。
2) 收益矩阵 M：公共起点 S* = max(train_size)（开头冗余仅可供首块 fit 回看，不产出 M 的行）；自 S* 铺共用块网格（每块 block_size 天）。按 train_size
   分组——每组 (视图 X.iloc[S*-ts:], WalkForward(test_size=block_size,train_size=ts)) 只定义一次、组内全体候选复用；候选在每个块前 fit 最近ts 天 ->
   predict 该块。N 条流同起点、同一日期索引（结构性对齐）-> M(T x N)；尾部不足一块丢弃。
3) CSCV 内核 + 四种分析（论文 3.1-3.3；下示默认度量 Sharpe，可用 measure 切换）：
   [1] PBO = freq(lambda <= 0)：IS 冠军 OOS 落在全体中位数之后的比例；
   [2] 性能退化：IS 冠军的 (SR_IS, SR_OOS) 对 + OLS 斜率（过拟合补偿效应表现为负斜率）+ 可达 OOS 区间；
   [3] 亏损概率：Prob[SR_OOS(IS 冠军) < 0]（与 PBO 独立）；
   [4] 随机占优：CDF(IS 冠军的 SR_OOS, 跨组合) vs CDF(全体配置的 SR_OOS,跨组合池)，FSD/SD2 是否成立。

与 v1/v2 的关系
--------------
v1/v2 保持不动（历史口径留档）。本模块复用 cpcv_search_base.build_pipeline（管道契约）；内核与 cscv_pbo.cscv_core 数学同构
（同块切分、同组合枚举、同Sharpe 增量/argmax/rank/logit 与 -inf 处理），因内存约束改为组合分块增量计算（结果逐位等价，可对拍），
并额外产出四种分析所需的全量 OOS 矩阵（float32，分析后随内核释放；峰值内存约 O(n_combos x N x 4B) + 分块瞬时量）。

用法
----
    from cscv_pbo_v3 import build_topk_report, cscv_pbo, plot_cscv, plot_topk
    out = cscv_pbo(X, space=search_space, block_size=252, S=16)
    print(out["report"])                    # 四种分析报告（文本）
    out["pbo"]; out["loss_prob"]; out["degradation"]["slope"]
    out["dominance"]["fsd_holds"]; out["matrix"]; out["meta"]
    plot_cscv(out)                          # 四种分析图（各一张独立画布）
    plot_dsr(out)                           # DSR 判定三联图（[5a]-[5c]）
    print(build_topk_report(out, k=5))      # topK 冠军候选验收（文本,零重算）
    plot_topk(out, k=5)                     # topK 评估区累计净值（黑虚线=全员截面中位）
    out_mdd = cscv_pbo(X, space=search_space, block_size=252, S=16,
                       measure="MAX_DRAWDOWN")  # 切换评价度量（skfolio 口径）

注：评价度量（论文 3.1 的 f = 任意表现统计量）可切换。measure=None 为日频
未年化 Sharpe（mean/std, rf=0，与 v1/v2 内核位级同构）；measure 传 skfolio
度量名/枚举（MEAN / ANNUALIZED_MEAN / VARIANCE / ANNUALIZED_VARIANCE /
STANDARD_DEVIATION / ANNUALIZED_STANDARD_DEVIATION / SKEW / KURTOSIS /
SHARPE_RATIO / ANNUALIZED_SHARPE_RATIO / MAX_DRAWDOWN / CALMAR_RATIO）即以
该度量算 PBO，四种分析同步跟随；方向按 skfolio 约定（Perf/Ratio 越大越优，
Risk/ExtraRisk 越小越优）。收益流整体缩放（如单位换算）不改变任何输出——
Sharpe = mean/std，分子分母同乘相消；对 Sharpe 读数的年化（乘 sqrt(252)）
也不改变判断结论：PBO / omega / lambda 依赖排名、亏损概率依赖符号、退化斜率
与 Spearman 相消、FSD / SD2 判决不变，仅截距 / OOS 分位 / SD2 幅值同比缩放
（年化可读值见 build_topk_report）。SORTINO / CVaR / VaR / MAD 等不在列表的原因、
直接重算的实测成本与一致性验证结论，见下节。

度量支持边界（SORTINO / CVaR / VaR / MAD 等为何不在可切换列表）
------------------------------------------------------------
可切换列表 = “可由块级充分统计精确重建”的度量，仅三类结构：
(a) 可加矩——mean/var/std/Sharpe/SKEW/KURTOSIS 都是 (Σr, Σr², Σr³, Σr⁴, n)
    的闭式函数，段和 = 块和相加；
(b) 可递推合并的路径摘要——MAX_DRAWDOWN 的 (mdd, 峰, 尾, 谷)（CALMAR 复用其
    再加段均收益）；
(c) 固定阈值部分矩（skfolio 默认口径未使用）。
SORTINO/半方差与 MAD 以“段内均值”为阈值：{r < 均值} 的归属须先聚合出均值再回扫
原始值（二次扫描），块级摘要判定不了归属（反例：两块 (Σr, Σr²) 完全相同，半方差
与最坏值可以不同）；VaR/CVaR/DAR 是段内顺序统计量（分位/尾部均值），同样不在摘要
信息内。注意：这是“块级合并路径”的边界，不是“不能算”——把组合的段拼出来、逐组合
直接调 skfolio 原生函数（2D 数组）即为精确路径（已对拍 Portfolio.get_measure 零偏
差）。实测成本（S=16、T=4032、12870 组合、IS+OOS 全量，逐组合直接重算）：N=8 三度
量合计 ~15s；N=128 ~4min；N=512 ~18min（MAD 250s / CVaR 404s / Sortino 372s，组装
仅 ~4%；近线性于 N，大 N 因内存带宽略超线性），同配置内核度量 ~1s。因实际 N 较大
暂不纳入；如需，按“分块物化 + skfolio 2D 函数”加分支即可，下游 argmax / rank /
PBO / 四种分析完全复用。块级合并与直接重算的一致性已全量对拍（S ∈ {4,6,8,16} 全组
合 x 全体候选 x 12 度量：n_star / omega / PBO 完全一致，数值差 ~1e-14 为浮点求和
顺序噪声；等价性由可加矩恒等式与 MDD 串接递推恒等式保证，非近似）。
"""

from __future__ import annotations

import itertools
import os
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# Windows GBK 控制台打印中文安全兜底（与其他模块顶层一致）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from skfolio.model_selection import WalkForward, cross_val_predict
from tqdm.auto import tqdm

from cpcv_search_base import build_pipeline

DEFAULT_CONFIG = Path(__file__).resolve().parent / "cpcv_parameter_search_config.toml"

__all__ = [
    "cscv_pbo",
    "build_returns_matrix",
    "calc_dsr_from_matrix",
    "cscv_core",
    "enumerate_space",
    "expand_range",
    "build_report",
    "build_topk_report",
    "plot_cscv",
    "plot_dsr",
    "plot_topk",
    "load_space",
]


# ---------------------------------------------------------------------------
# 参数空间 schema：与 walkforward_parameter_search / cpcv_search_base 同语义
# ---------------------------------------------------------------------------
def _decimals(x) -> int:
    """数值的小数位数（用于浮点结果舍入，消除累积误差）。"""
    s = format(x, ".15g")
    return len(s.split(".")[1]) if "." in s else 0


def expand_range(node: dict) -> list:
    """range 节点 -> 候选值列表（与 walkforward_parameter_search.expand_range 同语义）。"""
    low, high, step = node["low"], node["high"], node["step"]
    if isinstance(low, int) and isinstance(high, int) and isinstance(step, int):
        n = (high - low) // step + 1
        return [low + i * step for i in range(n)]
    n = round((high - low) / step) + 1
    decimals = max(_decimals(x) for x in (low, high, step))
    return [round(low + i * step, decimals) for i in range(n)]


def enumerate_space(space: dict) -> list[dict]:
    """参数空间 -> 全部网格节点（笛卡尔积；顺序确定，无 seed）。

    range 节点经 expand_range 展开，choice 列表原样；返回顺序 = 字典插入序
    各维候选值的 product 遍历序，保证可复现、可对拍。
    """
    names = list(space)
    values = [
        list(node) if isinstance(node, list) else expand_range(node)
        for node in space.values()
    ]
    return [dict(zip(names, combo)) for combo in itertools.product(*values)]


def load_space(path=None) -> dict:
    """读取 TOML 参数空间（[param_space] + [nested_space] 合并）；space=None 时默认。"""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    space: dict = {}
    for section in ("param_space", "nested_space"):
        space.update(cfg.get(section, {}))
    return space


# ---------------------------------------------------------------------------
# 收益矩阵：共用块网格 + 按 train_size 分组复用 walkforward（对齐机制核心）
# ---------------------------------------------------------------------------
def build_returns_matrix(X, params_list, *, start=None, block_size=252,
                         n_jobs=None, verbose=True):
    """构建 N 条对齐收益流 -> M (T x N)。

    对齐机制：公共起点 S* = start（默认 max(train_size)）；自 S* 起铺共用块
    网格 [S*+b*bs, S*+(b+1)*bs)。按 train_size 分组，每组
    (视图 X.iloc[S*-ts:], WalkForward(test_size=block_size, train_size=ts))
    只定义一次、组内候选复用（skfolio splitter 为无状态索引生成器，可安全
    复用）；候选在每个块前 fit 最近 ts 天（首块 fit 窗落在开头冗余内）-> 
    predict 该块。n_blocks = (len(X) - S*) // block_size，尾部残块丢弃。

    n_jobs：折级并行度（cross_val_predict）；None -> min(n_blocks, cpu 核数)
    ——并行上限即 cv 折数，无需手传；显式传参仅用于限制峰值内存
    （worker 各持视图副本，峰值 ≈ n_jobs x 视图）。
    """
    ts_list = [int(p["train_size"]) for p in params_list]
    max_ts = max(ts_list)
    if start is None:
        start = max_ts
    if start < max_ts:
        raise ValueError(f"build_returns_matrix: 公共起点 {start} < max(train_size) "
                         f"{max_ts}，首块 fit 回看不足")
    if len(X) - start < 2 * block_size:
        raise ValueError(f"build_returns_matrix: 评估区不足 2 块"
                         f"（{len(X) - start} 天 / {block_size} 天）")
    n_blocks = (len(X) - start) // block_size
    if n_jobs is None:
        n_jobs = min(n_blocks, max(1, os.cpu_count() or 1))
    axis = X.index[start:start + n_blocks * block_size]

    # 每组 walkforward 定义一次（视图与 splitter 必须成对：视图不按 ts 裁剪
    # 会导致首折 test 起点错位，共用网格即破坏）
    groups = {
        ts: (X.iloc[start - ts:], WalkForward(test_size=block_size, train_size=ts))
        for ts in sorted(set(ts_list))
    }
    if verbose:
        print(f"  收益流构建（v3）：{len(params_list)} 参数 x {n_blocks} 块 x "
              f"{block_size} 天 = {n_blocks * block_size} 天 | "
              f"公共起点第 {start} 天（冗余 {max_ts} 天）")
        print(f"  对齐网格：train_size 分 {len(groups)} 组，"
              f"walkforward 每组定义一次、组内复用 | 折级并行 n_jobs = {n_jobs}"
              f"（上限即折数 {n_blocks}；省内存可显式传小）")

    cols = {}
    iterator = enumerate(params_list)
    if verbose:
        iterator = tqdm(iterator, total=len(params_list),
                        desc="收益流构建", unit="候选")
    for j, p in iterator:
        x_view, cv = groups[int(p["train_size"])]
        mp = cross_val_predict(build_pipeline(p), x_view, cv=cv, n_jobs=n_jobs)
        ret = mp.returns_df
        if getattr(ret, "ndim", 1) > 1:
            ret = ret.iloc[:, 0]
        ret = ret.reindex(axis)
        miss = int(ret.isna().sum())
        if verbose and miss:
            print(f"  候选 {j + 1}/{len(params_list)}: 缺失 {miss} 天已置 0")
        cols[j] = ret.fillna(0.0)
    return pd.DataFrame(cols, index=axis)


# ---------------------------------------------------------------------------
# CSCV 内核（v1 同构超集）+ 论文四种分析
#
# 评价度量（measure 参数）：论文 3.1 的 f 可为任意表现统计量。默认 None 走
# 日频 Sharpe 快路径（与 v1/v2 位级同构）；其余切换为 skfolio 内置度量，以
# 块级充分统计精确重建——矩类由 sum/sumsq/cube/quad 的块和聚合，回撤类由
# (mdd, 峰, 尾, 谷) 充分统计按串接递推合并；方向按 skfolio fitness 约定。
# ---------------------------------------------------------------------------
_MEASURES = {
    # 收益指标（PerfMeasure）
    "MEAN": {"kind": "mean", "higher": True, "signed": True},
    "ANNUALIZED_MEAN": {"kind": "mean_x252", "higher": True, "signed": True},
    # 风险指标（RiskMeasure；方向约定：越小越优）
    "VARIANCE": {"kind": "var", "higher": False, "signed": False},
    "ANNUALIZED_VARIANCE": {"kind": "var_x252", "higher": False,
                            "signed": False},
    "STANDARD_DEVIATION": {"kind": "std", "higher": False, "signed": False},
    "ANNUALIZED_STANDARD_DEVIATION": {"kind": "std_xsqrt252", "higher": False,
                                      "signed": False},
    "MAX_DRAWDOWN": {"kind": "mdd", "higher": False, "signed": False},
    # 收益风险比（RatioMeasure）
    "SHARPE_RATIO": {"kind": "sharpe", "higher": True, "signed": True},
    "ANNUALIZED_SHARPE_RATIO": {"kind": "sharpe_xsqrt252", "higher": True,
                                "signed": True},
    "CALMAR_RATIO": {"kind": "calmar", "higher": True, "signed": True},
    # 高阶矩（ExtraRiskMeasure；同 skfolio fitness 方向约定）
    "SKEW": {"kind": "skew", "higher": False, "signed": False},
    "KURTOSIS": {"kind": "kurt", "higher": False, "signed": False},
}


def _resolve_measure(measure):
    """measure 参数 -> 注册表名；None -> None（默认日频 Sharpe 快路径）。"""
    if measure is None:
        return None
    from skfolio.measures import BaseMeasure
    name = (measure.name if isinstance(measure, BaseMeasure)
            else str(measure).strip().upper())
    if name not in _MEASURES:
        raise ValueError(
            f"cscv_core: 不支持 measure={measure!r}。可切换的 skfolio 度量（有"
            f"块级充分统计、可精确重建）: {', '.join(sorted(_MEASURES))}；"
            f"SORTINO / CVaR / VaR / MAD 等需段内分位或段内均值二次扫描，"
            f"块级合并无法精确重建，暂不支持（原因与直接重算成本见模块 "
            f"docstring“度量支持边界”节）。")
    return name


def _measure_values(kind, n, sums, sqs=None, cubes=None, quads=None):
    """按 skfolio 公式（annualized_factor=252）由段内原始矩和计算度量（向量化）。

    口径要点：VARIANCE/STD 为样本(ddof=1)；SKEW/KURTOSIS 为总体矩（skfolio
    biased 约定）；SHARPE_RATIO = mean/std；ANNUALIZED_SHARPE_RATIO =
    (mean x 252)/(std x sqrt(252))；std=0 的退化段 Sharpe 记 -inf（与默认
    内核一致），SKEW/KURT 的常数段记 NaN（不参与排名与四项分析）。
    """
    m1 = sums / n
    if kind == "mean":
        return m1
    if kind == "mean_x252":
        return m1 * 252.0
    var = (sqs - n * m1 ** 2) / (n - 1.0)
    if kind == "var":
        return var
    if kind == "var_x252":
        return var * 252.0
    std = np.sqrt(np.maximum(var, 0.0))
    if kind == "std":
        return std
    if kind == "std_xsqrt252":
        return std * np.sqrt(252.0)
    if kind == "sharpe":
        return np.divide(m1, std, out=np.full_like(m1, -np.inf),
                         where=std > 0)
    if kind == "sharpe_xsqrt252":
        ann_std = std * np.sqrt(252.0)
        return np.divide(m1 * 252.0, ann_std, out=np.full_like(m1, -np.inf),
                         where=ann_std > 0)
    m2b = sqs / n - m1 ** 2                       # 总体二阶中心矩（biased）
    if kind == "skew":
        m3b = (cubes - 3.0 * m1 * sqs + 3.0 * m1 ** 2 * sums
               - n * m1 ** 3) / n
        return np.divide(m3b, np.power(np.maximum(m2b, 0.0), 1.5),
                         out=np.full_like(m1, np.nan), where=m2b > 0)
    if kind == "kurt":
        m4b = (quads - 4.0 * m1 * cubes + 6.0 * m1 ** 2 * sqs
               - 4.0 * m1 ** 3 * sums + n * m1 ** 4) / n
        return np.divide(m4b, m2b * m2b, out=np.full_like(m1, np.nan),
                         where=m2b > 0)
    raise ValueError(f"_measure_values: 未知 kind={kind}")


def _block_mdd_stats(blocks):
    """每块 MDD 串接合并式充分统计 (mdd, 峰, 尾, 谷)。

    skfolio 默认 compounded=False：算术累计 cum 的 max(cummax - cum)
    （= Portfolio.max_drawdown，正号幅值；与搜索 fitness 的 mpt.max_drawdown
    同口径）。合并式（a 在前、b 在后）：
        mdd = max(mdd_a, mdd_b, 峰_a - 尾_a - 谷_b)
    依据 dd(t in b) = max(峰_a - 尾_a - cum_b(t), cummax_b(t) - cum_b(t))。
    """
    S, L, N = blocks.shape
    mdd = np.empty((S, N))
    peak = np.empty((S, N))
    end = np.empty((S, N))
    low = np.empty((S, N))
    for b in range(S):
        cum = np.cumsum(blocks[b], axis=0)
        mdd[b] = (np.maximum.accumulate(cum, axis=0) - cum).max(axis=0)
        peak[b] = cum.max(axis=0)
        end[b] = cum[-1]
        low[b] = cum.min(axis=0)
    return mdd, peak, end, low


def _merge_mdd(a, b):
    """MDD 充分统计串接合并（a 在前、b 在后）。"""
    mdd_a, peak_a, end_a, low_a = a
    mdd_b, peak_b, end_b, low_b = b
    return (np.maximum(np.maximum(mdd_a, mdd_b), peak_a - end_a - low_b),
            np.maximum(peak_a, end_a + peak_b),
            end_a + end_b,
            np.minimum(low_a, end_a + low_b))


def _mdd_segment(rows, stats):
    """按块行索引 (chunk, S/2) 计算逐组合段 MDD（块按自然序串接）。"""
    carry = None
    for k in range(rows.shape[1]):
        blk = tuple(st[rows[:, k]] for st in stats)
        carry = blk if carry is None else _merge_mdd(carry, blk)
    return carry[0]


def _crit_info(res):
    """当前评价度量的展示信息：(短标签, 越大越优, 是否默认 Sharpe)。"""
    name = res.get("measure")
    if name is None:
        return "SR", True, True
    spec = _MEASURES[name]
    return name, bool(spec["higher"]), False


def _finalize_analyses(res, sel_is, sel_oos, oos32, higher, signed, n_bins):
    """论文 3.2/3.3 分析收尾（方向感知）。

    higher=越大越优（占优判据朝向翻转）；signed=有亏损符号语义（f<0 即亏损，
    否则 loss_prob=NaN）。默认度量（Sharpe，higher/signed 均 True）与 v1/v2
    逐位同构。
    """
    # ---- 论文 3.2：性能退化 + 亏损概率（基于 IS 冠军的 (f_IS, f_OOS) 对）----
    sel_fin = np.isfinite(sel_is) & np.isfinite(sel_oos)
    n_sel_excl = int(sel_fin.size - int(sel_fin.sum()))
    is_f = sel_is[sel_fin]
    oos_f = sel_oos[sel_fin]
    if oos_f.size >= 2:
        slope, intercept = np.polyfit(is_f, oos_f, 1)
        rho = float(np.corrcoef(rankdata(is_f), rankdata(oos_f))[0, 1])
    else:
        slope = intercept = rho = np.nan
    p10, p50, p90 = (np.percentile(oos_f, [10, 50, 90])
                     if oos_f.size else (np.nan, np.nan, np.nan))
    res["loss_prob"] = ((float(np.mean(oos_f < 0)) if oos_f.size else np.nan)
                        if signed else np.nan)
    res["degradation"] = {"slope": float(slope), "intercept": float(intercept),
                          "spearman": rho, "n_used": int(oos_f.size),
                          "n_excluded": n_sel_excl,
                          "oos_p10": float(p10), "oos_p50": float(p50),
                          "oos_p90": float(p90)}

    # ---- 论文 3.3：随机占优（IS 冠军 CDF vs 全体配置池 CDF，分箱近似）----
    flat = oos32.reshape(-1)
    fin_mask = np.isfinite(flat)
    n_nonfin = int(flat.size - int(fin_mask.sum()))
    if n_nonfin:
        fin = flat[fin_mask]
        lo, hi = float(fin.min()), float(fin.max())
        pool = np.clip(flat, lo, hi)
        del fin
    else:
        lo, hi = float(flat.min()), float(flat.max())
        pool = flat
    if hi <= lo:
        hi = lo + 1e-12
    edges = np.linspace(lo, hi, n_bins + 1)        # CDF/SD2 统一对齐到边网格
    cnt_all, _ = np.histogram(pool, bins=edges)
    cnt_sel, _ = np.histogram(oos_f, bins=edges)
    tot_all = max(int(cnt_all.sum()), 1)
    tot_sel = max(int(cnt_sel.sum()), 1)
    cdf_all = np.concatenate([[0.0], np.cumsum(cnt_all) / tot_all])
    cdf_sel = np.concatenate([[0.0], np.cumsum(cnt_sel) / tot_sel])

    diff = (cdf_all - cdf_sel) if higher else (cdf_sel - cdf_all)
    sd2 = np.concatenate(
        [[0.0], np.cumsum((diff[:-1] + diff[1:]) * 0.5 * np.diff(edges))])
    fsd_violation = float(np.max(-diff))           # >=0 <=> IS 冠军占优
    fsd_gain = float(np.max(diff))
    tol = 1e-12
    res["dominance"] = {
        "fsd_holds": bool(fsd_violation <= tol and fsd_gain > tol),
        "sd2_holds": bool(sd2.min() >= -tol and sd2.max() > tol),
        "fsd_violation": fsd_violation,
        "fsd_gain": fsd_gain,
        "sd2_min": float(sd2.min()),
        "n_nonfinite_oos": n_nonfin,
        "grid": edges,
        "cdf_sel": cdf_sel,
        "cdf_all": cdf_all,
        "sd2": sd2,
        # 归一化逐箱频数（箱宽恒定，与密度仅差常数），供分布对比面板
        "hist_sel": cnt_sel / tot_sel,
        "hist_all": cnt_all / tot_all,
        "n_bins": int(n_bins),
    }


def cscv_core(m, S=16, chunk=512, n_bins=400, measure=None):
    """CSCV 内核：lambda/PBO + 论文 3.1-3.3 四种分析（度量可切换）。

    与 cscv_pbo.cscv_core 数学完全一致（块切分、组合枚举、sum/sumsq 增量
    Sharpe、IS argmax 选 n*、升序 rank（1=最差）、omega = r/(N+1)、
    lambda = ln(omega/(1-omega))、PBO = freq(lambda<=0)、std=0 时 -inf 兜底），
    差异仅两点：(a) 组合分块增量计算以控制峰值内存（结果与整矩阵运算逐位
    等价）；(b) 额外输出四种分析：pbo / loss_prob / degradation / dominance
    （dominance 需全量 OOS 度量值，内部以 float32 暂存、分析后释放；
    输出含 CDF/SD2 与归一化逐箱频数曲线）。

    measure=None 为日频 Sharpe 快路径（与 v1/v2 位级同构）。传入 skfolio
    度量名/枚举时，论文 3.1 的 f 即该度量：IS 夺冠按 skfolio 方向约定取
    argmax（Perf/Ratio 越大越优；Risk/ExtraRisk 越小越优），omega/lambda 与
    四种分析同步跟随；sel_sharpe_is/oos 与 dominance 池均为该度量的原始值
    （skfolio 原生刻度）。可切换清单见 _MEASURES——矩类由块级 sum/sumsq/
    cube/quad 精确重建；MAX_DRAWDOWN 为 skfolio 默认 compounded=False 算术
    回撤（正号幅值），由 (mdd, 峰, 尾, 谷) 充分统计串接递推合并（S 块任意
    深度合并均精确）；CALMAR_RATIO = 段均收益 / MAX_DRAWDOWN。SORTINO /
    CVaR / VaR / MAD 等无法由块级摘要精确重建（段内均值二次扫描 / 段内分位），
    故不在本路径；其精确计算需逐组合直接重算（实测分钟级、随 N 增长，原因与
    成本见模块 docstring“度量支持边界”节），当前未纳入。

    Parameters
    ----------
    m : array-like, shape (T, N)
        T x N 收益矩阵（行 = 时间，列 = 候选策略）。
    S : int
        等分块数（偶数），组合数 = C(S, S/2)。
    chunk : int
        组合分块大小（并行控制峰值内存；回撤类内部上限 128）。
    n_bins : int
        dominance 的 CDF/SD2 分箱数。
    measure : str or skfolio Measure, optional
        评价度量；None -> 日频 Sharpe。支持 _MEASURES 注册的度量名或枚举。

    Returns
    -------
    dict
        v1 全量键（pbo/lambdas/omega/n_star/sel_sharpe_is/sel_sharpe_oos/
        n_combos/S/block_len/T_used/N）+ measure + loss_prob + degradation +
        dominance。
    """
    m = np.asarray(m, dtype=float)
    if m.ndim != 2:
        raise ValueError("cscv_core: m 需为 (T, N) 二维矩阵")
    T, N = m.shape
    if S < 2 or S % 2 != 0:
        raise ValueError(f"cscv_core: S 需为 >=2 的偶数（收到 {S}）")
    if T < 2 * S:
        raise ValueError(f"cscv_core: T={T} 过短，无法等分 {S} 块")
    if not np.isfinite(m).all():
        raise ValueError("cscv_core: 矩阵含非有限值（请先对齐缺失为 0）")
    name = _resolve_measure(measure)
    if name is not None:                 # 度量切换路径（默认路径不受影响）
        return _cscv_core_measure(m, S=S, chunk=chunk, n_bins=n_bins,
                                  name=name)
    t_use = (T // S) * S
    L = t_use // S
    blocks = m[:t_use].reshape(S, L, N)
    sum_b = blocks.sum(axis=1)                 # (S, N) 每块收益和
    sq_b = np.square(blocks).sum(axis=1)       # (S, N) 每块平方和
    total_sums = sum_b.sum(axis=0)
    total_sq = sq_b.sum(axis=0)

    cmat = np.array(list(itertools.combinations(range(S), S // 2)))
    nc = cmat.shape[0]
    n_is = cmat.shape[1] * L
    n_oos = t_use - n_is

    n_star = np.empty(nc, dtype=int)
    omega = np.empty(nc)
    lam = np.empty(nc)
    sel_is = np.empty(nc)
    sel_oos = np.empty(nc)
    oos32 = np.empty((nc, N), dtype=np.float32)   # 全量 OOS Sharpe（占优分析）

    for i0 in range(0, nc, chunk):
        i1 = min(i0 + chunk, nc)
        rows = np.arange(i1 - i0)
        cc = cmat[i0:i1]
        is_sums = sum_b[cc].sum(axis=1)
        is_sq = sq_b[cc].sum(axis=1)
        mean_is = is_sums / n_is
        var_is = (is_sq - n_is * mean_is ** 2) / (n_is - 1)
        std_is = np.sqrt(np.maximum(var_is, 0.0))
        sharpe_is = np.divide(mean_is, std_is, out=np.full_like(mean_is, -np.inf),
                              where=std_is > 0)
        ns = sharpe_is.argmax(axis=1)             # (chunk,) 每组合 IS 冠军

        oos_sums = total_sums - is_sums
        oos_sq = total_sq - is_sq
        mean_oos = oos_sums / n_oos
        var_oos = (oos_sq - n_oos * mean_oos ** 2) / (n_oos - 1)
        std_oos = np.sqrt(np.maximum(var_oos, 0.0))
        sharpe_oos = np.divide(mean_oos, std_oos,
                               out=np.full_like(mean_oos, -np.inf),
                               where=std_oos > 0)

        ranks = rankdata(sharpe_oos, method="average", axis=1)   # 升序 1=最差
        n_star[i0:i1] = ns
        sel_is[i0:i1] = sharpe_is[rows, ns]
        sel_oos[i0:i1] = sharpe_oos[rows, ns]
        omega[i0:i1] = ranks[rows, ns] / (N + 1.0)
        oos32[i0:i1] = sharpe_oos

    lam[:] = np.log(omega / (1.0 - omega))
    pbo = float(np.mean(lam <= 0))

    res = {"pbo": pbo, "lambdas": lam, "omega": omega, "n_star": n_star,
           "sel_sharpe_is": sel_is, "sel_sharpe_oos": sel_oos,
           "n_combos": int(nc), "S": int(S), "block_len": int(L),
           "T_used": int(t_use), "N": int(N), "measure": None}
    _finalize_analyses(res, sel_is, sel_oos, oos32, True, True, n_bins)
    return res


def _cscv_core_measure(m, *, S, chunk, n_bins, name):
    """度量切换版内核：块级充分统计精确重建 skfolio 度量后走同一 PBO 装配。

    IS 夺冠 / omega 排名在方向归一后的 benefit 空间进行（RiskMeasure /
    ExtraRiskMeasure 取负：越小越优）；sel_sharpe_is/oos 与 oos32 存原始
    度量值（skfolio 原生刻度）。退化值（如 SKEW/KURT 的常数段 NaN）不参与
    排名与四项分析（NaN -> -inf，同默认 -inf 剔除机制）。
    """
    spec = _MEASURES[name]
    kind = spec["kind"]
    higher = bool(spec["higher"])
    T, N = m.shape
    t_use = (T // S) * S
    L = t_use // S
    blocks = m[:t_use].reshape(S, L, N)
    sum_b = blocks.sum(axis=1)
    total_sums = sum_b.sum(axis=0)

    need_sq = kind in ("var", "var_x252", "std", "std_xsqrt252", "sharpe",
                       "sharpe_xsqrt252", "skew", "kurt")
    need_cube = kind in ("skew", "kurt")
    need_quad = kind == "kurt"
    need_mdd = kind in ("mdd", "calmar")
    sq_b = np.square(blocks).sum(axis=1) if need_sq else None
    total_sq = sq_b.sum(axis=0) if need_sq else None
    cube_b = (blocks ** 3).sum(axis=1) if need_cube else None
    total_cube = cube_b.sum(axis=0) if need_cube else None
    quad_b = (blocks ** 4).sum(axis=1) if need_quad else None
    total_quad = quad_b.sum(axis=0) if need_quad else None
    if need_mdd:
        mdd_stats = _block_mdd_stats(blocks)

    cmat = np.array(list(itertools.combinations(range(S), S // 2)))
    nc = cmat.shape[0]
    n_is = cmat.shape[1] * L
    n_oos = t_use - n_is
    if need_mdd:
        full = set(range(S))
        comp_rows = np.array([sorted(full - set(r)) for r in cmat.tolist()],
                             dtype=int)
    chunk_eff = min(chunk, 128) if need_mdd else chunk   # 合并式瞬时内存控制

    n_star = np.empty(nc, dtype=int)
    omega = np.empty(nc)
    lam = np.empty(nc)
    sel_is = np.empty(nc)
    sel_oos = np.empty(nc)
    oos32 = np.empty((nc, N), dtype=np.float32)

    for i0 in range(0, nc, chunk_eff):
        i1 = min(i0 + chunk_eff, nc)
        rows = np.arange(i1 - i0)
        cc = cmat[i0:i1]
        if need_mdd:
            mdd_is = _mdd_segment(cc, mdd_stats)
            mdd_oos = _mdd_segment(comp_rows[i0:i1], mdd_stats)
            if kind == "mdd":
                is_v, oos_v = mdd_is, mdd_oos
            else:                                # calmar = 段均收益 / 段 MDD
                sums_is = sum_b[cc].sum(axis=1)
                sums_oos = total_sums - sums_is
                with np.errstate(divide="ignore", invalid="ignore"):
                    is_v = (sums_is / n_is) / mdd_is
                    oos_v = (sums_oos / n_oos) / mdd_oos
        else:
            sums_is = sum_b[cc].sum(axis=1)
            sums_oos = total_sums - sums_is
            if need_sq:
                sq_is = sq_b[cc].sum(axis=1)
                sq_oos = total_sq - sq_is
            else:
                sq_is = sq_oos = None
            if need_cube:
                cu_is = cube_b[cc].sum(axis=1)
                cu_oos = total_cube - cu_is
            else:
                cu_is = cu_oos = None
            if need_quad:
                qu_is = quad_b[cc].sum(axis=1)
                qu_oos = total_quad - qu_is
            else:
                qu_is = qu_oos = None
            is_v = _measure_values(kind, n_is, sums_is, sq_is, cu_is, qu_is)
            oos_v = _measure_values(kind, n_oos, sums_oos, sq_oos, cu_oos,
                                    qu_oos)

        b_is = is_v if higher else -is_v
        b_is = np.where(np.isnan(b_is), -np.inf, b_is)
        b_oos = oos_v if higher else -oos_v
        b_oos = np.where(np.isnan(b_oos), -np.inf, b_oos)
        ns = b_is.argmax(axis=1)

        ranks = rankdata(b_oos, method="average", axis=1)  # benefit 升序 1=最差
        n_star[i0:i1] = ns
        sel_is[i0:i1] = is_v[rows, ns]
        sel_oos[i0:i1] = oos_v[rows, ns]
        omega[i0:i1] = ranks[rows, ns] / (N + 1.0)
        oos32[i0:i1] = oos_v

    lam[:] = np.log(omega / (1.0 - omega))
    pbo = float(np.mean(lam <= 0))
    res = {"pbo": pbo, "lambdas": lam, "omega": omega, "n_star": n_star,
           "sel_sharpe_is": sel_is, "sel_sharpe_oos": sel_oos,
           "n_combos": int(nc), "S": int(S), "block_len": int(L),
           "T_used": int(t_use), "N": int(N), "measure": name}
    _finalize_analyses(res, sel_is, sel_oos, oos32, higher,
                       bool(spec["signed"]), n_bins)
    return res


# ---------------------------------------------------------------------------
# 端到端入口
# ---------------------------------------------------------------------------
def cscv_pbo(X, space=None, *, block_size=252, S=16, mat_n_jobs=None,
             start=None, verbose=True, plot=False, measure=None):
    """端到端 CSCV PBO 诊断 v3：空间枚举 -> N 次执行 -> M -> 四种分析。

    Parameters
    ----------
    X : DataFrame
        资产收益面板（列 = 资产，行 = 日期升序）。每个候选的管道在块前窗口
        fit、对块内 predict，得到该候选的收益流。
    space : dict, optional
        参数空间；None -> cpcv_parameter_search_config.toml 默认。
        range 节点 {"low","high","step"} 展开为网格，choice 列表原样。
    block_size : int
        共用块网格的块长（天）。默认 252。
    S : int
        CSCV 等分块数（偶数），组合数 = C(S, S/2)。默认 16。
    mat_n_jobs : int, optional
        收益矩阵折级并行度（cross_val_predict 的 n_jobs）；None ->
        min(n_blocks, cpu 核数)——由 cv 折数自动定，通常无需手传；显式
        传入仅用于限制峰值内存（worker 各持视图副本）。
    start : int, optional
        公共起点（行索引位）；None -> max(train_size)（最小冗余、全体可用）。
    verbose, plot : bool
        进度输出 / 是否绘制四张独立分析画布（每分析一张）。
    measure : str or skfolio Measure, optional
        评价度量（论文 3.1 的 f）：None -> 日频 Sharpe（与 v1/v2 位级同构）；
        否则为注册的 skfolio 度量名/枚举（如 "MEAN"、"MAX_DRAWDOWN"、
        "ANNUALIZED_SHARPE_RATIO"），PBO / omega / 四种分析均以该度量进行
        （方向按 skfolio 约定；支持清单见 _MEASURES；SORTINO / CVaR / MAD 等
        不支持项的原因与直接重算成本见模块 docstring“度量支持边界”节）。

    Returns
    -------
    dict
        {pbo, lambdas, omega, n_star, sel_sharpe_is, sel_sharpe_oos, n_combos,
         S, block_len, T_used, N, measure, loss_prob, degradation, dominance,
         candidates, space, matrix, meta, report}
        （measure = 当前评价度量名，None 为默认日频 Sharpe；report 为四种
        分析报告文本，不自动打印）
    """
    if space is None:
        space = load_space()
    params_list = enumerate_space(space)
    if len(params_list) < 5:
        raise ValueError(f"cscv_pbo: 候选数 {len(params_list)} 过少（<5），"
                         f"排名统计无意义")
    max_ts = max(int(p["train_size"]) for p in params_list)
    if start is None:
        start = max_ts
    if start < max_ts:
        raise ValueError(f"cscv_pbo: 公共起点 {start} < max(train_size) {max_ts}")
    if len(X) - start < 2 * block_size:
        raise ValueError(f"cscv_pbo: 数据 {len(X)} 天不足以覆盖起点 {start} "
                         f"+ 2 块 x {block_size} 天")
    if verbose:
        print(f"空间枚举（v3）：{len(space)} 维 -> N = {len(params_list)} 个配置"
              f"（确定性网格，无搜索层）")

    M = build_returns_matrix(X, params_list, start=start, block_size=block_size,
                             n_jobs=mat_n_jobs, verbose=verbose)
    core = cscv_core(M.to_numpy(), S=S, measure=measure)
    n_blocks = (len(X) - start) // block_size
    result = {**core, "candidates": params_list, "space": space, "matrix": M,
              "meta": {"block_size": int(block_size), "S": int(S),
                       "start": int(start), "max_ts": int(max_ts),
                       "n_blocks": int(n_blocks),
                       "T_used": int(n_blocks * block_size),
                       "N": int(len(params_list)),
                       "eval_start": str(M.index[0]),
                       "eval_end": str(M.index[-1])}}
    result["report"] = build_report(result)
    if plot:
        plot_cscv(result)
    return result


# ---------------------------------------------------------------------------
# DSR（论文口径）：从 result["matrix"] 直接计算 Deflated Sharpe Ratio
# ---------------------------------------------------------------------------
def calc_dsr_from_matrix(M):
    """论文口径 Deflated Sharpe Ratio（Bailey & López de Prado 2014）。

    输入 = cscv_pbo 的 result["matrix"]（日频 T×N 的 WF-OOS 收益流）：
    每个候选一条轨迹 -> 一个日频 SR 作为 trial 分数，被 deflate 的观测 =
    全样本 SR 最高的列。两步：
        SR* = √V·[(1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(N·e))]，γ = 欧拉常数
        DSR = Φ[(SR-SR*)·√(T-1) / √(1 - γ3·SR + (γ4-1)/4·SR²)]
    γ3/γ4 取冠军列的偏度/非超额峰度（正态 γ4 = 3）；全程日频口径。
    注意：与 cpcv_analysis.calc_dsr 不是同一个量（那边是 MC 经验 p 值，
    缺 PSR 变换）。

    Returns
    -------
    dict : dsr / z_score / sr_obs / sr_star / margin / V / N / T / skew /
        kurt / champion（冠军列标签）
    """
    from scipy.stats import norm

    if not isinstance(M, pd.DataFrame):
        M = pd.DataFrame(np.asarray(M, dtype=float))
    srs = M.mean() / M.std()                        # 日频 SR，每列一个
    champion = srs.idxmax()
    sr = float(srs[champion])
    N, T = int(M.shape[1]), int(M.shape[0])
    V = float(srs.var(ddof=1))                      # trial 分数间方差

    x = M[champion].to_numpy(dtype=float)
    xm = x - x.mean()
    g3 = float((xm ** 3).mean() / xm.std() ** 3)    # 偏度
    g4 = float((xm ** 4).mean() / xm.std() ** 4)    # 非超额峰度（正态=3）

    euler = np.euler_gamma
    sr_star = float(np.sqrt(V) * ((1 - euler) * norm.ppf(1 - 1 / N)
                                  + euler * norm.ppf(1 - 1 / (N * np.e))))
    inflation = max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2, 1e-8)
    z = (sr - sr_star) * np.sqrt(T - 1) / np.sqrt(inflation)

    return {"dsr": float(norm.cdf(z)), "z_score": float(z), "sr_obs": sr,
            "sr_star": sr_star, "margin": sr - sr_star, "V": V,
            "N": N, "T": T, "skew": g3, "kurt": g4, "champion": champion}


# ---------------------------------------------------------------------------
# 报告与绘图
# ---------------------------------------------------------------------------
def build_report(res) -> str:
    """四种分析诊断报告（结构化多行文本，不打印；随 res["measure"] 出标签）。"""
    meta = res["meta"]
    lam = np.asarray(res["lambdas"])
    pct = np.percentile(lam, [5, 50, 95])
    deg = res["degradation"]
    dom = res["dominance"]
    tag, higher, is_default = _crit_info(res)
    if np.isnan(res["loss_prob"]):
        loss_line = f"  [3] 亏损概率：n/a（{tag} 无亏损符号语义）"
    else:
        loss_line = (f"  [3] 亏损概率：Prob[{tag}_OOS(IS 冠军) < 0] = "
                     f"{res['loss_prob']:.1%}")
    dom_note = ("" if is_default
                else f"（{tag}：{'越大' if higher else '越小'}越优）")
    if dom["fsd_holds"]:
        dom_txt = "一阶占优成立（选择程序优于随机选择）"
    elif dom["sd2_holds"]:
        dom_txt = "仅二阶占优（风险厌恶下优于随机选择）"
    else:
        dom_txt = "均不成立（选择程序未优于随机选择）"

    lines = [
        "=" * 66,
        "   CSCV PBO 诊断 v3（空间枚举 N 次执行 + 四种分析，论文 3.1-3.3）",
        "=" * 66,
        f"  候选集：参数空间网格 N = {res['N']}（{len(res['space'])} 维，无搜索层）",
    ]
    for name, node in res["space"].items():
        spec = (f"choice x{len(node)}" if isinstance(node, list)
                else f"{node['low']}~{node['high']} step {node['step']}")
        lines.append(f"    - {name}: {spec}")
    if not is_default:
        lines.append(f"  评价度量：{tag}（skfolio 口径，"
                     f"{'越大' if higher else '越小'}越优，四种分析共用）")
    lines += [
        f"  收益流：{res['N']} 参数 x {meta['n_blocks']} 块 x {meta['block_size']} 天 "
        f"= {meta['T_used']} 天 | 公共起点第 {meta['start']} 天"
        f"（冗余 max(train_size) = {meta['max_ts']} 天）",
        f"  评估区：{meta['eval_start']} ~ {meta['eval_end']}",
        f"  CSCV：S={res['S']} 块（每块 {res['block_len']} 天）| "
        f"组合数 C({res['S']},{res['S'] // 2}) = {res['n_combos']}",
        "  ---------- 四种分析 ----------",
        f"  [1] PBO = freq(lambda<=0) = {res['pbo']:.1%}"
        f"（{int(round(res['pbo'] * res['n_combos']))}/{res['n_combos']}）| "
        f"lambda 5/50/95 分位 = {pct[0]:+.2f}/{pct[1]:+.2f}/{pct[2]:+.2f}"
        f" | 冠军多样性 {len(np.unique(res['n_star']))}/{res['N']}",
        f"  [2] 性能退化：{tag}_OOS = {deg['intercept']:+.3f}"
        f" {deg['slope']:+.3f} x {tag}_IS（Spearman {deg['spearman']:+.3f}）| "
        f"可达 OOS 区间 P10/50/90 = {deg['oos_p10']:+.2f}/"
        f"{deg['oos_p50']:+.2f}/{deg['oos_p90']:+.2f}",
        loss_line,
        f"  [4] 随机占优{dom_note}：FSD = {'是' if dom['fsd_holds'] else '否'}"
        f"（CDF 最大反超 {dom['fsd_violation']:+.4f}）| "
        f"SD2 min = {dom['sd2_min']:+.4f} -> {dom_txt}",
        "  解读：PBO 越接近 0 排名传递性越好（论文惯例阈值 5%）；≈50% 表示"
        "选择与随机无异；",
        "        亏损概率与 PBO 独立——PBO 低而亏损概率高说明 OOS 乏力且非"
        "过拟合所致；",
        "        斜率显著为负 + FSD/SD2 均不成立 = 过拟合的典型证据。",
    ]
    if deg["n_excluded"] or dom["n_nonfinite_oos"]:
        name_txt = "Sharpe" if is_default else tag
        why = "std=0 所致" if is_default else "退化值所致"
        lines.append(f"  注：非有限 {name_txt} 已剔除（组合 {deg['n_excluded']} 个 / "
                     f"OOS 值 {dom['n_nonfinite_oos']} 个，{why}）")
    lines.append("=" * 66)
    return "\n".join(lines)


def _topk_indices(res, k):
    """topK 候选编号：夺冠次数降序、并列按编号升序；k < 1 报错。

    实际数量 = min(k, 夺冠候选数)（后者 <= N）。build_topk_report 与
    plot_topk 共用，保证文本与图表的 topK 口径一致。
    """
    k = int(k)
    if k < 1:
        raise ValueError(f"topK: k 需 >= 1（收到 {k}）")
    vc = pd.Series(np.asarray(res["n_star"])).value_counts()
    return sorted(vc.index.tolist(),
                  key=lambda i: (-int(vc[i]), int(i)))[:k]


def build_topk_report(res, k=3) -> str:
    """topK 冠军候选样本外验收报告（零重算，直接消费 cscv_pbo 结果）。

    对夺冠（IS 冠军）次数最多的前 K 个候选：
    (a) 评估区整体表现（skfolio Portfolio 原生度量：算术年化收益、ddof=1
        年化波动、年化 Sharpe、compounded 净值比价回撤（正号幅值）、日胜率；
        对照全体中位 / 95 分位）；
    (b) 当选组合口径：当选频次与占比、IS/OOS 度量中位（默认 Sharpe，随
        measure 切换）、OOS 分位中位、掉后半占比，以及 topK 对 PBO 的
        加权贡献。

    返回结构化多行文本（不打印；落盘/回显交由调用方，如 log_print）。
    topK 口径与 plot_topk 一致（并列按编号升序；实际 = min(k, 夺冠候选数)）。
    """
    M = res["matrix"]
    from skfolio import Portfolio            # 绩效度量统一走 skfolio 原生属性
    top_idx = _topk_indices(res, k)
    k = len(top_idx)                           # 实际数量（定位/汇总标签口径）
    ns = np.asarray(res["n_star"])
    s_is = np.asarray(res["sel_sharpe_is"])
    s_oos = np.asarray(res["sel_sharpe_oos"])
    om = np.asarray(res["omega"])
    tag, higher, is_default = _crit_info(res)
    crit = "Sharpe" if is_default else tag

    vc = pd.Series(ns).value_counts()          # 各候选夺冠次数（定位行显示）

    def _stats(r):
        """评估区单条收益流的 skfolio 原生度量。

        Portfolio(X, 单位权重, compounded=True)：与项目绩效表（summarize_sl
        的 mpt）同口径——算术年化收益、ddof=1 年化波动、年化 Sharpe（rf=0）、
        净值比价最大回撤（正号幅值）。日胜率 skfolio 无此度量，保留计数。
        """
        x = np.asarray(r, dtype=float).reshape(-1, 1)
        ptf = Portfolio(X=x, weights=np.ones(1), compounded=True)
        return pd.Series({
            "年化收益": float(ptf.annualized_mean),
            "年化波动": float(ptf.annualized_standard_deviation),
            "Sharpe": float(ptf.annualized_sharpe_ratio),
            "最大回撤": float(ptf.max_drawdown),
            "日胜率": float((np.asarray(r) > 0).mean()),
        })

    lines = [
        f"[数据来源] 零重算 | 候选 N={len(M.columns)} x 评估区 T={len(M)} 天 | "
        f"对称组合 {len(ns)} 个 | S={res['S']}",
        f"[top{k} 定位(夺冠次数 | 参数)]",
    ]
    for i in top_idx:
        lines.append(f"  {int(vc[i])}  {res['candidates'][i]}")

    tbl = pd.DataFrame({i: _stats(M[i]) for i in top_idx}).T
    all_tbl = pd.DataFrame({i: _stats(M[i]) for i in M.columns}).T
    tbl.loc["全体-中位"] = all_tbl.median()
    tbl.loc["全体-95分位"] = all_tbl.quantile(0.95)
    lines += ["", f"[评估区整体指标({len(M)} 天,skfolio 口径)]",
              tbl.round(4).to_string()]

    crit_tail = ("为日频未年化" if is_default
                 else f"（skfolio 口径，{'越高' if higher else '越低'}越优）")
    lines += ["",
              f"[当选组合口径 | {crit} {crit_tail};OOS 分位 = omega 排名分位"
              "(越高越靠前)]"]
    for i in top_idx:
        m = ns == i
        lines.append(
            f"  cand{i:>4}: 当选 {m.sum():>5} 次({m.mean():>6.1%}) | "
            f"IS-{crit} 中位 {np.median(s_is[m]):+6.2f} | "
            f"OOS-{crit} 中位 {np.median(s_oos[m]):+6.2f} | "
            f"OOS 分位中位 {np.median(om[m]):>5.1%} | "
            f"掉后半占比 {(om[m] <= 0.5).mean():>5.1%}")

    w = np.array([(ns == i).sum() for i in top_idx]) / len(ns)
    cond = np.array([(om[ns == i] <= 0.5).mean() for i in top_idx])
    lines.append(
        f"  汇总: top{k} 合计份额 {w.sum():.1%} | "
        f"top{k} 加权贡献 lambda<=0 频率 {np.sum(w * cond):.1%} | "
        f"全组合 PBO {(om <= 0.5).mean():.1%}")
    return "\n".join(lines)


def plot_cscv(result, bins=60, figsize=(9, 4.6), dom_figsize=(14, 4.2),
              save_path=None):
    """四个分析各一张独立画布（不拼在同一张画布上）：

    [1] PBO：lambda 分布；
    [2] 性能退化：IS 冠军 (IS, OOS) 度量散点 + OLS 线（默认 SR，随 measure
        切换）；
    [3] 亏损概率：IS 冠军 OOS 度量分布（默认 SR；无亏损符号语义度量显示
        n/a，单色直方图）；
    [4] 随机占优：[4a] 归一化分布对比（看现象）/ [4b] CDF（占优判据）/
        [4c] SD2——同一分析的三视图，单张画布内三栏排布。

    save_path 给定时按 <stem>_pbo/_degradation/_loss/_dominance<suffix>
    分别落盘四张图；返回 {分析名: Figure}。
    """
    import matplotlib.pyplot as plt
    from IPython.display import display

    lam = np.asarray(result["lambdas"])
    deg = result["degradation"]
    dom = result["dominance"]
    tag, higher, is_default = _crit_info(result)
    dom_note = ("" if is_default
                else f"（{tag}：{'越大' if higher else '越小'}越优）")
    grid = np.asarray(dom["grid"])
    centers = 0.5 * (grid[:-1] + grid[1:])
    is_v = np.asarray(result["sel_sharpe_is"])
    oos_v = np.asarray(result["sel_sharpe_oos"])
    ok = np.isfinite(is_v) & np.isfinite(oos_v)
    n_combos = result["n_combos"]
    figs = {}

    # ---- [1] PBO：lambda 分布 ----
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.hist(lam, bins=bins, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(0.0, color="#C44E52", linestyle="--", linewidth=1.6)
    ax.set_title(f"[1] PBO = {result['pbo']:.1%} | lambda 分布"
                 f"（{n_combos} 组合）")
    ax.set_xlabel("lambda = ln(omega/(1-omega))")
    ax.set_ylabel("组合频数")
    fig.tight_layout()
    figs["pbo"] = fig

    # ---- [2] 性能退化：IS/OOS 冠军散点 + OLS 线 ----
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.scatter(is_v[ok], oos_v[ok], s=4, alpha=0.35, color="#4C72B0")
    if np.isfinite(deg["slope"]) and int(ok.sum()) >= 2:
        xs = np.linspace(is_v[ok].min(), is_v[ok].max(), 50)
        ax.plot(xs, deg["intercept"] + deg["slope"] * xs,
                color="#C44E52", linewidth=1.6)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title(f"[2] 性能退化：斜率 {deg['slope']:+.2f} | "
                 f"Spearman {deg['spearman']:+.2f}（{deg['n_used']} 组合）")
    ax.set_xlabel(f"{tag} IS（IS 冠军）")
    ax.set_ylabel(f"{tag} OOS（IS 冠军）")
    fig.tight_layout()
    figs["degradation"] = fig

    # ---- [3] 亏损概率：IS 冠军 OOS SR 分布（零以下着色）----
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    vals = oos_v[ok]
    if vals.size:
        lo = min(float(vals.min()), 0.0)
        hi = max(float(vals.max()), 0.0)
        if hi <= lo:
            hi = lo + 1e-12
        h_edges = np.linspace(lo, hi, bins + 1)
        if np.isnan(result["loss_prob"]):
            ax.hist(vals, bins=h_edges, color="#4C72B0", alpha=0.85,
                    label=f"{tag} OOS")
        else:
            ax.hist(vals[vals >= 0], bins=h_edges, color="#4C72B0", alpha=0.85,
                    label=f"{tag} OOS >= 0")
            ax.hist(vals[vals < 0], bins=h_edges, color="#C44E52", alpha=0.85,
                    label=f"{tag} OOS < 0（亏损）")
        ax.legend(fontsize=8)
    ax.axvline(0.0, color="#C44E52", linestyle="--", linewidth=1.6)
    loss_txt = ("n/a" if np.isnan(result["loss_prob"])
                else f"{result['loss_prob']:.1%}")
    ax.set_title(f"[3] 亏损概率 = {loss_txt}"
                 f" | IS 冠军 OOS {tag} 分布（{deg['n_used']} 组合）")
    ax.set_xlabel(f"{tag} OOS（IS 冠军）")
    ax.set_ylabel("组合频数")
    fig.tight_layout()
    figs["loss"] = fig

    # ---- [4] 随机占优：分布对比 / CDF / SD2 三联 ----
    fig, axes = plt.subplots(1, 3, figsize=dom_figsize)
    ax = axes[0]
    ax.fill_between(centers, dom["hist_sel"], color="#C44E52", alpha=0.35)
    ax.plot(centers, dom["hist_sel"], color="#C44E52", linewidth=1.2,
            label=f"IS 冠军（{deg['n_used']}）")
    ax.fill_between(centers, dom["hist_all"], color="#4C72B0", alpha=0.25)
    ax.plot(centers, dom["hist_all"], color="#4C72B0", linewidth=1.2,
            label=f"全体配置池（{n_combos * result['N']:,}）")
    ax.set_title(f"[4a] OOS {tag} 分布对比（归一化）")
    ax.set_xlabel(f"{tag} OOS")
    ax.set_ylabel("概率质量")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(grid, dom["cdf_sel"], color="#C44E52", linewidth=1.6,
            label=f"IS 冠军的 OOS {tag}")
    ax.plot(grid, dom["cdf_all"], color="#4C72B0", linewidth=1.6,
            label=f"全体配置的 OOS {tag}（池）")
    ax.set_title(f"[4b] 随机占优 CDF{dom_note}：FSD = {'是' if dom['fsd_holds'] else '否'}"
                 f" | 反超 {dom['fsd_violation']:+.4f}")
    ax.set_xlabel(f"{tag} OOS")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(grid, dom["sd2"], color="#55A868", linewidth=1.6)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title(f"[4c] SD2（{'≥0 成立' if dom['sd2_holds'] else '存在 <0'}）")
    ax.set_xlabel(f"{tag} OOS")
    ax.set_ylabel("SD2[x]")

    fig.tight_layout()
    figs["dominance"] = fig

    if save_path:
        p = Path(save_path)
        for key, f in figs.items():
            f.savefig(p.with_name(f"{p.stem}_{key}{p.suffix or '.png'}"),
                      bbox_inches="tight")
    for f in figs.values():
        display(f)
        plt.close(f)
    return figs


def plot_dsr(res, dsr=None, figsize=(14, 4.2), save_path=None):
    """DSR 判定三联图（单画布）：[5a] 试验分布与冠军/门槛 | [5b] 门槛-试验数
    曲线 SR*(n) | [5c] z 尾概率判定。

    消费 cscv_pbo 结果（result["matrix"]）：[5a] 用 N 条全样本日频 SR 直方图
    （V 的实体来源）；dsr=None 时内部调 calc_dsr_from_matrix(result["matrix"])，
    已算过可直接传入（如 info = calc_dsr_from_matrix(out["matrix"])）。口径与
    calc_dsr_from_matrix 一致：全部日频未年化（DSR 公式口径），关键数字旁附
    年化换算（x sqrt(252)，仅供参考、不改判定）。[5b] 大 n 段为“同 V、独立
    试验”外推（图上已标注假设），追平点 = 曲线与 sr_obs 的对数二分交点。
    save_path 给定时落盘；展示后关闭，不返回 Figure（notebook 中 display 与
    返回值会双渲染，与 plot_topk 处理一致）。
    """
    import matplotlib.pyplot as plt
    from IPython.display import display
    from scipy.stats import norm

    M = res["matrix"]
    if dsr is None:
        dsr = calc_dsr_from_matrix(M)
    srs = (M.mean() / M.std()).to_numpy()      # N 条全样本日频 SR
    sr, star, margin = dsr["sr_obs"], dsr["sr_star"], dsr["margin"]
    z, V, N, T = dsr["z_score"], dsr["V"], dsr["N"], dsr["T"]
    skew, kurt = dsr["skew"], dsr["kurt"]
    gamma = np.euler_gamma
    ann = np.sqrt(252.0)                       # 日频 -> 年化换算（仅标注用）

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ---- [5a] 试验分布 + 冠军/门槛两线 ----
    ax = axes[0]
    ax.hist(srs, bins=60, color="#4C72B0", alpha=0.75, edgecolor="white")
    ax.axvline(sr, color="#C44E52", linewidth=1.8)
    ax.axvline(star, color="black", linestyle="--", linewidth=1.4)
    ymax = ax.get_ylim()[1]
    ax.annotate("", xy=(sr, ymax * 0.55), xytext=(star, ymax * 0.55),
                arrowprops=dict(arrowstyle="<->", color="#C44E52", lw=1.4))
    ax.text(0.02, 0.97,
            f"冠军 cand{dsr['champion']}：{sr:.4f}（年化 {sr * ann:.2f}）\n"
            f"门槛 SR*：{star:.4f}（年化 {star * ann:.2f}）\n"
            f"margin：{margin:.4f}（年化 {margin * ann:.2f}）",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(fc="white", alpha=0.75, ec="none"))
    ax.set_title(f"[5a] 试验分布与冠军/门槛（N={N}，T={T}，V={V:.2e}）")
    ax.set_xlabel("全样本日频 SR（年化 = ×√252）")
    ax.set_ylabel("候选频数")

    # ---- [5b] 门槛-试验数曲线 SR*(n)（H0 噪声 deflate 法则可视化）----
    ax = axes[1]

    def _sr_star(n):
        # p 在 n > ~1e15 时下溢为 1.0（ppf=inf），钳位保数值安全
        p1 = np.minimum(1.0 - 1.0 / n, 1.0 - 1e-15)
        p2 = np.minimum(1.0 - 1.0 / (n * np.e), 1.0 - 1e-15)
        return np.sqrt(V) * ((1 - gamma) * norm.ppf(p1)
                             + gamma * norm.ppf(p2))

    lo_l, hi_l = np.log10(2.0), 16.0
    if _sr_star(10.0 ** hi_l) >= sr:           # 对数二分求“追平点” n*
        for _ in range(80):
            mid = 0.5 * (lo_l + hi_l)
            if _sr_star(10.0 ** mid) < sr:
                lo_l = mid
            else:
                hi_l = mid
        n_star = 10.0 ** (0.5 * (lo_l + hi_l))
        hi_l = min(hi_l + 0.35, 16.0)
    else:
        n_star = None
        hi_l = 16.0
    ns = np.logspace(np.log10(2.0), hi_l, 500)
    ax.plot(ns, _sr_star(ns), color="#55A868", linewidth=1.6)
    ax.axhline(sr, color="#C44E52", linewidth=1.4)
    ax.axvline(N, color="gray", linestyle=":", linewidth=1.0)
    ax.plot([N], [star], marker="o", color="black", markersize=5)
    ax.annotate(f"实际 N={N}\nSR*={star:.4f}", xy=(N, star), xytext=(8, -20),
                textcoords="offset points", ha="left", va="top", fontsize=8)
    if n_star is not None:
        ax.plot([n_star], [sr], marker="o", color="#C44E52", markersize=6)
        ax.annotate(f"追平点 n*≈{n_star:.1e}\n（同 V、独立外推）",
                    xy=(n_star, sr), xytext=(-8, 14),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=8, color="#C44E52")
    ax.text(0.02, sr, f"冠军 {sr:.4f}（年化 {sr * ann:.2f}）",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=8, color="#C44E52")
    ax.text(0.02, 0.02,
            "SR*(n) = √V·[(1-γ)$\\Phi^{-1}$(1−1/n) + γ$\\Phi^{-1}$(1−1/(n·e))]\n"
            "（H0：N(0,V) 抽 n 个取 max 的期望；γ=欧拉常数）",
            transform=ax.transAxes, va="bottom", fontsize=7, color="#444444")
    ax.set_xscale("log")
    ax.set_ylim(0.0, sr * 1.30)
    ax.set_title("[5b] 门槛随试验数 SR*(n)：√V 定高度、N 定位置")
    ax.set_xlabel("试验数 n（log 轴；大 n 段为外推）")
    ax.set_ylabel("门槛 SR*（日频，未年化）")

    # ---- [5c] z 尾概率判定 ----
    ax = axes[2]
    z95 = float(norm.ppf(0.95))
    xs = np.linspace(-4.0, 5.0, 600)
    ax.plot(xs, norm.pdf(xs), color="#4C72B0", linewidth=1.6)
    ax.fill_between(xs, norm.pdf(xs), where=xs >= z, color="#C44E52",
                    alpha=0.35)
    ax.axvline(z, color="#C44E52", linewidth=1.6)
    ax.axvline(z95, color="gray", linestyle="--", linewidth=1.2)
    p_tail = 1.0 - dsr["dsr"]
    ax.annotate(f"右尾 = 1 − DSR ≈ {p_tail:.2%}\n（运气假设下的极端概率）",
                xy=(z, 0.018), xytext=(z + 0.55, 0.30),
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.0),
                fontsize=8, color="#C44E52")
    infl = max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2, 1e-8)
    eff = 1.0 - 1.0 / np.sqrt(infl)            # 忽略峰度偏度对 z 的影响幅度
    ax.text(0.02, 0.97,
            f"DSR = Φ(z) = {dsr['dsr']:.1%}（> 95% 通过）\n"
            f"z = {z:.2f}｜95% 门槛 z = {z95:.2f}\n"
            f"峰度 {kurt:.2f} → 方差膨胀 ×{infl:.3f}（z 影响 {eff:.1%}）\n"
            f"偏度 {skew:.3f} ≈ 0",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(fc="white", alpha=0.75, ec="none"))
    ax.set_title(f"[5c] z 尾概率判定（DSR = {dsr['dsr']:.1%}）")
    ax.set_xlabel("z（标准差数）")
    ax.set_ylabel("标准正态密度")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    display(fig)
    plt.close(fig)
    #return fig


def plot_topk(res, k=3, figsize=(10, 4), save_path=None):
    """topK 冠军候选评估区累计净值图（黑虚线 = 全员截面中位参照）。

    与 build_topk_report 同一 topK 口径（夺冠次数降序、并列按编号升序；
    实际数量 = min(k, 夺冠候选数)）。save_path 给定时落盘；展示后关闭，
    不返回 Figure（对象仍可再次 savefig）。
    """
    import matplotlib.pyplot as plt
    from IPython.display import display

    top_idx = _topk_indices(res, k)
    M = res["matrix"]
    fig, ax = plt.subplots(figsize=figsize)
    cum = (1.0 + M[top_idx]).cumprod()
    cum.columns = [f"cand{c}" for c in cum.columns]
    cum.plot(ax=ax)
    (1.0 + M.median(axis=1)).cumprod().plot(
        ax=ax, style="k--", linewidth=1.0, label="全员截面中位")
    ax.axhline(1.0, color="gray", linewidth=0.8)
    ax.set_title(f"top{len(top_idx)} 候选：评估区累计净值"
                 f"（黑虚线 = 全员截面中位参照）")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    display(fig)
    plt.close(fig)
    #return fig
