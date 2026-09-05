# -*- coding: utf-8 -*-
"""CPCV OOS 路径分析模块：离散采样、PBO/DSR 诊断、消融统计、换手率。

从 WalkForward+CPCV+PS.ipynb 提取（notebook cell 8/24/35/38/52），函数行为
与原 notebook 一致，无 notebook 全局状态依赖（bench、mpps 等由调用方传入）。

用法::

    # 段单元采样（段单元由 wf_cpcv_search.build_test_parts 产出）
    from cpcv_analysis import discrete_lhs_safe, diagnose_oos_lhs
    samples = discrete_lhs_safe(all_test_parts, n_samples=1000, seed=42)
    res = diagnose_oos_lhs(samples, n_trials=1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm
from scipy.integrate import quad
from scipy.stats.qmc import LatinHypercube
from skfolio import MultiPeriodPortfolio


# ---------------------------------------------------------------------------
# 离散 LHS 采样
# ---------------------------------------------------------------------------
def discrete_lhs_safe(nested_lists, n_samples, seed=None):
    """离散LHS采样 - 基于scipy实现，不要求元素可比较。

    Parameters
    ----------
    nested_lists : List[List[Any]]，各子列表长度可不等（每个子列表为一维的
        候选集合，如某 (折, 块) 段单元上的全部路径 Portfolio）
    n_samples : int，采样数量（每条样本从各候选集合中取一个元素组成元组）
    seed : int | None，随机种子

    Returns
    -------
    List[Tuple[Any, ...]]，共 n_samples 条
    """
    n_factors = len(nested_lists)
    n_levels = np.array([len(lst) for lst in nested_lists])  # 支持各维度层级数不同

    # ① 连续空间 LHS（scipy 内部用优化算法，比手写 permutation 更均匀）
    sampler = LatinHypercube(d=n_factors, seed=seed)
    unit_samples = sampler.random(n=n_samples)  # (n_samples, n_factors) ∈ [0,1)

    # ② 向量化映射到离散层级索引（一行替代循环）
    level_indices = np.clip(
        (unit_samples * n_levels[np.newaxis, :]).astype(np.intp),
        0,
        n_levels[np.newaxis, :] - 1
    )  # (n_samples, n_factors)

    # ③ 构建 object 查找表（逐元素赋值，避免触发 __array__ 协议）
    max_levels = int(n_levels.max())
    lookup = np.empty((n_factors, max_levels), dtype=object)
    for j, lst in enumerate(nested_lists):
        for k, obj in enumerate(lst):
            lookup[j, k] = obj          # ← 单个 object 赋值，安全

    col_idx = np.arange(n_factors)[np.newaxis, :]
    samples_array = lookup[col_idx, level_indices]

    return [tuple(row) for row in samples_array]


# ---------------------------------------------------------------------------
# PBO / DSR 诊断（纯 OOS 策略诊断）
# ---------------------------------------------------------------------------
def _expected_max_std_normal(n):
    """E[max_N Z]：N 个独立标准正态最大值的期望。

    数值积分对任意 N 精确；Bailey & López de Prado(2014) 渐近式仅对大 N
    有效，N 小时失真。
    """
    if n <= 1:
        return 0.0

    def integrand(z):
        return z * norm.pdf(z) * norm.cdf(z) ** (n - 1)

    val, _ = quad(integrand, -8.0, 8.0, limit=400)
    return n * val


def diagnose_oos_lhs(lhs_paths, annual_factor=252, risk_free_rate=0.02, n_trials=1):
    """纯 OOS 策略诊断（复用离散 LHS 采样路径 + MultiPeriodPortfolio）。

    Parameters
    ----------
    lhs_paths : list，discrete_lhs_safe 的返回结果——每条元素是一个
        MultiPeriodPortfolio 的 block 组合（tuple of Portfolio），即采样后的
        路径列表，无需重新采样
    annual_factor : int，年化交易日
    risk_free_rate : float，无风险利率
    n_trials : int，DSR 的独立试验数 N（整个流程被独立评估并挑选最优的次数）；
        纯 OOS 单流程无跨数据集挑选 -> 取 1

    Returns
    -------
    dict : prob_loss / dsr / sr_obs / e_max / z_score / mc_sharpes /
        mc_mdds / T_total / n_trials
    """
    n_samples = len(lhs_paths)
    # OOS 总天数 = 首条路径各 block returns 长度之和（各路径等长）
    T_total = sum(len(np.asarray(part.returns)) for part in lhs_paths[0])

    print(f"[Info] {n_samples} LHS paths | OOS {T_total} days")

    # ==========================================================
    # 1. 逐条路径构造 MultiPeriodPortfolio → returns / MDD 矩阵
    # ==========================================================
    full_paths = np.empty((n_samples, T_total))
    mc_mdds = np.empty(n_samples)
    for i, path in enumerate(lhs_paths):
        mpp = MultiPeriodPortfolio(path)
        rets = np.asarray(mpp.returns, dtype=float)
        full_paths[i] = np.nan_to_num(rets, nan=0.0)
        mc_mdds[i] = mpp.max_drawdown          # skfolio 口径（负值，同绩效表 max_drawdown）

    # ==========================================================
    # 2. Sharpe
    # ==========================================================
    excess = full_paths - (risk_free_rate / annual_factor)
    means = excess.mean(axis=1)
    stds = full_paths.std(axis=1, ddof=1)
    stds = np.where(stds < 1e-12, 1e-12, stds)
    oos_sharpes = (means / stds) * np.sqrt(annual_factor)

    # ==========================================================
    # 3. DSR（偏度/峰度取首条采样路径，与 Sharpe 同用超额收益）
    # ==========================================================
    gamma3 = skew(excess[0])
    gamma4 = kurtosis(excess[0]) + 3

    sr_obs = np.median(oos_sharpes)

    e_max = _expected_max_std_normal(n_trials)
    # Lo(2002) Sharpe 方差项: 系数 (γ4-1)/4 (原代码误写 /24)
    inflation = max(1 - gamma3 * sr_obs + (gamma4 - 1) * sr_obs ** 2 / 4, 1e-8)
    # z = t(SR_obs) - e_max: e_max 为标准化 z 单位, 不能再乘 sqrt(T-1)
    z_score = np.sqrt(max(T_total - 1, 1)) * sr_obs / np.sqrt(inflation) - e_max
    dsr = norm.cdf(z_score)
    prob_loss = np.mean(oos_sharpes <= 0)

    # ==========================================================
    # 4. Report
    # ==========================================================
    print("\n" + "=" * 60)
    print("          纯 OOS 策略诊断报告 (LHS)")
    print("=" * 60)
    print(f"  ✅ P(SR<=0)              : {prob_loss:.2%}")
    print(f"  ✅ Sharpe 90% CI         : [{np.percentile(oos_sharpes, 5):.3f}, "
          f"{np.percentile(oos_sharpes, 95):.3f}]")
    print(f"  ✅ MDD median / worst-5%  : {np.median(mc_mdds):.2%} / "
          f"{np.percentile(mc_mdds, 95):.2%}")
    print("-" * 60)
    print(f"  ⚠️ SR_obs (median)       : {sr_obs:.4f}")
    print(f"  ⚠️ DSR                   : {dsr:.4f}  (z={z_score:.2f}, n_trials={n_trials})")
    print("=" * 60)

    return {
        "prob_loss": prob_loss, "dsr": dsr, "sr_obs": sr_obs,
        "e_max": e_max, "z_score": z_score,
        "mc_sharpes": oos_sharpes, "mc_mdds": mc_mdds,
        "T_total": T_total, "n_trials": n_trials
    }


# ---------------------------------------------------------------------------
# 换手率 / 消融统计基础件
# ---------------------------------------------------------------------------
def turnover_series(oos_mpt):
    """相邻 OOS 段的持仓换手率 (0~1)：weights(数组) + assets 对齐。
    oos_mpt : MultiPeriodPortfolio，OOS 段组合
    """
    tos = []
    prev_w, prev_a = None, None
    for ptf in oos_mpt:
        w, a = ptf.weights, ptf.assets          # 数组, 与 assets 对应
        if prev_w is not None:
            common = np.union1d(prev_a, a)
            w1 = np.zeros(len(common)); w2 = np.zeros(len(common))
            w1[np.isin(common, prev_a)] = prev_w
            w2[np.isin(common, a)] = w
            tos.append(0.5 * np.abs(w1 - w2).sum())
        prev_w, prev_a = w, a
    return pd.Series(tos)


def extract_metrics(mpp_list, metrics):
    """从 MPP 列表抽取指标矩阵: DataFrame(列=metrics, 行=各 MPP)。"""
    return pd.DataFrame([[getattr(m, met) for met in metrics] for m in mpp_list],
                        columns=metrics)


def boot_diff_ci(a, b, n_boot=3000, seed=42):
    """bootstrap 均值差及 95% CI（两样本独立重采样）。"""
    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    diffs = np.array([rng.choice(a, na, replace=True).mean()
                      - rng.choice(b, nb, replace=True).mean() for _ in range(n_boot)])
    return diffs.mean(), np.percentile(diffs, [2.5, 97.5])


# ---------------------------------------------------------------------------
# 年化分布报告（多路径分布 vs WF OOS 顺序路径）
# ---------------------------------------------------------------------------
def report_ann_distribution(path_anns, oos_ann):
    """LHS 多路径年化分布 vs WF OOS 顺序路径：打印分布统计并绘制直方图对比。

    Parameters
    ----------
    path_anns : array-like，各 LHS 路径的 annualized_mean（调用方从 oos_mpts 提取）
    oos_ann : float，WF OOS 顺序路径的 annualized_mean（图中红线标记）

    Notes
    -----
    从 WF+CPCV+PS_252.ipynb 抽取；plt.show() 语义保留，标题/标签全英文
    避免中文字体依赖。
    """
    path_anns = np.asarray(path_anns, dtype=float)

    print(f"LHS多路径分布({len(path_anns)}条): 均值 {path_anns.mean():.4f} | 中位 {np.median(path_anns):.4f} | "
          f"5% {np.percentile(path_anns, 5):.4f} | 95% {np.percentile(path_anns, 95):.4f}")
    print(f"WF真实路径: {oos_ann:.4f} | 分布中 {(oos_ann - path_anns.mean()) / path_anns.std():.2f}σ | "
          f"分位 {(path_anns < oos_ann).mean():.1%}")

    plt.figure(figsize=(10, 4))
    plt.hist(path_anns, bins=50, alpha=0.7)
    plt.axvline(x=oos_ann, c='red', ls='--', label=f'OOS path {oos_ann:.3f}')
    plt.axvline(x=np.median(path_anns), c='green', ls='--', label=f'median {np.median(path_anns):.3f}')
    plt.legend()
    plt.title('LHS path annualized distribution vs WF OOS sequential path')
    plt.show()


# ---------------------------------------------------------------------------
# OOS vs 基准净值对比图
# ---------------------------------------------------------------------------
def plot_oos_vs_bench(oos_mpt, bench_ret):
    """WF OOS 策略净值 vs 基准净值（沪深300ETF）对比图。

    Parameters
    ----------
    oos_mpt : MultiPeriodPortfolio，WF OOS 顺序路径组合（其 returns_df 决定区间）
    bench_ret : pd.Series，基准日收益序列；长于 OOS 区间时自动裁剪到
        [start, end] 并对齐 OOS 索引（ffill 补缺口）

    Notes
    -----
    从 WF+CPCV+PS_252.ipynb 抽取；图内中文字体用 rc_context 局部设置，
    不依赖调用方全局 rcParams。
    """
    returns_df = oos_mpt.returns_df
    start = returns_df.index[0]
    end = returns_df.index[-1]

    oos_nav = (1 + returns_df).cumprod()
    bench_ret = bench_ret.loc[start:end].reindex(oos_nav.index).ffill()
    bench_nav = (1 + bench_ret).cumprod()

    with plt.rc_context({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False}):
        plt.figure(figsize=(10, 4))
        plt.plot(oos_nav.index, oos_nav.values, label="OOS (参数自适应+t固定)")
        plt.plot(bench_nav.index, bench_nav.values, label="沪深300ETF", alpha=0.7)
        plt.legend()
        plt.title("OOS vs 沪深300ETF 净值")
        plt.show()


# ---------------------------------------------------------------------------
# 相对基准超额统计（几何/算术超额、IR、胜率）
# ---------------------------------------------------------------------------
def excess_vs_bench(oos_mpts, bench_ret):
    """OOS 路径相对基准（沪深300）的超额口径统计，一行一路径。

    Parameters
    ----------
    oos_mpts : list[MultiPeriodPortfolio]，OOS 顺序路径组合列表
    bench_ret : pd.Series，基准日收益序列；区间按首条路径裁剪，各路径
        相对收益逐条对齐（reindex + ffill）

    Returns
    -------
    pd.DataFrame : 行 = 各路径；列 = ann_excess（几何超额 = 相对净值年化）
        | arith_excess（算术超额）| IR | winrate（日胜率）。
        并打印 5%/50%/95%/mean 摘要（原 notebook 中 log_result 写入由调用方处理）。
    """
    returns_df = oos_mpts[0].returns_df
    start = returns_df.index[0]
    end = returns_df.index[-1]
    bench_full = bench_ret.loc[start:end]
    bench_ann = (1 + bench_full).prod() ** (252 / len(bench_full)) - 1

    excess_stats = []
    for mpp in oos_mpts:
        r = mpp.returns_df
        e = r - bench_full.reindex(r.index).ffill()
        excess_stats.append([
            (1 + mpp.annualized_mean) / (1 + bench_ann) - 1,  # 几何超额(相对净值年化)
            e.mean() * 252,                                    # 算术超额
            e.mean() / e.std() * np.sqrt(252),                 # IR (口径不变)
            (e > 0).mean(),                                    # 日胜率 (口径不变)
        ])

    df_ex = pd.DataFrame(excess_stats, columns=["ann_excess", "arith_excess", "IR", "winrate"])
    print(df_ex.describe(percentiles=[0.05, 0.5, 0.95]).loc[["5%", "50%", "95%", "mean"]].round(4))
    return df_ex
