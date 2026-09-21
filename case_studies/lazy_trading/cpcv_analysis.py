# -*- coding: utf-8 -*-
"""CPCV OOS 路径分析模块：离散采样、PBO/PSR 诊断、消融统计、换手率。

从 WalkForward+CPCV+PS.ipynb 提取（notebook cell 8/24/35/38/52），函数行为
与原 notebook 一致，无 notebook 全局状态依赖（bench、mpps 等由调用方传入）。

用法::

    # 段单元采样（段单元由 wf_cpcv_search.build_test_parts 产出）
    from cpcv_analysis import discrete_lhs_safe, diagnose_oos_lhs, empirical_deflate
    samples = discrete_lhs_safe(all_test_parts, n_samples=1000, seed=42)
    res = diagnose_oos_lhs(samples)
    d = empirical_deflate(fold["study"])        # 复合目标每折 deflate（键: p_luck/margin/...）
    t = efron_null_fdr(fold["study"])["table"]  # per-trial p / local fdr / q_bh / q_by
    # calc_dsr(fold["study"])                   # 旧口径仍保留, 仅适用 SR 类目标, 复合目标勿用
    r = rank_pbo_logit(X, folds)                # 全参数 train/test 重放 → CPCV-best 排名分位/λ/PBO
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm, genpareto, gaussian_kde
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
# PBO / PSR 诊断（纯 OOS 策略诊断：固定参数单次应用 -> 无选择偏差）
# ---------------------------------------------------------------------------
def diagnose_oos_lhs(lhs_paths, annual_factor=252, risk_free_rate=0.02):
    """纯 OOS 策略诊断（复用离散 LHS 采样路径 + MultiPeriodPortfolio）。

    每条 LHS 路径 = 同一组已筛选固定参数的 OOS 多段收益拼接实现；参数在
    OOS 前已固定、每条路径只应用一次，路径间不存在"挑选最优"竞争 -> 无
    选择偏差，不需要 DSR deflation（Bailey & López de Prado(2014) 的 DSR
    仅用于校正 IS 上 N 次试验挑选导致的虚高 SR）。此处按路径计算 PSR(0)，
    仅校正估计误差与非正态性（Bailey & López de Prado, 2012）。

    Parameters
    ----------
    lhs_paths : list，discrete_lhs_safe 的返回结果——每条元素是一个
        MultiPeriodPortfolio 的 block 组合（tuple of Portfolio），即采样后的
        路径列表，无需重新采样
    annual_factor : int，年化交易日
    risk_free_rate : float，无风险利率

    Returns
    -------
    dict : prob_loss / psr_median / psr_5 / sr_obs / mc_psrs /
        mc_sharpes / mc_mdds / T_total
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
    # 2. Sharpe / PSR(0)（逐路径：SR 与该路径自身偏度/峰度配对）
    # ==========================================================
    excess = full_paths - (risk_free_rate / annual_factor)
    means = excess.mean(axis=1)
    stds = full_paths.std(axis=1, ddof=1)
    stds = np.where(stds < 1e-12, 1e-12, stds)
    oos_sharpes = (means / stds) * np.sqrt(annual_factor)

    # PSR(0): 无跨路径挑选 -> 基准 SR*=0，无需 deflation
    # Lo(2002) Sharpe 方差项: 系数 (γ4-1)/4 (原代码误写 /24)
    gamma3 = skew(excess, axis=1)
    gamma4 = kurtosis(excess, axis=1) + 3
    inflation = np.maximum(
        1 - gamma3 * oos_sharpes + (gamma4 - 1) * oos_sharpes ** 2 / 4, 1e-8)
    z_scores = np.sqrt(max(T_total - 1, 1)) * oos_sharpes / np.sqrt(inflation)
    mc_psrs = norm.cdf(z_scores)

    sr_obs = np.median(oos_sharpes)
    prob_loss = np.mean(oos_sharpes <= 0)
    psr_median = np.median(mc_psrs)
    psr_5 = np.percentile(mc_psrs, 5)

    # ==========================================================
    # 3. Report
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
    print(f"  ⚠️ PSR(0) median / 5%    : {psr_median:.2%} / {psr_5:.2%}")
    print("=" * 60)

    return {
        "prob_loss": prob_loss, "psr_median": psr_median, "psr_5": psr_5,
        "sr_obs": sr_obs, "mc_psrs": mc_psrs,
        "mc_sharpes": oos_sharpes, "mc_mdds": mc_mdds,
        "T_total": T_total
    }


# ---------------------------------------------------------------------------
# 经验 DSR：IS 参数搜索 trial 分数（CPCV val）的选择偏差校正
# ---------------------------------------------------------------------------
def trial_param_key(params: dict) -> tuple:
    """参数组合 → 去重键（float 值 round 到 9 位消除采样浮点残差，str/int 原样）。

    与 calc_dsr / empirical_deflate 的 N 统计同口径：随机/TPE 采样会重复
    命中同一参数组，重复不增加候选信息，去重后才能得到真实试验次数 N。
    供 calc_dsr / empirical_deflate 与 top_trials（Top-K 提取）共用，
    保证去重口径一致。
    """
    return tuple(sorted(
        (k, round(v, 9) if not isinstance(v, str) else v)
        for k, v in params.items()))


def top_trials(trials, k: int = 5) -> list:
    """按分数降序、参数去重取前 k 个 trial（与 empirical_deflate 同去重口径）。

    Parameters
    ----------
    trials : list，optuna study.trials
    k : int，Top-K 候选数

    Returns
    -------
    list : 分数最高的 k 个不同参数组合的 trial，按分数降序排列。
        （分数相同的重复参数组合只保留先出现的 trial。）
    """
    valid = [t for t in trials
             if t.values is not None and np.isfinite(t.values[0])]
    valid.sort(key=lambda t: float(t.values[0]), reverse=True)
    seen, out = set(), []
    for t in valid:
        key = trial_param_key(t.params)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= k:
            break
    return out


def calc_dsr(sr_or_study, n_bootstrap=10000, seed=42):
    """经验 DSR（Bailey & López de Prado 2014）：N 次 trial 挑最优的选择偏差校正。

    与 diagnose_oos_lhs（OOS 侧固定参数单次应用、无选择偏差、只做 PSR）
    互补：本函数作用于 **IS 参数搜索**——每折内层 Optuna 在 CPCV val 上
    评估 N 个 trial 并选最优，被选中者的分数带选择偏差，需按实际试验
    次数 deflate。

    Parameters
    ----------
    sr_or_study : optuna Study | array-like
        Study：trials[i].values[0] 为该 trial 的分数（当前搜索目标 = mean
        path 年化夏普，见 wf_cpcv_search.inner_cpcv_score）；真实试验次数
        N = params 去重后的 trial 数（随机采样会重复命中同一参数组，重复
        不增加候选信息）。
        array-like：直接 N 个 SR 观测，N 默认取数组长度。
    n_bootstrap : int，蒙特卡洛重采样次数
    seed : int | None，随机种子

    Notes
    -----
    deflate 基准（Bailey & López de Prado 2014 的 MC 形式）：H0 = 策略无
    真实优势时，N 次试验分数为 0 均值噪声，噪声尺度 V 取观测分数序列
    （study 返回的各 trial mean path ASR）的方差——即 H0 下 trial 分数
    ~ N(0, V)。从 N(0, V) 抽 N 个取最大值 = "纯运气挑最优"分布：其均值
    对应解析式 √V·E[Z_(N)]（≈√V[(1−γ)Φ⁻¹(1−1/N)+γΦ⁻¹(1−1/(Ne))]，
    γ = 欧拉常数），MC 同时给出分布分位（max_p95）。观测方差混入真实
    参数差异 → V 上偏 → 运气基准偏高 → deflate 偏严（保守方向）。
    返回 dsr = 最优分数超过该 H0 运气基准的经验概率（Laplace 平滑
    (+1)/(B+1)，避免极端 0/100% 假象）。

    Returns
    -------
    dict : dsr / p_luck / sr_obs / exp_max / margin / max_p95 /
        n_trials / n_trials_total / srs / boot_max
    """
    # --- 解析输入：SR 观测数组 + 真实试验次数 N ---
    if hasattr(sr_or_study, "trials"):
        trials = [t for t in sr_or_study.trials
                  if t.values is not None and np.isfinite(t.values[0])]
        srs = np.array([float(t.values[0]) for t in trials])
        # 参数组合去重 → 真实 N：键口径统一走 trial_param_key
        # （与 top_trials 的 Top-K 提取共用，保证两处去重一致）
        n_trials = len({trial_param_key(t.params) for t in trials})
        n_trials_total = len(trials)
    else:
        srs = np.asarray(sr_or_study, dtype=float)
        srs = srs[np.isfinite(srs)]
        n_trials = n_trials_total = len(srs)
    if srs.size == 0 or n_trials < 1:
        raise ValueError("calc_dsr: 无有效 trial 分数（study 需含 COMPLETE trials）")

    # --- H0 蒙特卡洛：零均值噪声中抽 N 个取 max = 纯运气最优分布 ---
    # V = 观测 trial 分数序列方差: H0 下无真实优势时 trial 间散布全来自
    # 运气 → 观测方差是噪声尺度的(上偏)估计, deflate 偏严(保守方向)
    v_noise = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    rng = np.random.default_rng(seed)
    boot_max = rng.normal(0.0, np.sqrt(v_noise),
                          size=(n_bootstrap, n_trials)).max(axis=1)
    exp_max = float(boot_max.mean())
    max_p95 = float(np.percentile(boot_max, 95))
    sr_obs = float(srs.max())
    margin = sr_obs - exp_max
    n_exceed = int(np.count_nonzero(boot_max >= sr_obs))
    p_luck = n_exceed / n_bootstrap
    dsr = 1.0 - (n_exceed + 1) / (n_bootstrap + 1)   # Laplace 平滑

    return {"dsr": dsr, "p_luck": p_luck, "sr_obs": sr_obs,
            "exp_max": exp_max, "margin": margin, "max_p95": max_p95,
            "n_trials": n_trials, "n_trials_total": n_trials_total,
            "srs": srs, "boot_max": boot_max}


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


def fold_paths_frame(fold_paths, metrics):
    """多路径压测结果 → 绩效表：行 = (折, 路径)，列 = metrics。

    fold_paths : {fold: {path_id: [Portfolio块, ...]}}，adaptive_multi_paths
        返回结构（每条路径的 test 块序列，块间连续覆盖该折 OOS 段）。
    每条 (折, 路径) 的块序列拼接为 MultiPeriodPortfolio 后取 skfolio
    现成属性（不自算指标），供消融 / Top-K 候选对比 / 路径分布分析共用。

    Returns
    -------
    pd.DataFrame : MultiIndex 行 (fold, path)，列 = metrics
    """
    rows, idx = [], []
    for i in sorted(fold_paths):
        for pid in sorted(fold_paths[i], key=int):
            mpp = MultiPeriodPortfolio(fold_paths[i][pid])
            rows.append([getattr(mpp, met) for met in metrics])
            idx.append((i, pid))
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
        idx, names=["fold", "path"]), columns=metrics)


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


# ---------------------------------------------------------------------------
# Population 多路径收益/净值曲线（matplotlib 版，替代 plotly 防输出膨胀）
# ---------------------------------------------------------------------------
def plot_paths_curve(oos_mpts, cumprod=True, labels=None, color=None, lw=0.8,
                     alpha=0.5, mean_line=True, figsize=(12, 6),
                     title=None, save_path=None):
    """Population 多路径收益曲线：cumprod 累计净值 / 原始收益率。

    替代 plotly 系多路径可视化：plotly 会把整个 figure JSON 写入
    notebook outputs（多路径下可膨胀至数十 MB，曾致 ipynb 超 VSCode
    上限打不开）；matplotlib 输出为 PNG 位图，路径数再多体积恒定。

    显示方式为 display(fig) + close：只渲染本图；不能用 plt.show()，
    其语义是渲染全局所有打开的 figure，会把其他 cell 中挂起的图
    一并带出（显示到错误位置，即图被"劫持"）。

    Parameters
    ----------
    oos_mpts : Population | list[MultiPeriodPortfolio]
        多路径组合集合；逐元素取 .returns_df（带时间索引的收益率 Series）
    cumprod : bool
        True → (1+r).cumprod() 累计净值（初始=1）；False → 原始收益率曲线
    labels : list[str] | None
        逐路径曲线图例标签，长度需与 oos_mpts 一致；提供时（且未显式
        指定 color）各路径按色环逐条配色（超出色环换线型二次区分），
        图例可逐条对应；图例另保留"跨路径均值"（当 mean_line=True）
    color : str | None
        None（默认）→ 有 labels 时逐条配色、无 labels 时统一 tab:blue；
        显式指定 → 全部曲线同色
    mean_line : bool，是否叠加跨路径均值粗线（黑）
    save_path : str | None，给定则存图（dpi=150），可完全不占 notebook 输出

    Returns
    -------
    None : 仅显示/存图，不返回对象（避免返回值被 notebook 二次渲染）
    """
    if labels is not None and len(labels) != len(oos_mpts):
        raise ValueError(f"labels 长度需与 oos_mpts 一致：{len(labels)} vs {len(oos_mpts)}")
    curves = []
    for ptf in oos_mpts:
        r = ptf.returns_df
        curves.append((1 + r).cumprod() if cumprod else r)

    # 配色：显式 color → 全部同色；无 labels → 统一 tab:blue；有 labels →
    # 按 rcParams 色环逐条取色（项目 matplotlibrc 定义 6 色），超出色环用
    # 线型（虚线/点线/点划线）二次区分，保证图例与曲线一一对应
    if color is not None:
        line_specs = [(color, "-")] * len(curves)
    elif labels is None:
        line_specs = [("tab:blue", "-")] * len(curves)
    else:
        cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        linestyles = ["-", "--", ":", "-."]
        line_specs = [(cycle_colors[i % len(cycle_colors)],
                       linestyles[(i // len(cycle_colors)) % len(linestyles)])
                      for i in range(len(curves))]

    with plt.rc_context({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=figsize)
        for i, y in enumerate(curves):
            c, ls = line_specs[i]
            ax.plot(y.index, y.values, color=c, linestyle=ls, lw=lw, alpha=alpha,
                    label=None if labels is None else labels[i])
        if mean_line:
            mean_y = pd.concat(curves, axis=1).mean(axis=1)
            ax.plot(mean_y.index, mean_y.values, color="black", lw=2.0,
                    label="跨路径均值")
        ax.set_title(title or ("OOS 多路径累计净值" if cumprod else "OOS 多路径收益率"))
        ax.set_ylabel("净值（初始=1）" if cumprod else "收益率")
        ax.grid(alpha=0.3)
        if mean_line or labels is not None:
            ax.legend()
        if save_path:
            fig.savefig(save_path, dpi=150)
        # 精确显示本图并关闭：plt.show() 会渲染全局所有打开的 figure，
        # 把其他 cell 中挂起的图一并带出——这正是图被"劫持"的根源
        from IPython.display import display
        display(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 经验零分布多检验（复合目标）：GPD 上尾 deflate + Efron 主体法 local fdr
# ---------------------------------------------------------------------------
# 背景：calc_dsr 对 H0 的假设（trial 分数 ~ N(0, V)、max-of-N 用高斯极值）
# 只对 SR 类目标成立；复合目标 asr*0.5 - maxdd + skew 在 H0 下既有结构性
# 水平偏移（-E[maxdd] 负基线），分布也有界且偏斜，两个前提都不满足。
# 本段改为"用 trial 分数群自身估计零分布"，只读 study 分数即可：
#   empirical_deflate  变体1：GPD 上尾（POT；拟合剔除最优的"噪声律"，
#                      全量拟合作保守对照）→ per-fold 判据量（p_luck/
#                      margin/max_p95/n_trials；不复用 calc_dsr 的
#                      dsr/sr_obs 命名——分数非 Sharpe，DSR 名不副实）；
#   efron_null_fdr     变体2：中位数/MAD 主体法 → per-trial 尾部 p 值、
#                      local fdr 与 BH/BY q 值（BH/BY 的合法输入源）。
# 分层约定：per-fold 判决用 empirical_deflate；per-trial 集合筛选才用
# efron_null_fdr，两者勿在同一层叠加校正。


def _unique_trial_scores(sr_or_study):
    """study | 数组 → 去重 trial 分数（每个参数组合保留最高分）+ 评估总数。

    与 calc_dsr 的 N 口径一致：params 经 trial_param_key 去重（重复采样不
    增加候选信息）；同一组合的多次评估只留最高分（确定性评估下各次分数
    相同；留最高分对数值残差也更稳健）。
    """
    if hasattr(sr_or_study, "trials"):
        trials = [t for t in sr_or_study.trials
                  if t.values is not None and np.isfinite(t.values[0])]
        best = {}
        for t in trials:
            key = trial_param_key(t.params)
            v = float(t.values[0])
            if key not in best or v > best[key]:
                best[key] = v
        srs = np.array(list(best.values()), dtype=float)
        n_total = len(trials)
    else:
        srs = np.asarray(sr_or_study, dtype=float)
        srs = srs[np.isfinite(srs)]
        n_total = len(srs)
    return srs, n_total


def _fit_gpd_tail(srs, u_frac=0.80, min_exceed=10, xi_min=-0.10, xi_max=0.5):
    """POT/GPD 上尾拟合：超阈量 y = s - u 拟合广义 Pareto（MLE, floc=0）。

    阈值 u 取第 k+1 大分数，k = clip(ceil((1-u_frac)*m), min_exceed,
    max(3, m//2))——保证超阈点含最大值、阈值落在上半分布且不少于 3 个。
    MLE 失败或 xi 越界 [xi_min, xi_max] 时回退：xi 截断 + 矩估计 beta =
    mean(y)*(1-xi)。xi_min 是负向正则化底（默认 -0.10，校准实验选定）：
    负 xi（有界尾）在小样本下主要由局部曲率驱动，会把人工支撑上界放到
    观测范围内，使 max 概率虚假塌到 0（反保守）；压到缓负值=改用更重
    尾巴，方向保守。

    Returns
    -------
    (u, n_exceed, xi, beta) | None : 上尾正超阈点不足 3 个（并列/退化）时
        返回 None，调用方走保守退化分支。
    """
    s_desc = np.sort(srs)[::-1]
    m = len(s_desc)
    k = int(np.ceil((1.0 - u_frac) * m))
    k = min(max(k, min_exceed), max(3, m // 2))
    u = float(s_desc[k])                     # 第 k+1 大 → 严格超越 u 者 ≈ k 个
    y = s_desc[:k] - u
    y = y[y > 0]                             # 并列于 u 的点不算超阈
    if y.size < 3:
        return None
    xi = beta = np.nan
    try:
        xi_hat, _, beta_hat = genpareto.fit(y, floc=0)
        if np.isfinite(xi_hat) and np.isfinite(beta_hat) and beta_hat > 0:
            xi, beta = float(xi_hat), float(beta_hat)
    except Exception:
        pass
    if not (np.isfinite(xi) and np.isfinite(beta)):
        xi, beta = 0.0, float(y.mean())      # 回退：指数尾
    if xi < xi_min or xi > xi_max:           # xi 越界：截断 + 矩估计尺度
        xi = float(np.clip(xi, xi_min, xi_max))
        beta = float(y.mean() * (1.0 - xi))
    return u, int(y.size), xi, beta


def _gpd_tail_survival(s, u, xi, beta, q_tail):
    """GPD 上尾的无条件生存概率 P(S > s)（单点超越概率），q_tail = k/m。"""
    z = 1.0 + xi * (s - u) / beta
    z = max(z, 1e-12)                        # xi<0 时模型上界外的数值保护
    if abs(xi) < 1e-9:
        return float(q_tail * np.exp(-(s - u) / beta))
    return float(q_tail * z ** (-1.0 / xi))


def _degenerate_deflate(srs, n_total, why):
    """保守退化输出：无法估计经验零分布时 p_luck=1（最优分按纯运气论处）。"""
    s_obs = float(np.max(srs)) if srs.size else np.nan
    return {"p_luck": 1.0, "best_score": s_obs, "exp_max": s_obs, "margin": 0.0,
            "max_p95": s_obs, "n_trials": int(srs.size),
            "n_trials_total": n_total, "scores": srs, "u": np.nan,
            "n_exceed": 0, "xi": np.nan, "beta": np.nan, "xi_full": np.nan,
            "beta_full": np.nan, "p_luck_full": np.nan, "p_luck_mc": 1.0,
            "method": "gpd_pot", "degenerate": why}


def empirical_deflate(score_or_study, n_sim=200_000, seed=42, u_frac=0.80,
                      min_exceed=10, xi_min=-0.10):
    """经验零分布 deflate（变体1：GPD 上尾）：复合目标的逐折选择偏差判据。

    与 calc_dsr 同一输入消费结构（study|数组，参数去重口径相同），但
    零分布来源不同：不再假设 trial 分数 ~ N(0, V)，而是把分数群上尾
    当作噪声律的经验估计（POT/GPD），max-of-m 分位由该模型给出——

    * 结构性水平偏移（复合分数的 -E[maxdd] 负基线）由分数群位置吸收；
    * 分布形状（有界、偏斜）由形状参数 xi 刻画，不用高斯极值公式。

    零分布语义：p_luck = P(H0 下 m 次可交换试验的最优分数 ≥ s_obs)，
    H0 = "分数群其余部分是该随机系统的噪声散布"（多数 trial 无真信号；
    参数异质性把零分布拉宽 → 保守方向）。随机采样模式成立度最高；TPE
    自适应会让假设变弱，判读时打折。

    上尾拟合剔除最优本身（trim-top-1）：若把最优含进拟合，它会把上尾
    尺度撑大（遮蔽效应），真信号也会被打成"侥幸"；剔除后 H0 下 p_luck
    近似均匀、真信号下 p_luck → 0。假定至多一个真信号，多个时偏保守；
    另行给出全量拟合（含最优）的保守对照 p_luck_full。

    Parameters
    ----------
    score_or_study : optuna Study | array-like（解析与 calc_dsr 同口径）
    n_sim : int，max 分布 MC 次数（exp_max / max_p95 / p_luck_mc）
    seed : int | None，随机种子
    u_frac : float，POT 上尾占比（阈值取上尾 1-u_frac）
    min_exceed : int，最小超阈点数
    xi_min : float，GPD 形状下界（负向正则化，默认 -0.10；调低更激进、
        调高更保守）

    Returns
    -------
    dict : p_luck / best_score / exp_max / margin / max_p95 /
        n_trials / n_trials_total / scores / u / n_exceed / xi / beta /
        xi_full / beta_full / p_luck_full / p_luck_mc / method / degenerate
        （不复用 calc_dsr 的 dsr/sr_obs 命名——分数非 Sharpe，DSR 名不副实；
        best_score = 最优 trial 分数；分数全等/上尾并列不可拟合或有效
        trial 过少（<7）时返回 p_luck=1、degenerate 非空的保守退化输出）

    Notes
    -----
    * p_luck 解析式：1 - [1 - q_tail*(1+xi*(s_obs-u)/beta)^(-1/xi)]^m，
      小尾时 ≈ m × 单点超越概率；
    * xi_min=-0.10 负向正则化：负 xi 的人工支撑上界若落在观测范围内，
      会把 p_luck 虚假压到 0（反保守）；缓负值=更重尾巴，方向保守；
    * xi_full / beta_full / p_luck_full：全量拟合（含最优）的保守对照；
      两 p 差异大说明最优对拟合影响大（可能为真信号），决策以主 p 为准；
    * 若需要 per-trial 量（尾部 p / local fdr / BH-BY），用
      efron_null_fdr，且勿与本函数的 per-fold 判据叠加校正。
    """
    srs, n_total = _unique_trial_scores(score_or_study)
    m = len(srs)
    if m < 7:
        return _degenerate_deflate(srs, n_total, "有效 trial 过少（<7），无法估计上尾零分布")
    s_obs = float(srs.max())

    # 主拟合：剔除最优后的噪声律（trim-top-1，避免最优自身撑大上尾）
    rest = np.sort(srs)[:-1]
    fit = _fit_gpd_tail(rest, u_frac=u_frac, min_exceed=min_exceed,
                        xi_min=xi_min)
    if fit is None:
        return _degenerate_deflate(srs, n_total, "分数无散布或上尾并列不可拟合")
    u, n_exceed, xi, beta = fit
    q_tail = n_exceed / rest.size

    # 解析 p_luck：噪声律下 max-of-m 达到 s_obs 的概率
    sbar = _gpd_tail_survival(s_obs, u, xi, beta, q_tail)
    p_luck = 1.0 - (1.0 - float(np.clip(sbar, 0.0, 1.0))) ** m

    # max-of-m 的 MC 分布：M = F^-1(U^(1/m))，F = 经验主体 + GPD 上尾 混合
    rng = np.random.default_rng(seed)
    v_max = rng.random(n_sim) ** (1.0 / m)
    mc_max = np.empty(n_sim)
    lo = v_max < (1.0 - q_tail)              # 主体段：ECDF 分位反演
    body = rest[rest <= u]
    mc_max[lo] = np.quantile(body, v_max[lo] / (1.0 - q_tail))
    t = (1.0 - v_max[~lo]) / q_tail          # 尾部段：生存函数反演
    if abs(xi) < 1e-9:
        mc_max[~lo] = u - beta * np.log(t)
    else:
        mc_max[~lo] = u + beta / xi * (t ** (-xi) - 1.0)
    exp_max = float(mc_max.mean())
    max_p95 = float(np.percentile(mc_max, 95))
    p_luck_mc = float(np.mean(mc_max >= s_obs))

    # 保守对照：全量拟合（含最优）下的 p_luck（家族规模 m 不变）
    xi_full = beta_full = p_luck_full = np.nan
    fit_full = _fit_gpd_tail(srs, u_frac=u_frac, min_exceed=min_exceed,
                             xi_min=xi_min)
    if fit_full is not None:
        u_f, k_f, xi_f, beta_f = fit_full
        sbar_f = _gpd_tail_survival(s_obs, u_f, xi_f, beta_f, k_f / m)
        xi_full, beta_full = xi_f, beta_f
        p_luck_full = 1.0 - (1.0 - float(np.clip(sbar_f, 0.0, 1.0))) ** m

    return {"p_luck": p_luck, "best_score": s_obs, "exp_max": exp_max,
            "margin": s_obs - exp_max, "max_p95": max_p95, "n_trials": m,
            "n_trials_total": n_total, "scores": srs, "u": u,
            "n_exceed": n_exceed, "xi": xi, "beta": beta, "xi_full": xi_full,
            "beta_full": beta_full, "p_luck_full": p_luck_full,
            "p_luck_mc": p_luck_mc, "method": "gpd_pot", "degenerate": None}


def efron_null_fdr(score_or_study, z_cut=1.0, min_n_lfdr=20):
    """Efron 经验零分布（主体法）：per-trial 尾部 p 值 / local fdr / BH-BY。

    假设"多数 trial 为噪声"，用分布主体的稳健位置/尺度构造经验零分布：

    * center/scale = 中位数 / (1.4826 × MAD)（对少量右尾信号稳健）；
    * z = (s - center)/scale；上侧尾部 p = sf(z)（复合分数越大越好，单尾）；
    * pi0：|z| ≤ z_cut 的观测占比 ÷ 正态期望占比，截断到 [0, 1]；
    * local fdr = pi0·φ(z)/f_hat(z)，f_hat 为 z 的 KDE（m ≥ min_n_lfdr 时
      计算；m≈100 下为指示性量，不用于精确阈值判读）；
    * q_bh / q_by：对 p 值做 BH / BY 校正（statsmodels.multipletests）——
      分数本身不是 p 值，BH/BY 必须经本函数的 p 值列接入。

    与 empirical_deflate 分层使用：本函数出 per-trial 量（集合筛选），
    每折的判决量用 empirical_deflate（max 版），勿在同层叠加。

    Parameters
    ----------
    score_or_study : optuna Study | array-like（解析与 calc_dsr 同口径）
    z_cut : float，pi0 估计的中心区半宽（z 单位）
    min_n_lfdr : int，计算 local fdr 的最小 trial 数

    Returns
    -------
    dict : center / scale / pi0 / n_trials / n_trials_total / table
        table : pandas.DataFrame（按分数降序）score / z / p_value /
        local_fdr / q_bh / q_by
    """
    srs, n_total = _unique_trial_scores(score_or_study)
    m = len(srs)
    if m < 6:
        raise ValueError("efron_null_fdr: 有效 trial 过少（<6）")
    center = float(np.median(srs))
    scale = float(1.4826 * np.median(np.abs(srs - center)))
    if scale <= 0:                           # MAD 退化（大量并列）→ 退回 std
        scale = float(np.std(srs, ddof=1))
    if scale <= 0:                           # 全等分数：无散布 → 全部视为噪声
        z = np.zeros(m)
        p = np.ones(m)
        lfdr = np.ones(m)
        pi0 = 1.0
    else:
        z = (srs - center) / scale
        p = norm.sf(z)                       # 单尾上侧：复合分数越大越好
        exp_central = 2.0 * norm.cdf(z_cut) - 1.0
        pi0 = float(np.clip(np.mean(np.abs(z) <= z_cut) / exp_central, 0.0, 1.0))
        lfdr = np.full(m, np.nan)
        if m >= min_n_lfdr and np.std(z) > 0:
            try:
                dens = gaussian_kde(z)(z)
                with np.errstate(divide="ignore", invalid="ignore"):
                    lfdr = np.minimum(1.0, pi0 * norm.pdf(z) / dens)
                lfdr[~np.isfinite(lfdr)] = np.nan
            except Exception:
                lfdr = np.full(m, np.nan)
    from statsmodels.stats.multitest import multipletests
    _, q_bh, _, _ = multipletests(p, method="fdr_bh")
    _, q_by, _, _ = multipletests(p, method="fdr_by")
    table = (pd.DataFrame({"score": srs, "z": z, "p_value": p,
                           "local_fdr": lfdr, "q_bh": q_bh, "q_by": q_by})
             .sort_values("score", ascending=False).reset_index(drop=True))
    return {"center": center, "scale": scale, "pi0": pi0, "n_trials": m,
            "n_trials_total": n_total, "table": table}


# ---------------------------------------------------------------------------
# 选择排名 PBO 诊断：全参数 train/test 重放 + 分位 logit（逐折 CSCV 变体）
# ---------------------------------------------------------------------------
# 与 empirical_deflate（分数绝对值 + 零分布 max-of-N 校正）互补：本段看
# "分布内相对位置"——被选参数在参数池中的排名分位与 logit（跨折频率）。
def _unique_trials(study):
    """study 有效 trial 按参数去重（trial_param_key 口径）→ [(key, params), ...]。

    与 _unique_trial_scores 同去重口径（重复采样不增加候选信息），区别是
    保留参数本体供重放：同一参数组合只留先出现 trial 的 params（重复命中
    的参数值相同，浮点采样残差经 key 的 round(9) 归一）。
    """
    trials = [t for t in study.trials
              if t.values is not None and np.isfinite(t.values[0])]
    seen, out = set(), []
    for t in trials:
        key = trial_param_key(t.params)
        if key in seen:
            continue
        seen.add(key)
        out.append((key, dict(t.params)))
    return out


def rank_pbo_logit(X, folds, wf_kwargs=None, purged_size=1, reduce_test=True,
                   score="composite", include_train_argmax=True,
                   keep_params=True, verbose=True):
    """选择排名 PBO 诊断：全参数 train/test 重放 → 分位 logit λ（逐折 CSCV 变体）。

    消费 nested_adaptive_search 的 fold_results（或同构 fold 结果）：把每折
    study 中全部去重 trial 参数经 run_params_on_fold（fit 最近 train_size 天
    → predict 全 train 段 + 纯净 test 段；与扰动/敏感性/Top-K 同一公共应用
    入口，部署语义）重放，得到每折两条分数向量（train 侧 / test 侧），计算
    CPCV-best（路径均值最优，即生产参数）在参数池中的相对排名分位与 logit：

        w = rank / (N + 1)              rank 升序（最低分 1，CSCV 原文口径）
        lambda = ln( w / (1 - w) )      lambda>0 ⇔ 好于中位；<0 ⇔ 差于中位
        PBO = freq( lambda_test < 0 )   跨折频率：OOS 排名低于中位的折占比

    研究对象1（主）= CPCV-best：train 侧排名（w_train）与 test 侧排名
    （w_test）分别计算——"两次计算"共享同一次 fit（run_params_on_fold 一次
    返回 train/test 两段）。
    研究对象2（对照，include_train_argmax=True）= 纯 train 分数 argmax，
    即标准 CSCV 定义"IS 最优策略"的直接落地（同一批分数表，零额外成本）。

    机制备注（解读定位）：内层 CPCV 窗口 = 外层 train 段末段（ts+test_size
    天均在外层训练段内），选参已间接使用 train 的全部信息 → w_train 预期
    偏高，该量用于量化 CPCV 路径均值最优与单段 IS 实现的乖离（best 与
    train-argmax 是否同参、分位落差多大），不是独立的样本内检验；独立
    的样本外信息在 w_test（外层 test 段从未进入任何搜索）。

    纪律：只评估、不选参。若诊断显示 best 并非 test 排名最优，不得据此
    换参——用 OOS 排名回改策略会作废该段 OOS。

    折位对齐：wf_kwargs 缺省时从 folds 推导众数窗口（derive_outer_window），
    purged_size / reduce_test 显式透传（默认 1 / True 与搜索侧一致，务必与
    搜索配置核对）。校验：best 重放 ASR 必须复现 folds 的 "train ASR"/
    "test ASR"（max|Δ|>1e-6 视为口径错位直接报错，防静默错位）。

    Parameters
    ----------
    X : pd.DataFrame，与 nested_adaptive_search 同口径收益数据
    folds : list，nested_adaptive_search 返回结果（每折含 study / train ASR /
        test ASR / params）
    wf_kwargs : dict | None，run_params_on_fold 折位参数 {test_size,
        train_size, purged_size, reduce_test}；None = 从 folds 推导
    purged_size : int，外层 WF purge（仅 wf_kwargs 缺省推导时使用，默认 1）
    reduce_test : bool，外层 WF 尾段策略（默认 True 与搜索侧一致，含缩短尾折）
    score : "composite" | "asr"，主口径排名分数。composite = asr*0.5 - maxdd
        + skew（与 inner_cpcv_score 同构的单段版，选参效用函数一致）；两种
        口径的排名均在 scores 表输出，主口径决定 best 的 λ / PBO 报告
    include_train_argmax : bool，是否输出 train-argmax 对照（标准 CSCV 定义）
    keep_params : bool，scores 表是否展平参数列（审计；fitness 展平为 fitness）
    verbose : bool，打印逐折进度与总结报告

    Returns
    -------
    dict :
        scores : DataFrame，行 MultiIndex (fold, param_id)；列 = 双口径分数
            （composite/asr/mdd/skew × train/test）| w_train / w_test（主口径
            分位）| w_train_asr / w_test_asr（ASR 口径分位）| is_cpcv_best /
            is_train_argmax（+ keep_params 时参数列）
        best : DataFrame，行 = fold：n_params | rank/w/lambda（train/test 主
            口径）| same_as_train_argmax | train_gap_to_argmax（CPCV-best 的
            train 分距池内冠军的分数差）
        train_argmax : DataFrame | None，对照对象行 = fold：rank/w/lambda_test
            | same_as_cpcv_best（include_train_argmax=False 时为 None）
        summary : dict，汇总判据：折数 / 参数池规模 / best 的 w_train 与
            w_test 分位（median/p5/p95）/ lambda>0 折占比 / pbo_train 与
            pbo_test（freq(λ<0)）/ train-argmax 对照的 pbo_test 与同参折
            占比 / asr_max_diff（重放校验量）

    用法::

        from cpcv_analysis import rank_pbo_logit
        out = rank_pbo_logit(X, folds)              # 折位参数缺省从 folds 推导
        out["summary"]["best"]["pbo_test"]          # 主判据：OOS 排名 logit<0 频率
        out["best"]                                 # 一行一折（train/test 两次计算）
        out["scores"]                               # 全参数长表（可深挖 rank 传递性）
    """
    from scipy.stats import rankdata
    from wf_cpcv_search import derive_outer_window, run_params_on_fold

    if not folds:
        raise ValueError("rank_pbo_logit: folds 为空")
    if score not in ("composite", "asr"):
        raise ValueError(
            f"rank_pbo_logit: score 需为 'composite' | 'asr'，收到 {score!r}")

    # 折位对齐：缺省从 folds 推导（与 derive_wf_kwargs 同映射，不引入 robustness 依赖）
    if wf_kwargs is None:
        w = derive_outer_window(folds, purged_size=purged_size,
                                reduce_test=reduce_test)
        wf_kwargs = {"test_size": w["test_size"], "train_size": w["train_size"],
                     "purged_size": w["outer_purged_size"],
                     "reduce_test": w["outer_reduce_test"]}

    def _side_scores(ptf, side):
        """单段 Portfolio → 该侧分数四项（复合分 = inner_cpcv_score 公式的
        单段版：asr*0.5 - maxdd + skew；双口径共用一次取属性）。"""
        asr = float(ptf.annualized_sharpe_ratio)
        mdd = float(ptf.max_drawdown)
        skw = float(ptf.skew)
        return {f"asr_{side}": asr, f"mdd_{side}": mdd, f"skew_{side}": skw,
                f"composite_{side}": asr * 0.5 - mdd + skw}

    def _logit(w_):
        """w ∈ (0,1) 严格（rank/(N+1)）→ logit 无需 eps 截断。"""
        return float(np.log(w_ / (1.0 - w_)))

    rows, best_rows, am_rows = [], [], []
    pool_sizes, asr_dmax = [], 0.0
    for f in folds:
        fold_idx = f["fold"]
        pool = _unique_trials(f["study"])
        n = len(pool)
        if n < 2:
            raise ValueError(f"rank_pbo_logit: Fold {fold_idx} 有效参数池过小"
                             f"（N={n}<2），无法排名")
        keys = [k for k, _ in pool]
        best_key = trial_param_key(f["params"])
        if best_key not in keys:
            raise ValueError(f"rank_pbo_logit: Fold {fold_idx} 的 best 参数不在"
                             f" study 去重池中（口径异常）")
        pool_sizes.append(n)

        # ① 全参数重放：每参数一次 fit → train/test 两段（run_params_on_fold）
        recs = []
        for key, p in pool:
            tr_ptf, te_ptf = run_params_on_fold(X, fold_idx, p, **wf_kwargs)
            recs.append((key, p, {**_side_scores(tr_ptf, "train"),
                                  **_side_scores(te_ptf, "test")}))

        # ② 每侧排名分位 w = rank/(N+1)（主口径 + ASR 附口径；rankdata 处理并列）
        def _w_vec(side, kind):
            v = np.array([r[2][f"{kind}_{side}"] for r in recs], dtype=float)
            r = rankdata(v, method="average")
            return v, r, r / (n + 1.0)

        s_tr, rank_tr, w_tr = _w_vec("train", score)
        s_te, rank_te, w_te = _w_vec("test", score)
        _, _, w_tr_asr = _w_vec("train", "asr")
        _, _, w_te_asr = _w_vec("test", "asr")

        idx_best = keys.index(best_key)
        idx_am = int(np.argmax(s_tr)) if include_train_argmax else None

        # ③ best 行：train 侧 + test 侧两次计算（同一批 fit 的两侧输出）
        #    校验：重放 ASR 必须复现 folds 记录（同入口同口径）
        asr_dmax = max(asr_dmax,
                       abs(recs[idx_best][2]["asr_train"] - float(f["train ASR"])),
                       abs(recs[idx_best][2]["asr_test"] - float(f["test ASR"])))
        best_rows.append({
            "fold": fold_idx, "n_params": n,
            "rank_train": float(rank_tr[idx_best]),
            "w_train": float(w_tr[idx_best]),
            "lambda_train": _logit(w_tr[idx_best]),
            "rank_test": float(rank_te[idx_best]),
            "w_test": float(w_te[idx_best]),
            "lambda_test": _logit(w_te[idx_best]),
            "same_as_train_argmax": (bool(idx_am == idx_best)
                                     if include_train_argmax else None),
            "train_gap_to_argmax": (float(s_tr[idx_am] - s_tr[idx_best])
                                    if include_train_argmax else np.nan),
        })

        # ④ train-argmax 对照行：train 侧恒为池内冠军（rank=N），有信息的是 test 侧
        if include_train_argmax:
            am_rows.append({"fold": fold_idx, "n_params": n,
                            "rank_test": float(rank_te[idx_am]),
                            "w_test": float(w_te[idx_am]),
                            "lambda_test": _logit(w_te[idx_am]),
                            "same_as_cpcv_best": bool(idx_am == idx_best)})

        # ⑤ 全参数明细行（审计 + 供自行深挖 rank 传递性）
        for i, (key, p, s) in enumerate(recs):
            row = {"fold": fold_idx, "param_id": i, **s,
                   "w_train": float(w_tr[i]), "w_test": float(w_te[i]),
                   "w_train_asr": float(w_tr_asr[i]),
                   "w_test_asr": float(w_te_asr[i]),
                   "is_cpcv_best": bool(i == idx_best),
                   "is_train_argmax": (bool(i == idx_am)
                                       if include_train_argmax else None)}
            if keep_params:
                pp = dict(p)
                pp["fitness"] = str(pp.pop("nondomin__fitness_measures"))
                row.update(pp)
            rows.append(row)

        if verbose:
            print(f"Fold {fold_idx}: N={n} | w_train={w_tr[idx_best]:.2f} "
                  f"| w_test={w_te[idx_best]:.2f} "
                  f"| lambda_test={_logit(w_te[idx_best]):+.2f}")

    if asr_dmax > 1e-6:
        raise ValueError(
            f"rank_pbo_logit: best 重放 ASR 与 folds 记录不一致"
            f"（max|Δ|={asr_dmax:.3g}），请核对 X / wf_kwargs 与 "
            f"nested_adaptive_search 是否同口径")

    scores = pd.DataFrame(rows).set_index(["fold", "param_id"])
    best = pd.DataFrame(best_rows).set_index("fold")
    train_argmax = (pd.DataFrame(am_rows).set_index("fold")
                    if include_train_argmax else None)

    # ⑥ 汇总：best 的两次计算 + 主判据 PBO（跨折频率）
    lam_te = best["lambda_test"].to_numpy()
    lam_tr = best["lambda_train"].to_numpy()
    w_tr_b = best["w_train"].to_numpy()
    w_te_b = best["w_test"].to_numpy()

    def _q(v):
        return {"median": float(np.median(v)), "p5": float(np.percentile(v, 5)),
                "p95": float(np.percentile(v, 95))}

    summary = {
        "n_folds": len(best),
        "n_params": {"min": int(min(pool_sizes)), "max": int(max(pool_sizes)),
                     "total": int(sum(pool_sizes))},
        "best": {
            "w_train": _q(w_tr_b),
            "w_train_gt_half_frac": float((w_tr_b > 0.5).mean()),
            "lambda_train_gt0_frac": float((lam_tr > 0).mean()),
            "w_test": _q(w_te_b),
            "lambda_test": _q(lam_te),
            "pbo_train": float((lam_tr < 0).mean()),
            "pbo_test": float((lam_te < 0).mean()),
        },
        "train_argmax": None,
        "asr_max_diff": asr_dmax,
    }
    if include_train_argmax:
        lam_te_am = train_argmax["lambda_test"].to_numpy()
        summary["train_argmax"] = {
            "same_frac": float(best["same_as_train_argmax"].mean()),
            "lambda_test": _q(lam_te_am),
            "pbo_test": float((lam_te_am < 0).mean()),
        }

    if verbose:
        B = summary["best"]
        print("\n" + "=" * 60)
        print("        CPCV 选择排名 PBO 诊断（全参数 train/test 重放）")
        print("=" * 60)
        print(f"  折数={summary['n_folds']} | 每折参数池 "
              f"{summary['n_params']['min']}~{summary['n_params']['max']}"
              f"（unique 去重）")
        print("  [研究对象1] CPCV-best（路径均值最优；与 train 高度重合，"
              "w_train 预期偏高）")
        print(f"    w_train  : 中位 {B['w_train']['median']:.2f} | 5%~95% "
              f"[{B['w_train']['p5']:.2f}, {B['w_train']['p95']:.2f}] | "
              f">0.5 折占比 {B['w_train_gt_half_frac']:.0%}")
        print(f"    lambda_train > 0 折占比 {B['lambda_train_gt0_frac']:.0%}")
        print(f"    w_test   : 中位 {B['w_test']['median']:.2f} | 5%~95% "
              f"[{B['w_test']['p5']:.2f}, {B['w_test']['p95']:.2f}]")
        print(f"    PBO_test = freq(lambda_test < 0) = {B['pbo_test']:.1%} "
              f"（{int((lam_te < 0).sum())}/{summary['n_folds']}）")
        if include_train_argmax:
            A = summary["train_argmax"]
            print("  [研究对象2] train-argmax 对照（标准 CSCV 定义：IS 最优）")
            print(f"    与 CPCV-best 同参折占比 {A['same_frac']:.0%} | "
                  f"lambda_test 中位 {A['lambda_test']['median']:+.2f}")
            print(f"    PBO_test(标准) = {A['pbo_test']:.1%} "
                  f"（{int((lam_te_am < 0).sum())}/{summary['n_folds']}）")
        print("-" * 60)
        print(f"  重放校验 max|ΔASR| = {asr_dmax:.2e}（应≈0；不一致会直接报错）")
        print("=" * 60)

    return {"scores": scores, "best": best, "train_argmax": train_argmax,
            "summary": summary}
