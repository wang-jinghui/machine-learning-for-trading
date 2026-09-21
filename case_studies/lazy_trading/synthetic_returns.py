# -*- coding: utf-8 -*-
"""合成收益率生成器：逐标的边缘分布 + Gaussian copula 截面依赖 + 掩码统一 + 硬截断。

定位
----
输入 X（skfolio 风格收益率矩阵：行=日期 DatetimeIndex，列=资产，允许前导/内部
NaN），输出同索引、同列集、同缺失模式的合成数据，用于对策略做"随机数据零假设
检验"——若策略在随机数据上依然盈利，即为过拟合嫌疑。与 WalkForward+SyntheicData.ipynb
（VineCopula 逐折生成）互补：本模块走"解析边缘分布 + 高斯 copula"的轻量路线。

生成流程
--------
1. 逐列统计（仅有效值）：mean / std(ddof=1) / skew / 超额峰度 / min / max；
2. 逐列拟合边缘分布参数：
   - gaussian  : loc=mean, scale=std；
   - student_t : 标准化 z=(x-loc)/scale 后 MLE 估自由度 nu（nu_range 内有界优化，
     样本过少或失败时回落矩估计 4+6/超额峰度），scale 做方差匹配校正
     σ·sqrt((nu-2)/nu)，保证生成方差=样本方差；
3. 截面依赖：z 分数矩阵的配对完整 Pearson 相关（复用 X 的相关结构），
   特征值截断修复 PSD 后 Cholesky 采样 Z~N(0,Σ)，meta-Gaussian copula 将
   每列经 Φ→分位数反变换映射回自身边缘分布；
4. 掩码统一：X 缺失（NaN/±Inf）处输出一律 NaN，逐格一致；
5. 极值检查 + 硬截断：每列截断到 X 有效值的 [min, max]，统计截断比例，
   超阈值记 log 警告；最后断言掩码一致且无越界。

Notes
-----
- meta-Gaussian copula 保的是秩相关：t 边缘下输出列的 Pearson 相关会因分位数
  变换向 0 收缩（高斯边缘时无此现象，等价多元正态采样），秩结构忠实于 X；
- 硬截断削掉最厚尾部，实现峰度略低于拟合目标（return_report=True 的对比表可量化）；
- 同列序列为 iid 抽样，不保留时间自相关/波动聚集（零假设所需）；
- 全宇宙规模（约 2100 列）单次生成约数十秒：逐列 MLE + 相关矩阵特征值修复是大头。

用法
----
>>> stats = estimate_marginal_stats(X, distribution="student_t")    # 先看统计属性
>>> X_fake = synthesize_returns(X, random_state=42)                 # 默认 student_t
>>> X_fake, report = synthesize_returns(X, random_state=42, return_report=True)
>>> X_ctrl = synthesize_returns(X, distribution="gaussian")         # 高斯对照

命令行自检（在本目录 case_studies/lazy_trading 下运行）：
    python synthetic_returns.py
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln

logger = logging.getLogger(__name__)

__all__ = ["estimate_marginal_stats", "synthesize_returns"]

# ---------------------------------------------------------------------------
# 分布注册表：名称 -> 采样分支（后续扩展新分布只需在此登记 + 增加采样分支）
# ---------------------------------------------------------------------------
SUPPORTED_DISTRIBUTIONS = ("gaussian", "student_t")
_DISTRIBUTION_ALIASES = {
    "normal": "gaussian",
    "gauss": "gaussian",
    "t": "student_t",
    "student-t": "student_t",
    "studentt": "student_t",
}

_MIN_MLE_N = 30    # t 自由度 MLE 的最少有效样本数，低于此值改用矩估计
_PSD_EPS = 1e-8    # 相关矩阵 PSD 修复的特征值下限
_PPF_CLIP = 1e-12  # 分位数反变换的 u 截断，防止 ±inf


# ---------------------------------------------------------------------------
# 输入校验与预处理
# ---------------------------------------------------------------------------
def _validate_X(X: pd.DataFrame) -> None:
    """校验输入为数值型非空 DataFrame。"""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X 需为 pandas DataFrame，收到 {type(X).__name__}")
    if X.empty:
        raise ValueError("X 为空（无行或无列）")
    bad = X.columns[~X.dtypes.apply(pd.api.types.is_numeric_dtype)]
    if len(bad):
        raise TypeError(f"X 存在非数值列: {list(bad)[:5]}")


def _resolve_distribution(name: str) -> str:
    """分布名归一化 + 校验。"""
    key = _DISTRIBUTION_ALIASES.get(str(name).strip().lower(), str(name).strip().lower())
    if key not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(f"不支持的分布 {name!r}，可选: {SUPPORTED_DISTRIBUTIONS}")
    return key


def _validate_nu_range(nu_range) -> tuple[float, float]:
    """自由度范围校验：需满足 2 < 下界 < 上界。"""
    lo, hi = float(nu_range[0]), float(nu_range[1])
    if not 2.0 < lo < hi:
        raise ValueError(f"nu_range 需满足 2 < 下界 < 上界，收到 {nu_range}")
    return lo, hi


def _validate_fit_method(fit_method: str) -> None:
    if fit_method not in ("mle", "moments"):
        raise ValueError(f"fit_method 仅支持 'mle' / 'moments'，收到 {fit_method!r}")


def _clean_nonfinite(X: pd.DataFrame):
    """±Inf 按缺失处理：返回 (清理副本, 有限性掩码, 各列 Inf 计数)。

    有限性掩码为 True 表示该单元格在 X 中有限（参与统计/生成），NaN/±Inf 均为 False。
    """
    X_np = X.to_numpy(dtype=float)
    finite = np.isfinite(X_np)
    X_clean = X.mask(~finite)
    n_inf = pd.Series((~finite).sum(axis=0) - X.isna().to_numpy().sum(axis=0),
                      index=X.columns, dtype=np.int64)
    return X_clean, finite, n_inf


# ---------------------------------------------------------------------------
# 逐列统计与分布拟合
# ---------------------------------------------------------------------------
def _column_stats(X_clean: pd.DataFrame) -> pd.DataFrame:
    """逐列基础统计（仅有效值）：n_valid / mean / std(ddof=1) / skew / 超额峰度 / min / max。"""
    return pd.DataFrame({
        "n_valid": X_clean.count(),
        "mean": X_clean.mean(),
        "std": X_clean.std(ddof=1),
        "skew": X_clean.skew(),
        "kurt": X_clean.kurt(),  # 超额峰度（无偏估计，与 scipy bias=False 一致）
        "min": X_clean.min(),
        "max": X_clean.max(),
    })


def _nu_from_kurtosis(excess_kurt: float, nu_range: tuple[float, float]) -> float:
    """矩估计：nu ≈ 4 + 6/g2（g2 为超额峰度）；g2≤0（轻尾）近似高斯取上界。"""
    if not np.isfinite(excess_kurt) or excess_kurt <= 1e-10:
        return nu_range[1]
    return 4.0 + 6.0 / excess_kurt


def _nu_mle(z: np.ndarray, nu_range: tuple[float, float]) -> float:
    """标准化样本上 t 分布自由度的 MLE（log(ν−2) 参数化 + 有界优化）。

    返回 NaN 表示优化失败，由调用方回落矩估计。
    """
    z2 = z * z

    def nll(t: float) -> float:
        nu = 2.0 + np.exp(t)
        return -np.sum(
            gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi)
            - (nu + 1.0) / 2.0 * np.log1p(z2 / nu)
        )

    lo, hi = np.log(nu_range[0] - 2.0), np.log(nu_range[1] - 2.0)
    try:
        res = optimize.minimize_scalar(nll, bounds=(lo, hi), method="bounded")
    except Exception:  # 优化器内部异常（极端数据）→ 回落矩估计
        return np.nan
    if not res.success or not np.isfinite(res.fun):
        return np.nan
    return float(2.0 + np.exp(res.x))


def _fit_marginals(X_clean: pd.DataFrame, stats_df: pd.DataFrame, distribution: str,
                   nu_range: tuple[float, float], fit_method: str) -> pd.DataFrame:
    """逐列拟合边缘分布参数表（loc / scale / nu）。

    gaussian：loc=mean, scale=std，nu 全 NaN；
    student_t：nu 由 MLE（失败/样本过少回落矩估计）得到，并在 nu_range 内截断；
    退化列（有效样本 <2 或 std 非正）nu 留 NaN，采样时按常数处理。

    Notes：scale 保持为 σ̂（与统计表 std 同口径）；t 采样时另做方差匹配校正
    σ̂·sqrt((ν−2)/ν)，不在此处改写，便于与 X 的统计属性直接对照。
    """
    fit = pd.DataFrame({
        "loc": stats_df["mean"],
        "scale": stats_df["std"],
    }, index=X_clean.columns)

    if distribution == "gaussian":
        fit["nu"] = np.nan
        return fit

    nu = pd.Series(np.nan, index=X_clean.columns, dtype=float)
    for col in X_clean.columns:
        x = X_clean[col].to_numpy(dtype=float)
        x = x[~np.isnan(x)]
        s = fit.at[col, "scale"]
        if x.size < 2 or not np.isfinite(s) or s <= 0.0:
            continue  # 退化列：常数/全缺失，无随机性可言
        z = (x - fit.at[col, "loc"]) / s
        nu_hat = np.nan
        if fit_method == "mle" and x.size >= _MIN_MLE_N:
            nu_hat = _nu_mle(z, nu_range)
        if not np.isfinite(nu_hat):
            g2 = float(stats.kurtosis(x, fisher=True, bias=False)) if x.size >= 4 else np.nan
            nu_hat = _nu_from_kurtosis(g2, nu_range)
        nu[col] = float(np.clip(nu_hat, nu_range[0], nu_range[1]))
    fit["nu"] = nu
    return fit


# ---------------------------------------------------------------------------
# 截面相关结构与 copula 采样
# ---------------------------------------------------------------------------
def _pairwise_corr(z: np.ndarray) -> np.ndarray:
    """配对完整（pairwise-complete）Pearson 相关矩阵（掩码矩阵乘法，仅依赖 numpy）。

    对含 NaN 的 z 矩阵逐对计算：n_ij 为两列同时有效的样本数，只用有效单元格
    参与各阶矩；样本不足/零方差的对相关系数置 0（视为不相关）。
    """
    m = np.isfinite(z).astype(np.float64)
    z0 = np.where(m > 0, z, 0.0)
    n = m.T @ m              # n_ij：配对有效样本数
    sx = z0.T @ m            # Σ z_i（j 有效处）
    sxx = (z0 * z0).T @ m    # Σ z_i²（j 有效处）
    syy = sxx.T
    sxy = z0.T @ z0          # 无效位置 z0=0 不贡献
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = sx / n, sx.T / n
        cov = sxy / n - mx * my
        vx = sxx / n - mx * mx
        vy = syy / n - my * my
        corr = cov / np.sqrt(vx * vy)
    corr[~np.isfinite(corr)] = 0.0   # 样本不足/零方差 -> 不相关
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _cholesky_with_repair(corr: np.ndarray, max_iter: int = 6) -> np.ndarray:
    """Cholesky 分解；失败则特征值截断修复 PSD（对角重归一化）后重试。"""
    c = corr
    for it in range(max_iter):
        try:
            return np.linalg.cholesky(c)
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(c)
            w = np.clip(w, _PSD_EPS * (10.0 ** it), None)
            c = (v * w) @ v.T
            d = np.sqrt(np.diag(c))
            c = c / np.outer(d, d)
            np.fill_diagonal(c, 1.0)
    raise np.linalg.LinAlgError("相关矩阵 PSD 修复失败（特征值截断多轮后仍不正定）")


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------
def estimate_marginal_stats(X: pd.DataFrame, distribution: str = "student_t",
                            nu_range: tuple[float, float] = (3.0, 100.0),
                            fit_method: str = "mle") -> pd.DataFrame:
    """逐标的统计属性表：基础统计 + 拟合的边缘分布参数。

    Parameters
    ----------
    X : pd.DataFrame
        skfolio 风格收益率矩阵（行=日期，列=资产，允许 NaN/±Inf）。
    distribution : {"student_t", "gaussian"}
        边缘分布族；默认 student_t，gaussian 用作对照。
    nu_range : (float, float)
        student_t 自由度估计范围（截断，默认 (3, 100)）。
    fit_method : {"mle", "moments"}
        student_t 自由度估计方法：mle（默认，样本过少自动回落）/ moments（快）。

    Returns
    -------
    pd.DataFrame : 每列（资产）一行，列含 n_valid / n_missing / n_inf / mean / std /
        skew / kurt（超额峰度）/ min / max / loc / scale / nu。
    """
    _validate_X(X)
    dist = _resolve_distribution(distribution)
    nu_lo, nu_hi = _validate_nu_range(nu_range)
    _validate_fit_method(fit_method)

    X_clean, _, n_inf = _clean_nonfinite(X)
    out = _column_stats(X_clean)
    out.insert(0, "n_missing", X.isna().sum())
    out.insert(1, "n_inf", n_inf)
    fit = _fit_marginals(X_clean, out, dist, (nu_lo, nu_hi), fit_method)
    return pd.concat([out, fit], axis=1)


def synthesize_returns(X: pd.DataFrame, distribution: str = "student_t",
                       random_state=None, nu_range: tuple[float, float] = (3.0, 100.0),
                       fit_method: str = "mle", clip: bool = True,
                       clip_warn_ratio: float = 0.01,
                       return_report: bool = False):
    """在 X 的统计属性/缺失模式/值域约束下生成合成收益率矩阵。

    流程：逐列拟合边缘分布（student_t 默认 / gaussian 对照）→ Gaussian copula
    承接截面相关 → iid 抽样 → 掩码与 X 逐格一致 → 每列硬截断到 X 有效值
    [min, max] → 断言校验。

    Parameters
    ----------
    X : pd.DataFrame
        skfolio 风格收益率矩阵（行=日期，列=资产，允许 NaN/±Inf）。
    distribution : {"student_t", "gaussian"}
        边缘分布族，默认 student_t；gaussian 为对照项（等价多元正态采样）。
    random_state : int | np.random.Generator | None
        随机种子/生成器，保证可复现。
    nu_range : (float, float)
        student_t 自由度估计范围（截断，默认 (3, 100)）。
    fit_method : {"mle", "moments"}
        student_t 自由度估计方法；mle 逐列优化较慢（全宇宙约数十秒），moments 快。
    clip : bool
        是否对每列做硬截断（默认 True，截断到 X 有效值 [min, max]）。
    clip_warn_ratio : float
        截断比例告警阈值（默认 0.01）。实际阈值取 max(clip_warn_ratio,
        3×2/(n_valid+1))——新样本落在实测极差外的比例基准≈2/(n+1) 且与分布无关，
        由此避免短历史列（样本范围天然窄）误报；超过记 log 警告。
    return_report : bool
        为 True 时返回 (X_synth, report)；report 每列一行：拟合参数、截断统计、
        目标（X）与实现（合成）的 mean/std/skew/kurt 对比。

    Returns
    -------
    pd.DataFrame 或 (pd.DataFrame, pd.DataFrame)
        与 X 同索引、同列集、同缺失模式的合成数据（float64）。
    """
    _validate_X(X)
    dist = _resolve_distribution(distribution)
    nu_lo, nu_hi = _validate_nu_range(nu_range)
    _validate_fit_method(fit_method)
    rng = np.random.default_rng(random_state)

    n_rows, n_cols = X.shape
    cols = X.columns

    # ① 逐列统计与拟合（±Inf 按缺失处理）
    X_clean, finite, n_inf = _clean_nonfinite(X)
    stats_df = _column_stats(X_clean)
    fit = _fit_marginals(X_clean, stats_df, dist, (nu_lo, nu_hi), fit_method)
    loc = fit["loc"].to_numpy(dtype=float)
    scale = fit["scale"].to_numpy(dtype=float)
    nu = fit["nu"].to_numpy(dtype=float)
    n_valid = stats_df["n_valid"].to_numpy(dtype=float)

    # 活动列（可随机抽样）：>=2 个有效值且 std 为正；其余退化列按常数填充
    is_active = (n_valid >= 2) & np.isfinite(scale) & (scale > 0.0)
    active = np.flatnonzero(is_active)
    const_cols = np.flatnonzero(~is_active & (n_valid >= 1))

    # ② 生成：Gaussian copula（截面相关）+ 逐列边缘分位数反变换
    out = np.full((n_rows, n_cols), np.nan, dtype=float)
    if active.size:
        X_np = X_clean.to_numpy(dtype=float)[:, active]
        z = (X_np - loc[active]) / scale[active]
        corr = _pairwise_corr(z)
        chol = _cholesky_with_repair(corr)
        z_scores = rng.standard_normal((n_rows, active.size)) @ chol.T  # 行 ~ N(0, Σ)
        for k, j in enumerate(active):
            z_col = z_scores[:, k]
            if dist == "gaussian":
                out[:, j] = loc[j] + scale[j] * z_col
            else:
                nu_j = nu[j]
                scale_t = scale[j] * np.sqrt((nu_j - 2.0) / nu_j)  # 方差匹配校正
                u = np.clip(stats.norm.cdf(z_col), _PPF_CLIP, 1.0 - _PPF_CLIP)
                out[:, j] = loc[j] + scale_t * stats.t.ppf(u, nu_j)
    for j in const_cols:
        out[:, j] = loc[j]

    # ③ 掩码统一：X 缺失（NaN/±Inf）处一律 NaN
    mask = ~finite
    out[mask] = np.nan

    # ④ 极值检查 + 硬截断：每列截断到 X 有效值 [min, max]
    lo = X_clean.min().to_numpy(dtype=float)
    hi = X_clean.max().to_numpy(dtype=float)
    clip_low = np.zeros(n_cols, dtype=np.int64)
    clip_high = np.zeros(n_cols, dtype=np.int64)
    visible = ~mask
    if clip:
        clip_low = ((out < lo) & visible).sum(axis=0)
        clip_high = ((out > hi) & visible).sum(axis=0)
        out = np.clip(out, lo, hi)

    # ⑤ 校验：掩码逐格一致、可见单元格有限、无越界
    if not np.array_equal(np.isnan(out), mask):
        raise AssertionError("掩码一致性校验失败：合成结果与 X 的缺失模式不一致")
    if visible.any() and not np.isfinite(out[visible]).all():
        raise AssertionError("合成结果在可见单元格出现非有限值")
    if clip and (np.any((out < lo) & visible) or np.any((out > hi) & visible)):
        raise AssertionError("硬截断越界校验失败")

    # ⑥ 告警：±Inf 统一按缺失处理；截断比例显著偏离样本极差基准时提示
    n_inf_total = int(n_inf.sum())
    if n_inf_total:
        logger.warning("X 含 %d 个 ±Inf 单元格，已按缺失统一处理", n_inf_total)
    if clip and visible.any():
        clip_ratio = (clip_low + clip_high) / np.maximum(visible.sum(axis=0), 1)
        # 基准：新样本落在实测 [min, max] 之外的概率≈2/(n_valid+1)（秩论证，与分布无关）；
        # 3 倍基准内视为正常，超过才提示（大样本列则回落到 clip_warn_ratio 阈值）
        threshold = np.maximum(clip_warn_ratio, 3.0 * 2.0 / (n_valid + 1.0))
        over = clip_ratio > threshold
        if over.any():
            j = int(np.argmax(clip_ratio))
            logger.warning(
                "硬截断告警：%d 列截断比例超过阈值（最高 %s: 截断 %.2f%% / 阈值 %.2f%% / 样本 %d）",
                int(over.sum()), cols[j], clip_ratio[j] * 100.0,
                threshold[j] * 100.0, int(n_valid[j]))

    X_synth = pd.DataFrame(out, index=X.index, columns=cols)
    if return_report:
        report = _build_report(X, stats_df, fit, X_synth, mask, clip_low, clip_high, n_inf)
        return X_synth, report
    return X_synth


def _build_report(X: pd.DataFrame, stats_df: pd.DataFrame, fit: pd.DataFrame,
                  X_synth: pd.DataFrame, mask: np.ndarray,
                  clip_low: np.ndarray, clip_high: np.ndarray,
                  n_inf: pd.Series) -> pd.DataFrame:
    """逐列报告：拟合参数 + 截断统计 + 目标/实现统计对比。"""
    n_visible = np.maximum((~mask).sum(axis=0), 1)
    report = pd.DataFrame(index=X.columns)
    report["n_valid"] = stats_df["n_valid"]
    report["n_missing"] = X.isna().sum()
    report["n_inf"] = n_inf
    report["loc"] = fit["loc"]
    report["scale"] = fit["scale"]
    report["nu"] = fit["nu"]
    report["clip_low"] = clip_low
    report["clip_high"] = clip_high
    report["clip_ratio"] = (clip_low + clip_high) / n_visible
    report["mean_target"] = stats_df["mean"]
    report["std_target"] = stats_df["std"]
    report["skew_target"] = stats_df["skew"]
    report["kurt_target"] = stats_df["kurt"]
    report["mean_synth"] = X_synth.mean()
    report["std_synth"] = X_synth.std(ddof=1)
    report["skew_synth"] = X_synth.skew()
    report["kurt_synth"] = X_synth.kurt()
    return report


if __name__ == "__main__":
    # ---------- 命令行自检：需在本目录（case_studies/lazy_trading）下运行 ----------
    import polars as pl
    from skfolio.preprocessing import prices_to_returns

    prices = pl.read_parquet("hs_funds_prices.parquet").to_pandas()
    prices = prices.set_index("timestamp").ffill()
    prices = prices[prices.index.year > 2015]
    X = prices_to_returns(prices, drop_inceptions_nan=False).iloc[:, :80]  # 前 80 列快速自检
    print(f"X: {X.shape[0]} 天 x {X.shape[1]} 资产 | 缺失单元格 {int(X.isna().sum().sum())}")

    stats_table = estimate_marginal_stats(X, distribution="student_t")
    print("\n[student_t] 逐列统计属性（摘要）:")
    print(stats_table[["n_valid", "mean", "std", "skew", "kurt", "nu"]].describe().round(4))

    for dist_name in ("student_t", "gaussian"):
        X_fake, report = synthesize_returns(
            X, distribution=dist_name, random_state=42, return_report=True)
        assert (X.isna() == X_fake.isna()).all().all(), "缺失模式不一致"
        print(f"\n[{dist_name}] 合成完成 {X_fake.shape} | 掩码一致 | "
              f"平均截断比例 {report['clip_ratio'].mean():.4%}")
        print(report[["loc", "scale", "nu", "clip_ratio",
                      "std_target", "std_synth"]].head(5).round(5))