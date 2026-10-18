"""Pre-selection module: 资产预筛选转换器集合。

* DropCorrelated：基于相关性的冗余剔除，在 skfolio 的三步贪心框架上
  扩展相关矩阵的构造口径（朴素 / 降噪 / 去市场成分）。
* DropTailCorrelated：基于尾部风险相关性的冗余剔除，实现思路参考
  skfolio 的 DropCorrelated（三步贪心剔除框架），将全样本相关性替换
  为"单边条件尾部联动指标"（尾部依赖系数或条件尾部相关）。
* VolumePreSelection：基于平均成交量的流动性预筛选，按成交量保留
  流动性最好的资产（需额外传入 volumes，见类文档）。
* SelectKExtremes：按指定度量值保留 k 个最优或最差资产，k 支持
  整数数量或浮点比例（增强自 skfolio 的 SelectKExtremes）。
"""

from __future__ import annotations

import numbers
import warnings

import numpy as np
import scipy.optimize as sco
import sklearn.base as skb
import sklearn.feature_selection as skf
import sklearn.neighbors as skn
import sklearn.utils.validation as skv

from skfolio.measures import RatioMeasure
from skfolio.population import Population
from skfolio.portfolio import Portfolio
from skfolio.typing import ArrayLike, BoolArray

import skfolio.typing as skt

# 数值兜底阈值：避免除零与浮点边界造成的负开方/负对角元素
_EPS = 1e-13


class SelectKExtremes(skf.SelectorMixin, skb.BaseEstimator):
    """按指定度量值保留 k 个最优或最差资产（增强自 skfolio 的 SelectKExtremes）。

    业务目标：在组合构建前根据单一风险/收益度量（夏普比率、峰度、
    波动率等）缩小资产宇宙，保留度量值最高（highest=True）或最低
    （highest=False）的 k 个资产。

    k 参数增强（相对 skfolio 原版）：

        * k 为整数：直接作为保留数量，与原版语义一致；
        * k 为浮点且 0 < k < 1：视为保留比例，fit 时按输入资产数
          自动折算：k_actual = round(k * n_assets)，折算不足 1 只时
          至少保留 1 只（便于在资产数不固定的场景下使用比例筛选）；
        * k 为浮点且 k >= 1：按数量取整处理（与整数语义一致，避免
          将 200.0 误当作 20000% 比例导致意外全选）。

    与 skfolio 一致：解析后的 k 大于等于资产数量时，全部资产保留。

    Parameters
    ----------
    k : int or float, default=10
        保留数量（整数）或保留比例（浮点且 0 < k < 1）。

    measure : Measure, default=RatioMeasure.SHARPE_RATIO
        用于排序资产的 skfolio 度量，如 RatioMeasure.SHARPE_RATIO、
        ExtraRiskMeasure.KURTOSIS 等。

    highest : bool, default=True
        True 时保留度量值最高的 k 个资产，否则保留最低的 k 个。

    Attributes
    ----------
    to_keep_ : BoolArray of shape (n_assets,)
        布尔数组，True 表示该资产被保留。

    n_features_in_ : int
        fit 时观测到的资产数量。

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        fit 时观测到的资产名称，仅当 X 的列名全为字符串时定义。
    """

    to_keep_: BoolArray

    def __init__(
        self,
        k: int | float = 10,
        measure: skt.Measure = RatioMeasure.SHARPE_RATIO,
        highest: bool = True,
    ):
        self.k = k
        self.measure = measure
        self.highest = highest

    def _validate_params(self) -> None:
        if not isinstance(self.k, (numbers.Integral, numbers.Real)):
            raise ValueError("`k` must be an int (count) or a float (ratio)")
        if self.k <= 0:
            raise ValueError("`k` must be strictly positive")

    def _resolve_k(self, n_assets: int) -> int:
        """把 k 解析为实际保留数量：整数直接用，浮点比例折算。"""
        if isinstance(self.k, numbers.Integral):
            k = int(self.k)
        else:
            k = float(self.k)
            if 0 < k < 1:
                k = max(1, int(round(k * n_assets)))
            else:
                k = int(k)
        return min(k, n_assets)

    def fit(self, X: ArrayLike, y=None) -> SelectKExtremes:
        """运行 SelectKExtremes 筛选，得到应保留的资产。

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            资产收益率。

        y : Ignored
            未使用，仅为保持 API 一致性。

        Returns
        -------
        self : SelectKExtremes
            已拟合的转换器。
        """
        X = skv.validate_data(self, X)
        self._validate_params()
        n_assets = X.shape[1]
        k = self._resolve_k(n_assets)

        # k 达到或超过资产总数时直接全选，避免无谓的度量计算
        if k >= n_assets:
            self.to_keep_ = np.ones(n_assets, dtype=bool)
            return self

        # 构建单资产组合 Population，按 measure 排序取前 k 个
        population = Population([])
        for i in range(n_assets):
            weights = np.zeros(n_assets)
            weights[i] = 1
            population.append(Portfolio(X=X, weights=weights))

        selected = population.sort_measure(
            measure=self.measure, reverse=self.highest
        )[:k]
        selected_idx = [x.nonzero_assets_index[0] for x in selected]
        self.to_keep_ = np.isin(np.arange(n_assets), selected_idx)
        return self

    def _get_support_mask(self):
        skv.check_is_fitted(self)
        return self.to_keep_


class DropCorrelated(skf.SelectorMixin, skb.BaseEstimator):
    """剔除高相关冗余资产（增强自 skfolio 的 DropCorrelated）。

    业务目标：在组合构建前剔除跟踪同一标的（或高度重叠）的冗余资产，
    避免同一风险被重复计入。

    skfolio 原版的相关性统计量就是 `corr = np.corrcoef(X.T)`，既不降噪也
    不去市场成分。本实现保持三步贪心剔除框架与 `threshold`/`absolute`
    参数语义完全不变，仅将相关矩阵的构造口径抽成 `corr_method`：

        * "naive"（默认）：与 skfolio.pre_selection.DropCorrelated 一致，
          直接用全样本皮尔逊相关矩阵。
        * "denoise"：对相关矩阵作特征分解，用 Marchenko-Pastur 定律拟合
          噪声特征值上界，将落入噪声区间的特征值替换为常数后重建（
          López de Prado (2020)），再重新归一化对角。
        * "detone"：减去最大的 `n_markets` 个特征值对应的成分（市场成分），
          再重新归一化对角，只保留资产间的结构性联动。

    三步贪心剔除框架（与 skfolio 一致）：

        * Step 1: 选出所有相关性高于阈值的资产对；
        * Step 2: 按相关性从高到低排序；
        * Step 3: 逐对处理，若两资产都未被剔除，则剔除与其余资产平均
          相关性更强的那一方。

    口径选择提示（实测结论，按资产池 N、样本长度 T 选型）：

        * detone 去的是"共同市场 beta 造成的伪相关"：同指数克隆资产在去市场
          后相关仍接近 1（冗余照抓），而仅靠市场因子联动的相关会被压到 0 附近
          甚至为负（误杀消失）。代价是相关分布中心移到 0 且含负值，
          `threshold` 需要相对 naive 口径重新调域，`absolute=True` 的含义也变为
          "强负相关（对冲对）也算冗余"；`n_markets` 建议保持 1，取 3/5 时残差
          矩阵已非合法相关结构，个别元素反而会反弹。
        * denoise 在 q = T/N 较大时近似恒等变换（对剔除决策几乎无影响），q ≲ 2
          时会过度收缩、把真冗余的相关压到阈值以下造成漏检，q ≤ 1 时直接报错。
          建议只在 q ≳ 5 的长窗口上使用。
        * 成本量级（N=20, T=1500）：naive ≈ 0.1 ms，detone ≈ 0.9 ms，
          denoise ≈ 7 ms（含 Marchenko-Pastur 拟合与 KDE）。

    退化列处理：零方差/非有限收益的资产在朴素相关矩阵中会产生 NaN，而 NaN 一旦
    进入特征分解就会污染整个矩阵（全部配对比较均为 False → 静默不剔除）。因此遇到
    这类资产时本实现发出 UserWarning，并把它们的相关行/列置 0（自身对角仍为 1，
    等价于视为与其它资产无关的独立资产，因而总是被保留）；denoise/detone 的谱
    处理只在非退化资产子矩阵上进行后再回填——否则退化资产自带的单位特征值会污染
    Marchenko-Pastur 拟合（实测会使降噪完全失效、一个不剔）。
    无退化列时 `corr_method="naive"` 与 skfolio 逐资产结果完全一致；上游已按惯例
    将 `DropZeroVariance` 放在本转换器之前时，不会触发该告警。

    Parameters
    ----------
    threshold : float, default=0.95
        相关性阈值，高于该值的资产对将被剔除其一，取值 [-1, 1]。

    absolute : bool, default=False
        为 True 时对相关性取绝对值（高度负相关的资产也视为冗余）。

    corr_method : str, default="naive"
        相关矩阵构造口径："naive" 全样本相关；"denoise" 降噪后相关；
        "detone" 去市场成分后相关。

    n_markets : int, default=1
        市场成分对应的特征向量个数，仅对 `corr_method="detone"` 生效；
        必须满足 1 <= n_markets < n_assets。

    Attributes
    ----------
    corr_ : ndarray of shape (n_assets, n_assets)
        按 `corr_method` 处理后、取绝对值之前的相关矩阵。

    to_keep_ : ndarray of shape (n_assets,)
        布尔数组，True 表示该资产被保留。

    n_features_in_ : int
        fit 时观测到的资产数量。

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        fit 时观测到的资产名称，仅当 X 的列名全为字符串时定义。

    References
    ----------
    .. [1]  "Machine Learning for Asset Managers".
        Elements in Quantitative Finance.
        López de Prado (2020).
    """

    to_keep_: BoolArray
    corr_: np.ndarray

    def __init__(
        self,
        threshold: float = 0.95,
        absolute: bool = False,
        corr_method: str = "naive",
        n_markets: int = 1,
    ):
        self.threshold = threshold
        self.absolute = absolute
        self.corr_method = corr_method
        self.n_markets = n_markets

    def _validate_params(self, n_valid: int) -> None:
        if not -1 <= self.threshold <= 1:
            raise ValueError("`threshold` must be between -1 and 1")
        if not isinstance(self.absolute, (bool, np.bool_)):
            raise ValueError("`absolute` must be a boolean")
        if self.corr_method not in ("naive", "denoise", "detone"):
            raise ValueError(
                "`corr_method` must be 'naive', 'denoise' or 'detone', "
                f"got {self.corr_method!r}"
            )
        if not isinstance(self.n_markets, numbers.Integral) or self.n_markets < 1:
            raise ValueError("`n_markets` must be a positive integer")
        if self.corr_method == "detone" and self.n_markets >= n_valid:
            raise ValueError(
                "`n_markets` must be smaller than the number of assets with "
                f"non-degenerate returns ({n_valid}), got {self.n_markets}"
            )

    @staticmethod
    def _base_corr(X: np.ndarray) -> tuple[np.ndarray, BoolArray]:
        """朴素全样本相关矩阵，并将退化列产生的 NaN 污染归零。

        零方差/非有限收益的资产行与列被置 0（对角仍为 1），等价于将其视为
        与其它资产无关的独立资产；返回的 valid 掩码标记可参与谱处理的资产。
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            # 零方差列会让 np.corrcoef / np.std 自己报 invalid 或 divide，带来的
            # RuntimeWarning 会淹没本类的专用告警，此处压掉，统一由下方消毒处理。
            corr = np.corrcoef(X.T)
            std = X.std(axis=0, ddof=1)
        if corr.ndim == 0:  # 单资产时 np.corrcoef 返回标量
            corr = corr.reshape(1, 1)
        invalid = ~np.isfinite(std) | (std <= 0)
        corr = np.where(np.isfinite(corr), corr, 0.0)
        if invalid.any():
            corr[invalid, :] = 0.0
            corr[:, invalid] = 0.0
        np.fill_diagonal(corr, 1.0)
        return corr, ~invalid

    @staticmethod
    def _eigh_sorted(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """对称矩阵特征分解，特征值从大到小排序（含同步的特征向量列）。"""
        e_val, e_vec = np.linalg.eigh(corr)
        indices = e_val.argsort()[::-1]
        return e_val[indices], e_vec[:, indices]

    @staticmethod
    def _renormalize_corr(corr: np.ndarray) -> np.ndarray:
        """按对角重新归一化为相关矩阵（等价 skfolio 的 cov_to_corr）。

        降噪/去市场后的矩阵对角不再为 1，且可能出现非正对角（矩阵已不是
        合法相关结构），此处做兜底：非正对角按 1 处理，避免开方与除零产生
        NaN/Inf。
        """
        diag = np.diag(corr)
        scale = np.where(diag > _EPS, np.sqrt(np.clip(diag, _EPS, None)), 1.0)
        corr = corr / np.outer(scale, scale)
        np.fill_diagonal(corr, 1.0)
        return corr

    def _denoise_corr(self, corr: np.ndarray, n_observations: int) -> np.ndarray:
        """Marchenko-Pastur 降噪：将噪声区间的特征值替换为常数后重建相关矩阵。"""
        n_assets = corr.shape[0]
        q = n_observations / n_assets
        if q <= 1:
            raise ValueError(
                "`corr_method='denoise'` needs more observations than assets "
                f"(q = T/N > 1), got T={n_observations}, N={n_assets} "
                f"(q={q:.2f}); use 'naive' or 'detone' for short windows"
            )

        e_val, e_vec = self._eigh_sorted(corr)

        def _noise_upper(x_var: float) -> float:
            """噪声区间的特征值上界（Marchenko-Pastur 右边缘）。"""
            return x_var * (1.0 + (1.0 / q) ** 0.5) ** 2

        def _marchenko(x: np.ndarray) -> float:
            """Marchenko-Pastur 密度与特征值 KDE 之间的平方误差。"""
            x_var = x[0]
            e_min = max(x_var * (1.0 - (1.0 / q) ** 0.5) ** 2, _EPS)
            e_max = _noise_upper(x_var)
            e_val_lin = np.linspace(e_min, e_max, 1000)
            pdf_0 = (
                q
                / (2 * np.pi * x_var * e_val_lin)
                * np.sqrt(np.clip((e_max - e_val_lin) * (e_val_lin - e_min), 0.0, None))
            )
            kde = skn.KernelDensity(kernel="gaussian", bandwidth=0.01).fit(
                e_val.reshape(-1, 1)
            )
            pdf_1 = np.exp(kde.score_samples(e_val_lin.reshape(-1, 1)))
            return float(np.sum((pdf_1 - pdf_0) ** 2))

        res = sco.minimize(_marchenko, x0=0.5, bounds=((1e-5, 1 - 1e-5),))
        n_facts = e_val.shape[0] - e_val[::-1].searchsorted(_noise_upper(res["x"][0]))
        if n_facts >= n_assets:
            # 全部特征值都在信号区间，无噪声可压，保持原矩阵
            return corr
        e_val_ = e_val.copy()
        e_val_[n_facts:] = e_val_[n_facts:].sum() / float(n_assets - n_facts)
        return self._renormalize_corr(e_vec @ np.diag(e_val_) @ e_vec.T)

    def _detone_corr(self, corr: np.ndarray) -> np.ndarray:
        """去市场化：减去最大 `n_markets` 个特征值对应的市场成分后重建。"""
        e_val, e_vec = self._eigh_sorted(corr)
        market_val = e_val[: self.n_markets]
        market_vec = e_vec[:, : self.n_markets]
        market_corr = market_vec @ np.diag(market_val) @ market_vec.T
        return self._renormalize_corr(corr - market_corr)

    def _warn_degenerate(self, valid: BoolArray) -> None:
        """对零方差/非有限收益资产发出告警。"""
        idx = np.flatnonzero(~valid)
        names = getattr(self, "feature_names_in_", None)
        labels = ", ".join(str(names[i]) if names is not None else str(i) for i in idx)
        warnings.warn(
            f"DropCorrelated: {len(idx)}/{len(valid)} asset(s) with zero or "
            f"non-finite variance [{labels}] are treated as uncorrelated with "
            "all other assets (hence always kept) and excluded from the "
            "spectral estimation; put `DropZeroVariance` before this "
            "transformer to drop them.",
            UserWarning,
            stacklevel=2,
        )

    def _process_corr(
        self, corr: np.ndarray, valid: BoolArray, n_observations: int
    ) -> np.ndarray:
        """按 `corr_method` 加工用于剔除决策的相关矩阵。

        谱处理（denoise/detone）只在非退化资产构成的子矩阵上进行：退化资产在
        行/列置零后自带一个单位特征值，会混进 Marchenko-Pastur 拟合并扭曲
        降噪结果；处理完再回填，保持这些资产与其它资产零联动。无退化资产时
        子矩阵即全矩阵，结果与直接处理等价。
        """
        if self.corr_method == "naive":
            return corr
        idx = np.flatnonzero(valid)
        if idx.size < 2:
            return corr
        sub = corr[np.ix_(idx, idx)]
        if self.corr_method == "denoise":
            sub = self._denoise_corr(sub, n_observations)
        else:
            sub = self._detone_corr(sub)
        corr[np.ix_(idx, idx)] = sub
        return corr

    def fit(self, X: ArrayLike, y=None) -> DropCorrelated:
        """运行相关性筛选，得到应保留的资产。

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            资产收益率。

        y : Ignored
            未使用，仅为保持 API 一致性。

        Returns
        -------
        self : DropCorrelated
            已拟合的转换器。
        """
        X = skv.validate_data(self, X)
        n_assets = X.shape[1]
        corr, valid = self._base_corr(X)
        self._validate_params(n_valid=int(valid.sum()))
        if not valid.all():
            self._warn_degenerate(valid)
        corr = self._process_corr(corr, valid, X.shape[0])
        self.corr_ = corr
        # 配对排序与平均相关度量保持同一量纲（与 skfolio 一致）
        decisions = np.abs(corr) if self.absolute else corr
        mean_corr = decisions.mean(axis=0)

        triu_idx = np.triu_indices(n_assets, 1)

        # 选出所有相关性高于阈值的资产对
        selected_idx = np.argwhere(decisions[triu_idx] > self.threshold).flatten()

        # 按相关性从高到低排序
        selected_idx = selected_idx[np.argsort(-decisions[triu_idx][selected_idx])]

        # 逐对处理：若两资产都未被剔除，保留与其余资产平均相关性更低的一方
        to_remove = set()
        for idx in selected_idx:
            i, j = triu_idx[0][idx], triu_idx[1][idx]
            if i not in to_remove and j not in to_remove:
                if mean_corr[i] > mean_corr[j]:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
        self.to_keep_ = ~np.isin(np.arange(n_assets), list(to_remove))
        return self

    def _get_support_mask(self):
        skv.check_is_fitted(self)
        return self.to_keep_


class DropTailCorrelated(skf.SelectorMixin, skb.BaseEstimator):
    """剔除在市场极端行情下与持仓高度联动的资产。

    业务目标：当组合持仓暴跌时，候选资产不应跟着暴跌。因此对每对资产
    (i, j) 计算两个方向的"单边条件"尾部联动指标：

        * i→j：以 i 处于尾部为条件，衡量 j 的联动程度；
        * j→i：以 j 处于尾部为条件，衡量 i 的联动程度。

    矩阵元素取两方向的较大值（保守口径）：只要任一方暴跌时另一方跟跌，
    即视为尾部联动，随后复用 DropCorrelated 的贪心剔除算法：

        * Step 1: 选出所有尾部联动高于阈值的资产对；
        * Step 2: 按联动强度从高到低排序；
        * Step 3: 逐对处理，若两资产都未被剔除，则剔除与其余资产平均
          尾部联动更强的那一方。

    注意：不使用"双方同时处于尾部"的联合条件（超越相关 Corr(X_i, X_j |
    X_i∈Tail ∧ X_j∈Tail)）——它只保留共跌样本，会系统性高估尾部联动，
    误杀"i 暴跌而 j 不跌"的优质分散化资产。

    联动指标（measure）：

        * "tdc"（默认）：尾部依赖系数 λ(i→j) = P(X_j ∈ Tail_j |
          X_i ∈ Tail_i)，即"i 跌入尾部时 j 也跌入尾部的概率"，直接
          对应风险传染强度。两资产独立时期望值约为 quantile，完全
          联动时为 1。基于计数实现，样本需求小、估计稳定。
        * "corr"：条件尾部相关 Corr(X_i, X_j | X_i ∈ Tail_i)，仅以
          单个资产的尾部为条件，条件样本量约为 q·n。

    Parameters
    ----------
    threshold : float, default=0.9
        尾部联动阈值，高于该值的资产对将被剔除其一。tdc 模式下独立
        基线约为 quantile（如 0.05），λ=0.5 已表示"条件资产一半的
        尾部日都在联动"，属强传染；建议从 0.3~0.5 开始调参。

    tail : str, default="lower"
        尾部方向："lower" 表示左侧尾部（极端下跌），
        "upper" 表示右侧尾部（极端上涨）。

    quantile : float, default=0.05
        尾部分位数，取值 (0, 1)。例如 0.05 表示收益率最低的 5% 作为
        左侧尾部样本，0.1 表示最低的 10%。

    measure : str, default="tdc"
        尾部联动指标："tdc" 为尾部依赖系数（条件概率），
        "corr" 为条件尾部相关系数。

    absolute : bool, default=False
        为 True 时对联动指标取绝对值（高度负相关的资产也视为冗余）。
        仅对 measure="corr" 生效，tdc 为概率值本身非负。

    min_joint_obs : int, default=5
        条件资产（处于尾部的一方）所需的最小尾部观测数，低于该值的
        方向联动视为 0。注意：仅 2 个观测点的相关系数恒为 ±1，会误
        触发阈值，因此该值不能小于 2，建议不小于 5。

    Attributes
    ----------
    to_keep_ : ndarray of shape (n_assets,)
        布尔数组，True 表示该资产被保留。

    n_features_in_ : int
        fit 时观测到的资产数量。

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        fit 时观测到的资产名称，仅当 X 的列名全为字符串时定义。
    """

    to_keep_: BoolArray

    def __init__(
        self,
        threshold: float = 0.9,
        tail: str = "lower",
        quantile: float = 0.05,
        measure: str = "tdc",
        absolute: bool = False,
        min_joint_obs: int = 5,
    ):
        self.threshold = threshold
        self.tail = tail
        self.quantile = quantile
        self.measure = measure
        self.absolute = absolute
        self.min_joint_obs = min_joint_obs

    def _validate_params(self) -> None:
        if not -1 <= self.threshold <= 1:
            raise ValueError("`threshold` must be between -1 and 1")
        if not 0 < self.quantile < 1:
            raise ValueError("`quantile` must be between 0 and 1 (exclusive)")
        if self.tail not in ("lower", "upper"):
            raise ValueError("`tail` must be 'lower' or 'upper'")
        if self.measure not in ("tdc", "corr"):
            raise ValueError("`measure` must be 'tdc' or 'corr'")
        if self.min_joint_obs < 2:
            raise ValueError("`min_joint_obs` must be at least 2")

    def _directional_corr(
        self, X: np.ndarray, tail_mask: np.ndarray, i: int, j: int
    ) -> float:
        """i→j 方向的条件尾部相关：在 i 处于尾部的样本上计算 corr(X_i, X_j)。"""
        mask = tail_mask[:, i]
        if mask.sum() < self.min_joint_obs:
            return 0.0
        xi, xj = X[mask, i], X[mask, j]
        if xi.std() == 0.0 or xj.std() == 0.0:
            return 0.0
        c = np.corrcoef(xi, xj)[0, 1]
        return 0.0 if np.isnan(c) else float(c)

    def _build_tail_corr(self, X: np.ndarray) -> np.ndarray:
        """构造 n_assets × n_assets 的尾部联动矩阵，两方向取较大者。"""
        n_assets = X.shape[1]
        q_level = self.quantile if self.tail == "lower" else 1.0 - self.quantile
        per_asset_q = np.quantile(X, q_level, axis=0)
        tail_mask = X <= per_asset_q if self.tail == "lower" else X >= per_asset_q

        if self.measure == "tdc":
            # λ(i→j) = P(j ∈ Tail_j | i ∈ Tail_i)，计数实现可向量化
            n_tail = tail_mask.sum(axis=0)
            joint = tail_mask.astype(np.float64).T @ tail_mask.astype(np.float64)
            lam = np.zeros((n_assets, n_assets))
            valid = n_tail >= self.min_joint_obs
            lam[valid, :] = joint[valid, :] / n_tail[valid, None]
            np.fill_diagonal(lam, 0.0)
            tail_corr = np.maximum(lam, lam.T)
        else:  # "corr"
            tail_corr = np.zeros((n_assets, n_assets))
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    c = max(
                        self._directional_corr(X, tail_mask, i, j),
                        self._directional_corr(X, tail_mask, j, i),
                    )
                    tail_corr[i, j] = tail_corr[j, i] = c
        np.fill_diagonal(tail_corr, 1.0)
        if self.absolute:
            tail_corr = np.abs(tail_corr)
        return tail_corr

    def fit(self, X: ArrayLike, y=None):
        """运行尾部相关性筛选，得到应保留的资产。

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            资产收益率。

        y : Ignored
            未使用，仅为保持 API 一致性。

        Returns
        -------
        self : DropTailCorrelated
            已拟合的转换器。
        """
        X = skv.validate_data(self, X)
        self._validate_params()

        n_assets = X.shape[1]
        tail_corr = self._build_tail_corr(X)
        mean_corr = tail_corr.mean(axis=0)

        triu_idx = np.triu_indices(n_assets, 1)

        # 选出所有尾部相关性高于阈值的资产对
        selected_idx = np.argwhere(tail_corr[triu_idx] > self.threshold).flatten()

        # 按尾部相关性从高到低排序
        selected_idx = selected_idx[np.argsort(-tail_corr[triu_idx][selected_idx])]

        # 逐对处理：若两资产都未被剔除，保留与其余资产平均尾部相关性更低的一方
        to_remove = set()
        for idx in selected_idx:
            i, j = triu_idx[0][idx], triu_idx[1][idx]
            if i not in to_remove and j not in to_remove:
                if mean_corr[i] > mean_corr[j]:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
        self.to_keep_ = ~np.isin(np.arange(n_assets), list(to_remove))
        return self

    def _get_support_mask(self):
        skv.check_is_fitted(self)
        return self.to_keep_


class VolumePreSelection(skf.SelectorMixin, skb.BaseEstimator):
    """按平均成交量保留流动性最好的资产（流动性预筛选）。

    业务目标：在组合构建前剔除流动性差的资产，降低交易成本与冲击
    成本的影响。对每只资产计算全样本平均成交量，保留成交量最高的
    前 `pct_to_keep` 比例的资产。

    注意：本选择器需要额外的成交量输入 `volumes`，而普通 `Pipeline.fit`
    只会传递 X 与 y，因此 `model.fit(X)` 场景下无法拿到 volumes。
    两种可用方式：

    1) skfolio 的 cross_val_predict（官方推荐）：启用 metadata routing
       后 volumes 作为路由参数传入，且按每折 train 索引自动切片：

           set_config(enable_metadata_routing=True)
           pipe = Pipeline([
               ("pre_selection", VolumePreSelection(pct_to_keep=0.5)
                                 .set_fit_request(volumes=True)),
               ("optimization", EqualWeighted()),
           ])
           cross_val_predict(pipe, X, cv=cv, params={"volumes": volumes})

       volumes 须与全量 X 同形状（行数 = 总观测数）；skfolio 0.20 的
       该通道不支持 callable 值（会直接访问 .shape 报错）。

    2) 两段式手动 fit（方案B 等普通 fit 场景）：先 fit 再 transform

           sel = VolumePreSelection(pct_to_keep=0.5)
           sel.fit(X_window, volumes=volumes_window)   # 按窗口手动对齐
           X_sel = sel.transform(X_window)

    Parameters
    ----------
    pct_to_keep : float, default=0.5
        保留比例，取值 (0, 1]。例如 0.5 表示保留平均成交量最高的 50%
        资产；比例折算不足 1 只时至少保留 1 只。

    Attributes
    ----------
    to_keep_ : BoolArray of shape (n_assets,)
        布尔数组，True 表示该资产被保留。

    n_features_in_ : int
        fit 时观测到的资产数量。

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        fit 时观测到的资产名称，仅当 X 的列名全为字符串时定义。
    """

    to_keep_: BoolArray

    def __init__(self, pct_to_keep: float = 0.5):
        self.pct_to_keep = pct_to_keep

    def _validate_params(self) -> None:
        if not 0 < self.pct_to_keep <= 1:
            raise ValueError("`pct_to_keep` must be between 0 and 1 (exclusive of 0)")

    def fit(self, X: ArrayLike, y=None, volumes: ArrayLike | None = None):
        """按平均成交量运行流动性筛选，得到应保留的资产。

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            资产收益率（仅用于对齐资产维度与数量）。

        y : Ignored
            未使用，仅为保持 API 一致性。

        volumes : array-like of shape (n_observations, n_assets), optional
            与 X 同形状的成交量数据（如日成交量/成交额），必填。

        Returns
        -------
        self : VolumePreSelection
            已拟合的转换器。
        """
        X = skv.validate_data(self, X)
        self._validate_params()

        if volumes is None:
            raise ValueError("`volumes` is required: pass an array of shape X.shape")

        # Validate and convert volumes to a NumPy array
        volumes = skv.check_array(
            volumes,
            accept_sparse=False,
            ensure_2d=False,
            dtype=[np.float64, np.float32],
            order="C",
            copy=False,
            input_name="volumes",
        )
        if volumes.shape != X.shape:
            raise ValueError(
                f"Volume data {volumes.shape} must have the same dimensions as X {X.shape}"
            )

        n_assets = X.shape[1]
        mean_volumes = volumes.mean(axis=0)

        # Select the top `pct_to_keep` assets with the highest average volumes
        n_to_keep = max(1, int(round(self.pct_to_keep * n_assets)))
        selected_idx = np.argsort(mean_volumes)[-n_to_keep:]

        # Performance tip: `argpartition` could be used here for better efficiency
        # (O(n log(n)) vs O(n)).
        self.to_keep_ = np.isin(np.arange(n_assets), selected_idx)
        return self

    def _get_support_mask(self):
        skv.check_is_fitted(self)
        return self.to_keep_
