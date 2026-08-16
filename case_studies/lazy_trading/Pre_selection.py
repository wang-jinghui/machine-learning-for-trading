"""Pre-selection DropTailCorrelated module.

基于尾部风险相关性的资产预筛选转换器，实现思路参考 skfolio 的
DropCorrelated（三步贪心剔除框架），将全样本相关性替换为"单边条件
尾部联动指标"（尾部依赖系数或条件尾部相关）。
"""

from __future__ import annotations

import numpy as np
import sklearn.base as skb
import sklearn.feature_selection as skf
import sklearn.utils.validation as skv

from skfolio.typing import ArrayLike, BoolArray


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
