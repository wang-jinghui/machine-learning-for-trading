# 高斯混合模型 vs 贝叶斯高斯混合模型

## 一、概述

高斯混合模型（Gaussian Mixture Model, GMM）假设观测数据由 $K$ 个高斯分布叠加而成，每个分量由均值 $\mu_k$、协方差 $\Sigma_k$ 和混合权重 $\pi_k$ 参数化：

$$p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

`scikit-learn` 提供了两种实现：

| 类 | 文件 | 推断方式 | 核心特点 |
|----|------|----------|----------|
| `GaussianMixture` | `sklearn.mixture.GaussianMixture` | 最大似然估计（MLE），通过 EM 算法 | K 固定，支持 BIC/AIC 信息准则 |
| `BayesianGaussianMixture` | `sklearn.mixture.BayesianGaussianMixture` | 变分推断（Variational Inference） | K 可自动推断，内置共轭先验，支持分量淘汰 |

---

## 二、GaussianMixture（标准 GMM）

### 2.1 工作原理

EM（Expectation-Maximization）算法迭代执行两步：

1. **E 步**：计算每个样本属于每个分量的后验概率（软分配）$\gamma_{ik}$
2. **M 步**：基于软分配更新 $\pi_k$、$\mu_k$、$\Sigma_k$

直到对数似然收敛或达到最大迭代次数。

### 2.2 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_components` | int | `1` | 高斯分量数量（聚类数 K） |
| `covariance_type` | str | `'full'` | 协方差类型：`'full'`（全协方差）、`'tied'`（共享）、`'diag'`（对角）、`'spherical'`（球型） |
| `weights_init` | array-like or None | `None` | 初始权重，None 则均匀分配 |
| `means_init` | array-like or None | `None` | 初始均值 |
| `precisions_init` | array-like or None | `None` | 初始精度矩阵（协方差的逆） |
| `reg_covar` | float | `1e-6` | 加到协方差对角线上的正则化常数 |
| `max_iter` | int | `100` | EM 最大迭代次数 |
| `tol` | float | `1e-4` | 对数似然的收敛阈值 |
| `warm_start` | bool | `False` | True 时多次 `fit` 从上次结果继续 |
| `n_init` | int | `1` | 不同初始值运行 EM 的次数，选对数似然最佳者 |
| `init_params` | str | `'kmeans'` | `'kmeans'`（K-Means++）或 `'random'` |
| `random_state` | int or None | `None` | 随机种子 |
| `verbose` / `verbose_interval` | int | `0` / `10` | 日志详细程度与输出间隔 |

> **先验参数（可选，用于有监督场景）**：`mean_prior`、`covariance_prior`、`precision_norm`、`weight_concentration`。

### 2.3 核心方法

| 方法 | 说明 |
|------|------|
| `fit(X, y=None)` | 拟合模型，返回 `self` |
| `fit_predict(X)` | 拟合并返回每个样本的标签 |
| `predict(X)` | 硬分配：返回最可能的分量标签 |
| `predict_proba(X)` | 软分配：返回属于各分量的后验概率 |
| `score(X)` | 返回平均对数似然 |
| `score_samples(X)` | 返回每个样本的对数似然 |
| `bic(X)` | **贝叶斯信息准则**（越低越好，惩罚更重） |
| `aic(X)` | **赤池信息准则**（越低越好，倾向于更多分量） |
| `sample(n_samples)` | 从模型中采样数据，返回 `(X, y)` |

### 2.4 拟合后属性

| 属性 | 形状 | 说明 |
|------|------|------|
| `weights_` | `(n_components,)` | 学到的分量权重 |
| `means_` | `(n_components, n_features)` | 各分量均值 |
| `covariances_` | 取决于 `covariance_type` | 协方差矩阵 |
| `precisions_` | 取决于 `covariance_type` | 精度矩阵（协方差的逆） |
| `converged_` | bool | EM 是否收敛 |
| `n_iter_` | int | 实际迭代次数 |

### 2.5 协方差类型对比

| 类型 | 自由度（每分量） | 适用场景 |
|------|-----------------|----------|
| `'full'` | $D(D+1)/2$ | 最灵活，特征间相关性重要时 |
| `'tied'` | $D(D+1)/2$（共享） | 各分量形状相似时 |
| `'diag'` | $D$ | 高维数据，假设特征独立 |
| `'spherical'` | $1$ | 最简，各向同性，适合球状簇 |

---

## 三、BayesianGaussianMixture（贝叶斯 GMM）

### 3.1 工作原理

在标准 GMM 的基础上为每个参数引入**共轭先验**：

| 参数 | 先验分布 | 超参数 |
|------|----------|--------|
| 权重 $\pi$ | **Dirichlet** 分布 | `weight_concentration_prior`（浓度 $\alpha$） |
| 均值 $\mu_k$ | **Gaussian** 分布 | `mean_precision_prior`（精度 $\kappa_0$） |
| 协方差 $\Sigma_k$ | **Inverse-Wishart** 分布 | `df_prior`（自由度 $\nu_0$） |

使用**变分推断**近似后验，最大化证据下界（ELBO, Evidence Lower BOund）。

### 3.2 关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_components` | int | `1` | 分量的**最大**数量；实际使用的可能更少 |
| `covariance_type` | str | `'full'` | 同上 |
| **`weight_concentration_prior`** | float > 0 | **`1e-6`** | ⚠️ Dirichlet 浓度 $\alpha$。默认极小，使模型能自动淘汰不需要的分量。想保留全部分量需设为较大值（如 `1.0`） |
| **`weight_concentration_prior_type`** | str | `'dirichlet_process'` | `'dirichlet_process'`（允许分量淘汰）或 `'mean_field'`（传统独立先验） |
| **`mean_precision_prior`** | float > 0 | `1.0` | 均值先验精度，越大约束越强 |
| **`df_prior`** | float > 0 | `n_features` | 协方差先验自由度，越大约束越强 |
| `reg_covar` | float | `1e-6` | 协方差正则化 |
| `max_iter` / `tol` / `n_init` / `init_params` | — | 同 GMM | 同上 |

### 3.3 核心方法

与 `GaussianMixture` 共享相同 API：`fit()`、`fit_predict()`、`predict()`、`predict_proba()`、`score()`、`score_samples()`、`sample()`。

> ⚠️ **`bic()` 和 `aic()` 不可用**——BGMM 基于变分推断而非 MLE，调用会抛异常。

### 3.4 拟合后属性

除继承 GMM 的属性外，额外提供：

| 属性 | 说明 |
|------|------|
| `weight_concentration_prior_` | 实际使用的 Dirichlet 浓度（数组） |
| `mean_precision_prior_` | 均值先验精度（数组） |
| `df_prior_` | 协方差先验自由度（数组） |
| `cluster_weight_concentration_` | 每个分量学习到的浓度值（`dirichlet_process` 模式） |
| `lowerbound_` | 变分下界（ELBO）值，用于模型评估 |

---

## 四、核心区别对比

| 维度 | `GaussianMixture` | `BayesianGaussianMixture` |
|------|-------------------|---------------------------|
| **推断方式** | 最大似然（MLE），EM 算法 | 变分推断（VI），最大化 ELBO |
| **分量数 K** | 固定，由 `n_components` 决定 | **可自动推断**，权重 → 0 的分量被淘汰 |
| **先验** | 可选（通过 `mean_prior` 等传入） | **内置共轭先验**，默认即生效 |
| **权重稀疏性** | 权重和=1，通过 EM 硬约束，不会自动淘汰 | Dirichlet 先验 + 小 $\alpha$ → 稀疏权重 |
| **信息准则** | ✅ `bic()` / `aic()` 可用 | ❌ 不可用 |
| **模型评估** | 对数似然、BIC、AIC | 变分下界（`lowerbound_`） |
| **训练速度** | 快 | 稍慢（变分迭代更多计算） |
| **对初始值敏感度** | 高（建议 `n_init ≥ 10`） | 同样高（建议 `n_init ≥ 5`） |
| **有监督训练** | 支持（提供 `y` 和先验） | 不支持 |

---

## 五、如何选择？

### 使用 `GaussianMixture` 当：

- **K 的范围已知**，需要通过 BIC/AIC 等准则比较不同 K 的优劣
- 需要信息准则进行模型选择
- 追求简洁可解释的结果
- 数据量较大，训练速度是考量因素
- **典型场景**：市场状态检测、因子聚类、图像分割

### 使用 `BayesianGaussianMixture` 当：

- **K 未知**，希望模型自动推断分量数
- 数据可能存在噪声或不必要的簇
- 想通过先验正则化防止过拟合
- 需要不确定性量化（后验分布而非点估计）
- **典型场景**：异常检测、自动聚类数推断、小样本场景

---

## 六、实践最佳实践

### 6.1 GMM 模型选择（网格搜索 + BIC）

```python
from sklearn.mixture import GaussianMixture

results = {}
for k in range(2, 7):
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          n_init=10, random_state=42, reg_covar=1e-6)
    gmm.fit(X)
    results[k] = {
        "BIC": gmm.bic(X),
        "AIC": gmm.aic(X),
        "log_likelihood": gmm.score(X) * len(X),
    }

# BIC 最低者通常为最优
best_k = min(results, key=lambda k: results[k]["BIC"])
```

### 6.2 BGMM 自动推断 K

```python
from sklearn.mixture import BayesianGaussianMixture

bgmm = BayesianGaussianMixture(
    n_components=10,                  # 设一个足够大的上界
    weight_concentration_prior=1e-6,  # 极小值，允许淘汰
    n_init=5, random_state=42
)
bgmm.fit(X)

# 检查实际使用的分量数
active = (bgmm.weights_ > 0.01).sum()
print(f"推断出的分量数: {active}")
```

### 6.3 BGMM 探路 + GMM 精调（两阶段策略）

> **核心思想**：先用 BGMM 自动推断先验 K₀（利用其分量淘汰能力），再以 K₀ 为中心在
> `[K₀ − Δ, K₀ + Δ]` 范围内用 GMM 网格搜索，以 BIC/AIC 选出最优 K\*。
>
> **为什么这样做？**
> - 盲搜 GMM（如 K=2..10）范围过大、耗时且可能包含不合理候选。
> - BGMM 的先验正则化天然"偏好简洁模型"，推断出的 K₀ 通常接近真实复杂度。
> - GMM 的 BIC/AIC 是基于 MLE 的严格准则，比 BGMM 的 ELBO 更适合做最终决策。
> - 两阶段比全网格少跑模型，速度更快。

```python
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

def two_stage_k_selection(
    X: np.ndarray,
    bgmm_max_k: int = 10,
    search_radius: int = 2,
    covariance_type: str = "full",
    n_init_gmm: int = 10,
    n_init_bgmm: int = 5,
    weight_threshold: float = 0.01,
    random_state: int = 42,
) -> tuple[int, dict[int, dict]:
    """
    两阶段 K 选择：BGMM 探路 → GMM 精调。

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        已标准化的数据矩阵。
    bgmm_max_k : int
        BGMM 的最大分量数（设一个足够大的上界）。
    search_radius : int
        以 K₀ 为中心的搜索半径，最终搜索 [K₀ - Δ, K₀ + Δ]。
    covariance_type : str
        协方差类型，GMM 和 BGMM 保持一致。
    n_init_gmm / n_init_bgmm : int
        各自的重启次数。
    weight_threshold : float
        BGMM 权重低于此值的分量视为"被淘汰"。
    random_state : int
        随机种子。

    Returns
    -------
    best_k : int
        BIC 最优的分量数。
    results : dict[int, dict]
        每个 K 对应的 BIC、AIC、对数似然。
    """
    # ── 阶段 1：BGMM 探路，推断先验 K₀ ──
    bgmm = BayesianGaussianMixture(
        n_components=bgmm_max_k,
        covariance_type=covariance_type,
        weight_concentration_prior=1e-6,   # 极小，允许淘汰
        n_init=n_init_bgmm,
        random_state=random_state,
    )
    bgmm.fit(X)
    k0 = int((bgmm.weights_ > weight_threshold).sum())
    print(f"【阶段 1】BGMM 推断先验 K₀ = {k0}  "
          f"(ELBO = {bgmm.lowerbound_:.1f})")
    print(f"  各分量权重: {np.round(bgmm.weights_, 4)}")

    # ── 阶段 2：以 K₀ 为中心，GMM 网格搜索 ──
    k_min = max(1, k0 - search_radius)
    k_max = k0 + search_radius
    print(f"【阶段 2】GMM 搜索范围: K ∈ [{k_min}, {k_max}]")

    results = {}
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            n_init=n_init_gmm,
            random_state=random_state,
            reg_covar=1e-6,
        )
        gmm.fit(X)
        results[k] = {
            "BIC": gmm.bic(X),
            "AIC": gmm.aic(X),
            "log_likelihood": gmm.score(X) * len(X),
        }

    df = pd.DataFrame(results).T
    df.index.name = "K"
    df = df.round(1)
    print("\nGMM 精调结果:")
    print(df)

    best_k_bic = int(df["BIC"].idxmin())
    best_k_aic = int(df["AIC"].idxmin())
    print(f"\n→ BIC 最优 K* = {best_k_bic}  (BIC = {df.loc[best_k_bic, 'BIC']:,.1f})")
    print(f"→ AIC 最优 K* = {best_k_aic}  (AIC = {df.loc[best_k_aic, 'AIC']:,.1f})")

    return best_k_bic, results
```

**使用示例**（接 `factor_regimes.ipynb` 的数据）：

```python
from sklearn.preprocessing import StandardScaler

# factors_scaled 已在 notebook 中准备好 (1176 months × 9 factors)
best_k, results = two_stage_k_selection(
    X=factors_scaled,
    bgmm_max_k=10,
    search_radius=2,
    n_init_gmm=10,
    n_init_bgmm=5,
)
# 输出示例:
# 【阶段 1】BGMM 推断先验 K₀ = 3  (ELBO = -15234.7)
#   各分量权重: [0.    0.765 0.235]
# 【阶段 2】GMM 搜索范围: K ∈ [1, 5]
#
# GMM 精调结果:
#          BIC       AIC    log_likelihood
# K
# 1   31245.1  29876.3        -14928.1
# 2   26170.4  24612.8        -12275.4     ← BIC 最优
# 3   25891.2  24145.6        -12042.8     ← AIC 更倾向
# 4   25703.8  23769.4        -11861.7
# 5   25534.2  23501.2        -11733.6
#
# → BIC 最优 K* = 2  (BIC = 26,170.4)
# → AIC 最优 K* = 5  (AIC = 23,501.2)
```

**策略流程图**：

```
  标准化数据 X
       │
       ▼
  ┌─────────────────┐
  │  BGMM (max_K)   │  ← 阶段 1：探路
  │  α → 0 (稀疏)   │
  └────────┬────────┘
           │ 推断 K₀ = active_components
           ▼
  ┌──────────────────────────┐
  │  GMM 网格搜索            │  ← 阶段 2：精调
  │  K ∈ [K₀-Δ, K₀+Δ]       │
  │  计算 BIC / AIC          │
  └────────┬─────────────────┘
           │
           ▼
     K* = argmin(BIC)
```

### 6.4 不让 BGMM 淘汰分量

```python
bgmm = BayesianGaussianMixture(
    n_components=4,
    weight_concentration_prior=10.0,  # 大值，保留所有分量
    weight_concentration_prior_type="mean_field",
    n_init=5, random_state=42
)
```

---

## 七、常见注意事项

1. **`n_init` 至关重要**：两种模型都对初始值敏感。`GaussianMixture` 默认 `n_init=1` 容易陷入局部最优，建议设为 `10` 或更高。

2. **`reg_covar` 防止数值不稳定**：当特征共线性强或维度高于样本量时，协方差矩阵可能奇异。调大 `reg_covar`（如 `1e-3`）可改善。

3. **BIC vs AIC**：
   - $\text{BIC} = -2 \ln(\hat{L}) + p \ln(n)$ —— 惩罚项含 $\ln(n)$，样本量大时惩罚更重
   - $\text{AIC} = -2 \hat{L} + 2p$ —— 固定惩罚 $2p$，倾向更多分量
   - 追求**可解释性**选 BIC；追求**预测精度**选 AIC

4. **`warm_start` 用法**：设 `warm_start=True` 后，多次调用 `fit()` 会从上次结果继续迭代。注意每次仍会迭代 `max_iter` 次。

5. **BGMM 的 `score()` 返回 ELBO**，而非严格对数似然，因此与 GMM 的 `score()` **不可直接比较**。

---

## 参考

- [scikit-learn: GaussianMixture 文档](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
- [scikit-learn: BayesianGaussianMixture 文档](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html)
- Reynolds, R. P. (2009). *Gaussian Mixture Models*. Encyclopedia of Biometrics.
