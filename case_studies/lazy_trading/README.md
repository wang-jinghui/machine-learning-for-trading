# lazy_trading 参数搜索完成后的检验手册

> 适用架构：外层 WalkForward × 内层 CPCV 逐折 Optuna 自适应搜索（`nested_adaptive_search`）。
> 本文回答一个问题：**参数搜索跑完后，接下来按什么顺序、调哪些函数、看什么指标做全套后验检验**。
> 各检验与 [策略研发全流程Checklist.md](./策略研发全流程Checklist.md) 的章节逐条对应。

---

## 0. 铁律（先读）

1. **OOS 封印**：`nested_adaptive_search` 每折 test 段是绝对样本外，一旦据检验结果**回改策略再重搜**，该段 OOS 永久作废。
2. **评估不费子弹**：下面所有检验都是**固定参数评估**——只评估、不搜索、不选参，不消耗 OOS。
   即使 Top-K 压测显示 top1 不是 OOS 最优，**也不得据 OOS 换参**（换参 = 用 OOS 选择 = 消耗该段）。
3. **折位对齐**：检验函数内部都通过 `derive_outer_window(folds)` 从搜索结果自动还原外层窗口
   （test_size / train_size / purged_size / reduce_test），消费侧与搜索侧折位 1:1 对齐。
   自行调用 `adaptive_multi_paths` 时请务必对齐窗口，否则 split 会**静默错位**（比报错更危险）。
4. **统计口径**：同折扰动样本共享同一条真实 OOS 收益段 → 非独立样本，"参数族评估"语义，
   判读以全折池化趋势为准，勿当独立试验计数。

## 1. 一次性骨架（notebook 会话开头）

```python
import sys; sys.path.insert(0, r"case_studies\lazy_trading")
import walkforward_parameter_search as wfps
from wf_cpcv_search import (
    load_nested_config, nested_adaptive_search, summarize_fold_params, summarize_top_k_params,
)
from wf_cpcv_robustness import (
    perturbation_mc, sensitivity_curves, sensitivity_heatmap, topk_paths_eval,
)
from wf_cpcv_ablation import ablate

X_all, X_net, info = wfps.load_data()
X = X_all                     # 252 实验口径 = X（含基准列同源）；去中性化口径用 X_net
cfg = load_nested_config()    # 空间 + 搜索/压测窗口 + Purge/Embargo + 采样器 + seed 全部从 toml 来

folds = nested_adaptive_search(X, space=cfg["space"], **cfg["search_kwargs"])
# folds: list[{fold, params, score, test, train, "train ASR", "test ASR",
#             p_luck, margin, max_p95, n_trials, study}]
```

> 若搜索已完成（实验日志/notebook 缓存中），**直接复用 fold_results 对象**，跳过重跑——
> 下面的全部检验只吃 `folds`，不再触碰内层搜索。

## 2. 检验路线图（建议顺序）

| 步 | 检验 | Checklist | 函数 | 产出 | 终止条件 |
|---|---|---|---|---|---|
| 1 | IS 选择偏差与 Top-K 审计 | 2.1 / 2.3 | `summarize_fold_params` + `summarize_top_k_params` | 折级 p_luck + 候选表 | p_luck 普遍高 + gap 大 → 回调参 |
| 2 | 过拟合判据（p_luck vs OOS 对照） | 2.1 / 3.2 | fold 内置字段 | 折级表 | "p_luck 高 + test ASR 显著掉" |
| 3 | Top-K 候选 OOS 多路径压测 | 3.1 | `topk_paths_eval` | paths / audit / frames | rank1 与 2~k 差距悬殊 → 警惕孤峰 |
| 4 | 单参数敏感性曲线 | 4.1 | `sensitivity_curves` | 宽表（画折线） | 孤峰 → 回调参；高原 → 通过 |
| 5 | 双参数热力图 | 4.2 | `sensitivity_heatmap` | 逐折网格 + mean 帧 | 联合孤峰 → 回调参 |
| 6 | 消融实验 | 4.3 | `ablate` | frames + summary | 某模块负贡献 → 考虑剔除重搜 |
| 7 | 参数扰动蒙特卡洛 | 5.1 / 5.3 | `perturbation_mc` | samples + verdict | verdict 三条件全过 → 放行 |

---

## 3. 各步操作手册

### Step 1｜IS 审计：每折 p_luck + Top-K 候选（checklist 2.1 / 2.3）

```python
fold_df = summarize_fold_params(folds)            # 行 = fold
print(fold_df[["fold", "score", "p_luck", "margin",
               "n_trials", "train ASR", "test ASR"]])

topk = summarize_top_k_params(folds, k=5)         # 行 = (fold, rank)
print(topk[topk["rank"] <= 2])                    # 每折 top1/top2 候选 + gap2top1
```

判读：
- `p_luck` = 该折最优分数是"纯运气挑出"的经验概率（经验零分布 GPD 上尾 deflate，N = 参数去重后的真实试验数）；越高越可疑。
  `margin = score − E[max_N]`，`max_p95` 为运气基准 95% 分位。
- `gap2top1`（summarize_top_k_params 的列）= IS 侧 top1 与 top2 的分数落差：
  落差小 → 参数处在平台（好信号）；落差大 → top1 可能是运气孤峰（坏信号）。
- 目标函数 = mean path 年化夏普（`inner_cpcv_score`，单值，Optuna 约束）。

### Step 2｜过拟合判据：p_luck 高 + OOS 衰减（checklist 3.2）

```python
df = summarize_fold_params(folds).copy()
df["WFE"] = df["test ASR"] / df["train ASR"]      # 每折 IS→OOS 衰减比
df["flag"] = (df["p_luck"] > 0.5) & (df["WFE"] < 0.5)   # 运气嫌疑高 + 衰减 = 过拟合嫌疑折
print(df[["fold", "p_luck", "train ASR", "test ASR", "WFE", "flag"]])
```

- Checklist 3.2 判据：WFE > 0.5 为稳健泛化（多数折满足）。
- "p_luck 高 + OOS 显著下降" = 内层选择偏差校正后无真实优势 + 样本外掉队 → 该折结论不可信，
  属**策略逻辑问题而非参数问题**，回改须谨慎（见铁律 1）。

### Step 3｜Top-K 候选 OOS 多路径压测（checklist 3.1）

```python
eval_ = topk_paths_eval(X, folds, k=5)            # 每折 top1~5 全部进 CPCV 多路径压测
paths = eval_["paths"]                            # MultiIndex (fold, rank, path)
audit = eval_["audit"]                            # IS 侧审计，与 paths 按 (fold, rank) 对齐

# 方案 A：unstack 后 rank 已转入列方向（列=rank 1~5，行=(fold, path)），
#         直接对列求均值即各 rank 的跨路径对比（勿再按列名 "rank" groupby，会 KeyError）
sharpe = paths["annualized_sharpe_ratio"].unstack("rank")
print(sharpe.mean())                              # 每列均值 = 该 rank 的平均年化夏普

# 方案 B（等价，语义更直白）：不 unstack，直接在 Series 上按索引层级 rank 分组
print(paths["annualized_sharpe_ratio"].groupby("rank").mean())
```

- 窗口自动从 folds 推导（折位 1:1）；如需 notebook 原压测口径（如 `n_test_folds=4`），
  以显式键传入：`topk_paths_eval(X, folds, paths_kwargs={"n_test_folds": 4})`。
- 价值在**参数高原的 OOS 侧印证**：gap2top1 小的候选若 OOS 同样接近，说明生产参数处在稳定平台。

### Step 4｜单参数敏感性曲线（checklist 4.1）

```python
curves = sensitivity_curves(X, folds)             # 默认 ±10/20/30% 六档 × 全部数值参数
c = curves[curves["param"] == "extremes__k"]
ax = c.pivot_table(index="frac", columns="fold",
                   values="annualized_sharpe_ratio").plot(marker="o")
```

- 每折只动目标参数、其余保持生产值（fitness 固定 top1）；`frac=0` 档即生产值本身（锚点）。
- 默认覆盖**全部数值参数含 `train_size`**（它是搜索维度之一，扰动后评估即用扰动后的拟合窗口长度）。
- 读图：参数高原 = ±30% 内绩效平缓；孤峰 = 仅生产值附近突出（与 IS 侧 gap2top1 互证）。
- 只评估单折 OOS → 无 CPCV 展开，成本低，可全参数全档位跑。

### Step 5｜双参数联合热力图（checklist 4.2）

```python
import numpy as np
import matplotlib.pyplot as plt

grids = sensitivity_heatmap(X, folds, param_a="extremes__k",
                            param_b="correlate__threshold")
mean = grids["mean"]                             # 行=frac_a(param_a), 列=frac_b(param_b)
# DataFrame.plot 无 contour 图型 → 用 matplotlib 直接画（X/Y/Z 同形对齐）
Xg, Yg = np.meshgrid(mean.columns, mean.index)   # x=frac_b, y=frac_a
cf = plt.contourf(Xg, Yg, mean.values, levels=15, cmap="viridis")   # 彩色填充
cs = plt.contour(Xg, Yg, mean.values, levels=8, colors="k", linewidths=0.6)  # 等值线
plt.clabel(cs, inline=True, fontsize=8)          # 标注等值线数值
plt.axhline(0, color="gray", lw=0.8, ls="--")   # 虚线十字 = 生产值锚点 (0,0)
plt.axvline(0, color="gray", lw=0.8, ls="--")
plt.xlabel("correlate__threshold 扰动 frac_b")
plt.ylabel("extremes__k 扰动 frac_a")
plt.colorbar(cf, label="mean annualized_sharpe_ratio")
```

- 返回 `{fold: DataFrame(index=frac_a, columns=frac_b), "mean": 跨折逐格平均}`。
- 读图：各档平缓 → 联合高原（稳健）；仅 (0,0) 突出 → 联合孤峰（脆弱）；
  对角陡峭 → 两参数互补/冲突（交互项存在）。

### Step 6｜消融实验（checklist 4.3）

```python
frames, summary = ablate(X, folds)                # arms=None = 全部 5 臂
print(summary[["说明", "annualized_mean_mean", "annualized_sharpe_ratio_mean",
               "max_drawdown_mean", "样本数"]])
```

- 臂：`full`（基准）/ `no_extremes` / `no_nondomin` / `no_correlate` / `no_extremes_correlate`。
- 每臂 = 固定每折最优参数 × 剔除对应管线步骤 × CPCV 多路径压测（同窗同口径对比）。
- 判读：全臂样本路径数一致（每折 = `n_test_folds` 组合路径），绩效差距即模块边际贡献；
  某模块剔除后绩效**不降反升** → 该模块负贡献，考虑剔除后重搜（会作废旧 OOS）。

### Step 7｜参数扰动蒙特卡洛 + 判定（checklist 5.1 / 5.3）

```python
out = perturbation_mc(X, folds)                   # R=None → ceil(500 / 折数)，总样本 ≥ 500
print(out["verdict"])                             # 判定表：prob>0 / median / p95 / 通过
print(out["fold_summary"])                        # 逐折分布
out["samples"]                                    # 行 (fold, sample)；含扰动后参数列（审计用）
```

- 扰动几何：每折锚 = 该折生产参数 top1；数值参数在 `±max(frac×(high−low), step)` 内均匀扰动
  （半径取与 step 的 max → 保证大步长参数至少跨 ±1 合法格点，防恒等退化）→ step 圆整 → 边界钳制；
  fitness 在该折 IS 去重 top-3 名内随机切换（扰动"选 fitness"的排序不确定性）。
- 默认**不含排除项**：`train_size` 与其他数值参数一样参与扰动（它也是搜索维度）。
- 逐折独立随机流（`seed + fold`），整体可复现。
- 评估走 `run_params_on_fold`（普通 OOS 应用，无 CPCV 展开，与部署语义一致）。

**5.3 判定规则**（`verdict` 自动给出，按 `JUDGE` 判据）：

| 条件 | 判据 | 说明 |
|---|---|---|
| 正收益概率 | ≥ 80% | 扰动样本池化 `prob > 0` |
| 收益中位数 | > 0 | 多数扰动组合不亏 |
| 95% 分位收益 | > 0 | 极端扰动组合也不亏 |
| 三者全过 | → 放行 | 进入实盘模拟准备；任一不过 → 返回内层调参或审视策略逻辑 |

---

## 4. 判读速查

| 信号 | 判读 | 动作 |
|---|---|---|
| `p_luck` 低（≤ ~0.1） | IS 最优不是运气 | 可信 |
| `p_luck` 高（> 0.5） | 最优可能纯运气 | 看 margin / 回调参 |
| `WFE = test/IS ASR > 0.5` | 泛化正常 | 可信 |
| `WFE < 0.5` 多数折 | OOS 显著衰减 | 策略逻辑审计 |
| `gap2top1` 小 | 候选平台宽 | 好（高原） |
| `gap2top1` 大 | top1 孤立 | 警惕孤峰 |
| 敏感性曲线孤峰 / 热力图仅 (0,0) 亮 | 参数尖点 | 回调参（平台区域） |
| 扰动 `verdict` 全过 | 参数族稳健 | 放行 |
| 消融某臂不降反升 | 模块负贡献 | 考虑剔除重搜 |

## 5. 已知缺口（checklist 中未工程化、交付前需另行实现/人工完成）

- 5.2 时间延迟扰动：低频策略有意排除。
- 6.2 交易成本/滑点、容量评估、极端行情压力测试。
- 6.3 真 PBO（<0.3）：现有 IS 侧经验零分布 deflate（`empirical_deflate`）与 OOS 侧 PSR/LHS 诊断（`cpcv_analysis.diagnose_oos_lhs`）非真 PBO，需 CSCV 另算。
- 6.3 参数高原"通过"判定：由 2.3 + 3.1 + 4.1/4.2 交叉印证后人工综合。

## 6. 模块速查

| 文件 | 内容 |
|---|---|
| `cpcv_search_base.py` | 共享底座：FITNESS_MEASURES / build_pipeline / suggest_from_space / 早停（fork 自 walkforward，两侧独立演进，改动需手动同步） |
| `wf_cpcv_search.py` | 嵌套搜索主模块：配置加载、`inner_cpcv_score`、`search_inner_params`、`nested_adaptive_search`、`run_params_on_fold`、`summarize_fold_params`、`summarize_top_k_params`、`adaptive_multi_paths`、`derive_outer_window` |
| `wf_cpcv_robustness.py` | 后验稳健性：`perturbation_mc`（5.1/5.3）、`sensitivity_curves`（4.1）、`sensitivity_heatmap`（4.2）、`topk_paths_eval`（3.1）；附 CLI（冒烟用） |
| `wf_cpcv_ablation.py` | 固定参数消融（4.3）：`ablate` / `arm_fold_paths` / `build_variant_pipeline`；附 CLI |
| `cpcv_analysis.py` | 分析件：`empirical_deflate`（复合目标经验零分布 deflate）/ `efron_null_fdr`（per-trial p/local fdr）、`calc_dsr`（旧 SR 类口径）、`top_trials`/`trial_param_key`（Top-K 去重口径）、`fold_paths_frame`/`extract_metrics`（skfolio 属性抽取）、`turnover_series`（换手）、LHS/PSR 诊断 |
| `cpcv_parameter_search_config.toml` | 全部配置：`[param_space]`+`[nested_space]`（train_size）、`[cpcv]`（窗口、n_test_folds、Purge/Embargo、n_trials、sampler、seed） |

冒烟 CLI（小预算自检，完整检验请在 notebook 会话内复用已完成的搜索）：

```bat
G:\Anaconda3\envs\ml4t\python.exe wf_cpcv_robustness.py --n-trials 3 --tail-days 1500
G:\Anaconda3\envs\ml4t\python.exe wf_cpcv_ablation.py --n-trials 3 --tail-days 1500 --arms full no_correlate
```
