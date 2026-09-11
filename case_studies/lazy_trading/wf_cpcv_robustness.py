# -*- coding: utf-8 -*-
"""WF+CPCV 嵌套流程的后验稳健性检验模块（checklist 4.1/4.2/5.1/5.3 增量件）。

定位：在 nested_adaptive_search 已产出的 fold_results（每折自适应最优参数）
之上做三类**固定参数评估**——只评估、不搜索、不选参，不构成 OOS 消耗
（"子弹=选择，评估不费子弹"；任何基于评估结果回改策略的动作都会作废
该段 OOS，使用时请自行遵守该封印纪律）：

    1. 参数随机扰动 MC（5.1）：以每折生产参数 top1 为锚，数值参数在其
       搜索空间内 ±max(frac×(high−low), step) 均匀扰动（半径取与 step
       的 max，保证大步长参数至少跨 ±1 格点，按 step 圆整、边界钳制），
       fitness 在每折 IS 去重 top-k 名称内随机切换；逐折独立生成 R 组
       扰动组合，每组经 run_params_on_fold（普通 OOS 应用，无内层 CPCV
       展开，与部署语义一致）评估该折真实 test 段绩效。指标全取 skfolio
       现成属性，不自算。
    2. 单参数敏感性曲线（4.1）：每折对单个数值参数做 ±10~30% 定点偏移
       网格（fitness 固定该折 top1），其余参数保持生产值 → 参数高原
       （平台期）vs 参数孤峰（尖点）的绩效形状判读。
    3. 双参数联合热力图（4.2）：两两参数 ±20% 网格 → 每折绩效网格 +
       跨折平均帧，观察参数间交互（互补/冲突/独立）。

统计口径（重要）：
    同折扰动样本共享同一条真实 OOS 收益段 → 样本**非独立**（共享市场
    路径），正收益概率/分位数是"参数族评估"语义而非独立重复试验——
    判读时以全折池化趋势为准，勿把非独立样本当独立试验计数。

CLI（冒烟/小预算验证用；完整检验请在 notebook 会话内复用已完成的搜索）::

    G:\\Anaconda3\\envs\\ml4t\\python.exe wf_cpcv_robustness.py --n-trials 3 --tail-days 1200

notebook 用法（fold_results 来自 nested_adaptive_search）::

    from wf_cpcv_robustness import perturbation_mc, sensitivity_curves, topk_paths_eval
    out = perturbation_mc(X, fold_results, R=50)        # dict: samples/fold_summary/verdict
    print(out["verdict"])                                # 5.3 判定表（正收益概率≥80%等）
    curves = sensitivity_curves(X, fold_results)         # 4.1 行=(折,参数,档位)
    eval_ = topk_paths_eval(X, fold_results, k=5)        # 每 rank 一条完整OOS路径（全折test段拼接）
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd
from skfolio import MultiPeriodPortfolio

from cpcv_analysis import top_trials
from wf_cpcv_search import (
    derive_outer_window,
    load_nested_config,
    load_space,
    nested_adaptive_search,
    run_params_on_fold,
    summarize_top_k_params,
)

# Windows GBK 控制台打印中文安全兜底（与同目录脚本一致）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

DEFAULT_METRICS = ["annualized_mean", "annualized_sharpe_ratio",
                   "max_drawdown", "skew", "kurtosis"]

# checklist 5.3 判据（扰动样本池化口径）
JUDGE = {"pos_prob_min": 0.80, "median_min": 0.0, "p95_min": 0.0}


# ---------------------------------------------------------------------------
# 基础件：扰动几何（按 space 节点驱动）/ fitness 池 / 折位对齐 / 单次评估
# ---------------------------------------------------------------------------
def perturb_params(params, space, rng, frac=0.2, fitness_pool=None,
                   exclude=()):
    """单组参数数值邻域扰动：数值键 ±frac×(high−low) 均匀扰动。

    space : 参数空间（range 节点 {low, high, step}；choice 节点保持不动）。
    rng : np.random.Generator（逐折独立流，由调用方按折分叉）。
    frac : 相对扰动幅度。扰动半径 D = max(frac×(high−low), step)：前者
        是连续参数邻域的常规扰动；但 step 大的离散参数（如 train_size
        step=126）在 ±frac×span < step/2 时按 step 圆整会全部落回原值，
        扰动退化为恒等——取与 step 的 max 保证每个参数至少可能跨 ±1 个
        合法格点（扰动无效比不扰动更危险，静默的恒等样本会稀释分布）。
    圆整与钳制：扰动值先按 step 圆整、再钳制回 [low, high]；int 型节点
        （suggest_int 采样）cast 回 int —— 保证扰动点仍是合法采样点。
    覆盖范围：space 中全部数值节点一视同仁，含 [nested_space] 的
        train_size（它与其他参数一样是每折自适应搜索结果，扰动后评估
        即用扰动后的拟合窗口长度，检验"窗口长度选择"本身的稳健性）；
        如需刻意跳过某键（如只做管线参数扰动）再传 exclude。
    fitness_pool : 每折去重 top-k fitness 名称列表；非空时扰动样本的
        fitness 在池内随机切换（选参时 fitness 也是自适应维度之一，5.1
        需覆盖它——见 fitness_pool_from_study）。None 则保持生产值。
    exclude : tuple，不扰动的数值键（默认不排除任何键）。
    """
    out = dict(params)
    for key, node in space.items():
        if not isinstance(node, dict) or "low" not in node:
            continue                      # choice 节点：扰动只作用于数值键
        if key in exclude:
            continue
        lo, hi = float(node["low"]), float(node["high"])
        step = float(node.get("step", 0) or 0)
        radius = max(frac * (hi - lo), step)   # ≥ 1 step：大步长参数扰动不致恒等
        v = float(params[key]) + rng.uniform(-radius, radius)
        if step > 0:
            v = step * round(v / step)
        v = min(max(v, lo), hi)
        out[key] = int(v) if isinstance(params[key], int) else v
    if fitness_pool:
        out["nondomin__fitness_measures"] = rng.choice(list(fitness_pool))
    return out


def fitness_pool_from_study(study, k=3):
    """每折 study 内按 IS 分数去重取前 k 个 fitness 名称（扰动切换池）。

    fitness 是搜索空间 choice 维度：最优 fitness 与其他参数一样带选择
    偏差。扰动时在"该折 IS 上表现最好的 k 个 fitness"内切换，即只扰动
    fitness 的排序不确定性、不采样整空间的低分 fitness（后者会让扰动
    分布混入明显劣质的策略族，稀释稳健性信号）。
    """
    valid = [t for t in study.trials
             if t.values is not None and np.isfinite(t.values[0])]
    valid.sort(key=lambda t: float(t.values[0]), reverse=True)
    seen, out = set(), []
    for t in valid:
        fm = t.params["nondomin__fitness_measures"]
        if fm in seen:
            continue
        seen.add(fm)
        out.append(fm)
        if len(out) >= k:
            break
    return out


def derive_wf_kwargs(folds, purged_size=1, reduce_test=True):
    """从 fold_results 推导 run_params_on_fold 折位参数（扰动/敏感性用）。

    折位对齐要求：扰动评估必须与 nested_adaptive_search 使用相同的
    test_size / train_size / purged_size / reduce_test，否则 split 错位、
    评估段不再是该折真实 OOS。窗口两键取自公共推导 derive_outer_window
    （fold 内 train/test Portfolio 实际段长度众数）；purged_size /
    reduce_test 按嵌套搜索默认 1 / True（搜索侧 reduce_test=True 时含
    尾折，扰动必须同样产出尾折才能逐折对齐），调用方须与搜索配置核对。
    """
    w = derive_outer_window(folds, purged_size=purged_size,
                            reduce_test=reduce_test)
    return {"test_size": w["test_size"], "train_size": w["train_size"],
            "purged_size": w["outer_purged_size"],
            "reduce_test": w["outer_reduce_test"]}


def _eval_params_on_fold(X, fold, params, metrics, wf_kwargs):
    """固定参数在该折的普通 OOS 单测：fit 最近 train_size 天 → 预测纯净
    test 段（run_params_on_fold，无内层 CPCV 展开），返回 skfolio 属性。
    """
    _, te_ptf = run_params_on_fold(X, fold["fold"], params, **wf_kwargs)
    return {m: getattr(te_ptf, m) for m in metrics}


def _space_range_keys(space, exclude=()):
    """space 中的数值（range 节点）参数键，剔除 exclude（默认不剔除）。"""
    return [k for k, node in space.items()
            if isinstance(node, dict) and "low" in node and k not in exclude]


# ---------------------------------------------------------------------------
# 5.1 参数随机扰动 MC（每折独立扰动 + 普通 OOS 单测）
# ---------------------------------------------------------------------------
def perturbation_mc(X, folds, R=None, frac=0.2, space=None, fitness_topk=3,
                    exclude=(), metrics=DEFAULT_METRICS,
                    wf_kwargs=None, purged_size=1, reduce_test=True,
                    keep_params=True, seed=42, verbose=True):
    """参数随机扰动蒙特卡洛（checklist 5.1）：每折独立扰动 R 组 → 逐组
    普通 OOS 单测 → 扰动绩效分布与 5.3 判定。

    Parameters
    ----------
    X : pd.DataFrame，与 nested_adaptive_search 同口径收益数据
    folds : list，nested_adaptive_search 返回结果（每折 top1 参数为扰动锚）
    R : int | None，每折扰动组合数；None = ceil(500 / n_folds)（checklist
        5.1 总样本 ≥500 的折算；扰动评估成本 ≈ R×折数 次轻量单折应用，
        远低于 CPCV 多路径压测，可放宽预算）
    frac : float，扰动幅度 = ±max(frac×(high−low), step)（见 perturb_params：
        与 step 取 max 保证大步长参数扰动至少跨 ±1 合法格点）
    space : dict | None，数值键扰动范围（None = 载入默认配置空间）
    fitness_topk : int，每折 fitness 切换池深度（study 内去重 top-k）
    exclude : tuple，不扰动的数值键（默认不排除——含 train_size 在内的
        全部自适应参数都参与扰动，见 perturb_params）
    wf_kwargs : dict | None，run_params_on_fold 的折位参数（None = 从
        folds 推导众数窗口，purged_size/reduce_test 用下方显式参数）
    keep_params : bool，样本表是否保留扰动后参数列（审计扰动实际值）
    seed : int，随机种子（逐折分叉流：每折 rng = default_rng(seed+fold)，
        保证逐折独立且整体可复现）

    Returns
    -------
    dict :
        samples     : DataFrame，行 MultiIndex (fold, sample)；列 = metrics
                      （+ keep_params 时扰动后各参数值）
        fold_summary: DataFrame，行 = fold；列 = 每指标 mean/median/p5/p95
                      + pos_prob（年化收益 >0 占比）与 n 样本数
        verdict     : DataFrame，5.3 判定表：扰动样本池化的 pos_prob /
                      median / p95 vs 判据（≥80% / >0 / >0），行 = 指标
        space/folds : 快照（扰动锚与范围留痕）
    """
    space = space if space is not None else load_space()
    if wf_kwargs is None:
        wf_kwargs = derive_wf_kwargs(folds, purged_size=purged_size,
                                     reduce_test=reduce_test)
    n_folds = len(folds)
    R = R if R is not None else max(10, math.ceil(500 / n_folds))
    pkeys = _space_range_keys(space, exclude)

    rows, idx = [], []
    for f in folds:
        fold_idx = f["fold"]
        rng = np.random.default_rng(seed + fold_idx)      # 逐折独立流
        pool = fitness_pool_from_study(f["study"], k=fitness_topk)
        for s in range(R):
            p = perturb_params(f["params"], space, rng, frac=frac,
                               fitness_pool=pool, exclude=exclude)
            met = _eval_params_on_fold(X, f, p, metrics, wf_kwargs)
            if keep_params:
                met = {**met,
                       **{k: p.get(k) for k in pkeys},
                       "fitness": p["nondomin__fitness_measures"]}
            rows.append(met)
            idx.append((fold_idx, s))
        if verbose:
            print(f"Fold {fold_idx}: R={R} | fitness池={pool} | "
                  f"扰动脉冲完成（{R} 组 × 普通OOS单测）")
    samples = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
        idx, names=["fold", "sample"]))

    frows = []
    for fold_idx, g in samples.groupby(level=0):
        g2 = g[metrics]
        s = {"fold": fold_idx, "n": len(g2)}
        for m in metrics:
            s[f"{m}_mean"] = g2[m].mean()
            s[f"{m}_median"] = g2[m].median()
            s[f"{m}_p5"] = g2[m].quantile(0.05)
            s[f"{m}_p95"] = g2[m].quantile(0.95)
        s["pos_prob"] = float((g2["annualized_mean"] > 0).mean())
        frows.append(s)
    fold_summary = pd.DataFrame(frows).set_index("fold")

    verdict = _verdict_rows(samples[metrics], ["annualized_mean",
                                               "annualized_sharpe_ratio"])
    return {"samples": samples, "fold_summary": fold_summary,
            "verdict": verdict, "space": space, "folds": folds}


def _verdict_rows(samples_df, metrics):
    """checklist 5.3 判定表：样本池化 pos_prob/median/p95 vs 判据。"""
    rows = []
    for m in metrics:
        v = samples_df[m].dropna()
        pos = float((v > 0).mean())
        med, p95 = float(v.median()), float(v.quantile(0.95))
        ok = (pos >= JUDGE["pos_prob_min"] and med > JUDGE["median_min"]
              and p95 > JUDGE["p95_min"])
        rows.append({"指标": m, "样本数": len(v), "prob>0": pos,
                     "median": med, "p95": p95, "通过": ok})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4.1 单参数敏感性曲线 / 4.2 双参数热力图（fitness 固定每折 top1）
# ---------------------------------------------------------------------------
def _shift_grid_value(base, frac, node):
    """生产值 ×(1+frac) → step 圆整 → 边界钳制（4.1/4.2 定点偏移网格）。"""
    lo, hi = float(node["low"]), float(node["high"])
    step = float(node.get("step", 0) or 0)
    v = float(base) * (1.0 + frac)
    if step > 0:
        v = step * round(v / step)
    v = min(max(v, lo), hi)
    return int(v) if isinstance(base, int) else v


def sensitivity_curves(X, folds, params=None, fracs=(-0.3, -0.2, -0.1, 0.1, 0.2, 0.3),
                       space=None, metrics=DEFAULT_METRICS,
                       wf_kwargs=None, purged_size=1, reduce_test=True,
                       verbose=True):
    """单参数敏感性曲线（checklist 4.1：±10~30% 定点偏移）。

    每折只动目标参数、其余参数保持生产值（fitness 亦固定 top1——单参
    敏感性回答"该参数取值微调后绩效形状"，不引入 fitness 切换噪声）。
    档位 frac=0 即生产值本身（曲线锚点）。输出宽表供 notebook 画
    (折 × 参数) 折线：参数高原 = 平台期（±30% 内绩效平缓），孤峰 = 仅
    生产值附近突出——与 summarize_top_k_params 的 IS gap 落差互证。

    Returns
    -------
    pd.DataFrame : 行 = (fold, 参数, 档位)；列 = frac | value（实际偏移后
        参数值）| metrics（该档位的普通 OOS 单测绩效）
    """
    space = space if space is not None else load_space()
    if wf_kwargs is None:
        wf_kwargs = derive_wf_kwargs(folds, purged_size=purged_size,
                                     reduce_test=reduce_test)
    params = params if params is not None else _space_range_keys(space)
    rows = []
    for f in folds:
        for pk in params:
            node = space[pk]
            for frac in fracs:
                p = dict(f["params"])
                p[pk] = _shift_grid_value(p[pk], frac, node)
                met = _eval_params_on_fold(X, f, p, metrics, wf_kwargs)
                rows.append({"fold": f["fold"], "param": pk, "frac": frac,
                             "value": p[pk], **met})
        if verbose:
            print(f"Fold {f['fold']}: 单参网格完成 "
                  f"({len(params)} 参数 × {len(fracs)} 档)")
    return pd.DataFrame(rows)


def sensitivity_heatmap(X, folds, param_a, param_b,
                        fracs_a=(-0.2, -0.1, 0.0, 0.1, 0.2),
                        fracs_b=(-0.2, -0.1, 0.0, 0.1, 0.2),
                        metric="annualized_sharpe_ratio",
                        space=None, wf_kwargs=None, purged_size=1,
                        reduce_test=True, verbose=True):
    """双参数联合热力图（checklist 4.2）：a×b 档位网格的绩效响应。

    每折以生产值为中心展开 (frac_a, frac_b) 笛卡尔网格（其余参数含
    fitness 固定），评估后返回逐折网格与跨折平均帧。读图要点：
    a/b 两参数各档绩效均平缓 → 联合高原（稳健）；仅 (0,0) 突出 →
    联合孤峰（脆弱）；对角方向陡峭 → 参数互补/冲突（交互项存在）。

    Returns
    -------
    dict : {fold: DataFrame(index=frac_a, columns=frac_b, values=metric)},
        另含 "mean" 键 = 各折网格的逐格平均（池化趋势帧）
    """
    space = space if space is not None else load_space()
    if wf_kwargs is None:
        wf_kwargs = derive_wf_kwargs(folds, purged_size=purged_size,
                                     reduce_test=reduce_test)
    grids = {}
    for f in folds:
        na, nb = space[param_a], space[param_b]
        grid = np.empty((len(fracs_a), len(fracs_b)))
        for i, fa in enumerate(fracs_a):
            for j, fb in enumerate(fracs_b):
                p = dict(f["params"])
                p[param_a] = _shift_grid_value(p[param_a], fa, na)
                p[param_b] = _shift_grid_value(p[param_b], fb, nb)
                met = _eval_params_on_fold(X, f, p, [metric], wf_kwargs)
                grid[i, j] = met[metric]
        grids[f["fold"]] = pd.DataFrame(grid, index=fracs_a, columns=fracs_b)
        if verbose:
            print(f"Fold {f['fold']}: {param_a}×{param_b} 热力图 "
                  f"({len(fracs_a)}×{len(fracs_b)}) 完成")
    grids["mean"] = pd.concat(list(grids.values())).groupby(level=0).mean()
    return grids


# ---------------------------------------------------------------------------
# Top-K 候选完整 OOS 路径对比（每 rank = 一条全折拼接路径；只评估、不选参）
# ---------------------------------------------------------------------------
def topk_paths_eval(X, folds, k=5,
                    metrics=("annualized_sharpe_ratio", "annualized_mean",
                             "max_drawdown"),
                    wf_kwargs=None, purged_size=1, reduce_test=True,
                    verbose=True):
    """每折 Top-K 候选各自生成一条完整 OOS 路径后对比。

    每个 rank = 一个候选组合：每折取该折 study 第 rank 档候选参数，普通
    OOS 单测（run_params_on_fold，无内层 CPCV 展开，与部署语义一致）得到
    该折纯净 test 段，**所有折的 test 段拼接为一条 MultiPeriodPortfolio**
    —— 一条完整 OOS 路径，在完整路径上取 skfolio 指标。rank 间对比 =
    "若部署第 rank 档参数，完整 OOS 路径会是什么样"。

    不用 CPCV 多路径压测：每折内层 OOS 组数（重组路径数）随该折窗口/
    折数可变，跨折无法对齐成"同一路径 id"的完整路径；多路径分布由
    LHS 采样场景负责，此处每 rank 一条路径即可。

    只评估、不选参：即使候选对比显示 top1 并非 OOS 最优，也不得据
    此换参（换参 = 用 OOS 选择 = 消耗该段）；候选对比的价值在"参数
    高原的 OOS 侧印证"——gap2top1 小的候选若 OOS 同样接近，说明生产
    参数处在一个稳定平台而非运气孤峰。

    wf_kwargs 未给时从 folds 推导折位参数（与嵌套搜索 1:1 对齐，防静默
    错位）；purged_size / reduce_test 默认 1 / True（与搜索侧一致，含
    缩短尾折），须与搜索配置核对。某折候选不足该 rank 档时整档跳过
    （保证各 rank 均为全折完整路径，长度可比）。

    Returns
    -------
    dict :
        paths : DataFrame，行 = rank；列 = metrics——每 rank 一条完整
            OOS 路径的绩效（rank=1 即生产参数）
        mpts  : dict {rank: MultiPeriodPortfolio}，完整 OOS 路径本体
            （供画累计收益 / 分年统计等）
        audit : DataFrame，summarize_top_k_params 的 IS 侧审计（score /
            gap2top1 / 参数快照），与 paths 按 rank 对齐
    """
    if wf_kwargs is None:
        wf_kwargs = derive_wf_kwargs(folds, purged_size=purged_size,
                                     reduce_test=reduce_test)
    cands_by_fold = {f["fold"]: top_trials(f["study"].trials, k=k) for f in folds}
    rows, labels, mpts = [], [], {}
    for rank in range(1, k + 1):
        missing = [f["fold"] for f in folds
                   if len(cands_by_fold[f["fold"]]) < rank]
        if missing:
            if verbose:
                print(f"rank{rank}: 折 {missing} 候选不足，整档跳过")
            continue
        # 每折该 rank 档候选的纯净 test 段（部署语义：fit 训练尾段 → 预测 test）
        parts = [run_params_on_fold(X, f["fold"],
                                    cands_by_fold[f["fold"]][rank - 1].params,
                                    **wf_kwargs)[1]
                 for f in folds]
        mpp = MultiPeriodPortfolio(parts)          # 全折 test 段拼接 = 完整 OOS 路径
        mpts[rank] = mpp
        rows.append([getattr(mpp, met) for met in metrics])
        labels.append(rank)
        if verbose:
            print(f"rank{rank}: 完整OOS路径 | {len(parts)} 折拼接 | "
                  f"{sum(len(p.returns) for p in parts)} 天")
    paths = pd.DataFrame(rows, index=pd.Index(labels, name="rank"),
                         columns=list(metrics))
    return {"paths": paths, "mpts": mpts,
            "audit": summarize_top_k_params(folds, k=k)}


# ---------------------------------------------------------------------------
# CLI（冒烟/小预算；完整检验请在 notebook 会话复用已完成的 fold_results）
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="WF+CPCV 参数扰动/敏感性稳健性冒烟 CLI")
    parser.add_argument("--n-trials", type=int, default=3,
                        help="每折内层 optuna trial 数（小预算冒烟）")
    parser.add_argument("--tail-days", type=int, default=None,
                        help="只取数据尾部 N 天控制折数（None=全量）")
    parser.add_argument("--r-per-fold", type=int, default=5,
                        help="每折扰动组合数（完整 5.1 建议总样本 ≥500）")
    parser.add_argument("--frac", type=float, default=0.2, help="扰动幅度 ±frac×(high-low)")
    parser.add_argument("--data", choices=("X", "X_net"), default="X",
                        help="数据口径（默认 X，与 WF+CPCV+PS_252 一致）")
    parser.add_argument("--n-jobs", type=int, default=2, help="trial 级并行度")
    parser.add_argument("--cv-n-jobs", type=int, default=1, help="单 trial 内 CPCV 并行度")
    args = parser.parse_args(argv)

    import walkforward_parameter_search as wfps
    from log_result import init_logger, log_print, log_result

    init_logger("wf_cpcv_robustness_smoke")
    X_all, X_net, info = wfps.load_data()
    X_use = X_net if args.data == "X_net" else X_all
    if args.tail_days:
        X_use = X_use.tail(args.tail_days)
    log_print(f"数据: X | 行数={len(X_use)} | inf={info['inf_cols']}",
              section="数据加载", echo=True)

    cfg = load_nested_config()
    search_kwargs = {**cfg["search_kwargs"], "n_trials": args.n_trials,
                     "n_jobs": args.n_jobs, "cv_n_jobs": args.cv_n_jobs}
    log_print(search_kwargs, section="搜索配置（冒烟）", echo=False)
    folds = nested_adaptive_search(X_use, space=cfg["space"], **search_kwargs)

    out = perturbation_mc(X_use, folds, R=args.r_per_fold, frac=args.frac)
    print("\n==== 5.3 判定表（参数扰动 MC）====")
    print(out["verdict"].to_string(index=False))
    log_result(out["verdict"], section="参数扰动MC判定（checklist 5.3）")


if __name__ == "__main__":
    main()
