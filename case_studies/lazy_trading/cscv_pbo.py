# -*- coding: utf-8 -*-
"""标准 CSCV PBO 诊断模块（单次搜索 + 静态参数族 + 对称重排）。

定位：以 Bailey et al.(2014) 的标准 CSCV 口径，检验"当前筛选流程"
（参数空间 + 选择机制）的回测过拟合风险。与逐折版 rank_pbo_logit
（cpcv_analysis，事件样本 = 折数、保留时序结构）互补：本模块样本 =
C(S, S/2)（估计稳健），但假设时间块可交换（忽略时序结构）——适用于
"参数固定不再重搜"的静态诊断场景。

流程（三段严格分离，防泄漏）：

  1) 唯一一次搜索：在搜索段（X 前 train_size 天，默认 1008 = 756+252，
     与 WF 第一折训练段同口径）上复用 search_inner_params
     —— objective / 参数空间 / purge 与 WF+CPCV 嵌套搜索完全一致
     （train_size 保留在空间内，因为要检验的就是"当前流程"；
     sampler 支持 tpe / random）。候选集 = study 去重 trial 的
     top_k / random / all 三种模式。
  2) 收益矩阵 M：每个候选参数固定后，在评估区（搜索段末尾之后，该区
     从未被搜索触碰）逐块滚动：fit 最近 train_size 天 → predict 下一块
     （block_size 天，默认 = test_size），拼接为连续收益流。全程无前视
     （fit 窗口允许回看搜索段——拟合不是选择，部署时同样如此）。
     N 条收益流构成 T x N 矩阵 M。
  3) 标准 CSCV 内核：M 等分 S 块（偶数）→ 枚举 C(S, S/2) 个对称
     IS/OOS 组合（每块既可能当 IS 也可能当 OOS）→ 每组合 IS Sharpe
     argmax 选 n* → OOS 相对排名 omega = r / (N+1) →
     lambda = ln(omega / (1-omega)) → PBO = freq(lambda <= 0)
     （IS 最优在 OOS 掉到中位数以下的组合占比；lambda>0 排名一致，
     lambda<0 过拟合）。

  注意两层"块"不要混淆：收益流生成的块（滚动 refit 步长 block_size）
  与 CSCV 的 S 等分块（重排用）是相互独立的两层。

纪律：本模块只做诊断，不做任何参数替换。PBO 高说明"在这组候选里挑
最好"这一动作不可信，正确的反应是审视流程（扩大样本/简化空间/降
自由度），而不是回头看 OOS 换参。

用法::

    from cscv_pbo import cscv_pbo
    out = cscv_pbo(X, S=16, n_trials=2000, sampler="tpe")  # 空间缺省载自 TOML
    out["pbo"], out["lambdas"]                              # 主判据与 lambda 分布
    plot_cscv(out)                                          # lambda 分布图
"""
from __future__ import annotations

import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from skfolio import MultiPeriodPortfolio
from tqdm.auto import tqdm

from cpcv_analysis import top_trials, trial_param_key
from cpcv_search_base import build_pipeline
from wf_cpcv_search import load_space, search_inner_params

# Windows GBK 控制台打印中文安全兜底（与同目录模块一致）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def search_candidates_once(X, space=None, train_size=1008, test_size=252,
                           n_test_folds=2, n_jobs=4, cv_n_jobs=10, n_trials=2000,
                           patience=100, min_delta=1e-4, sampler="tpe", seed=42,
                           inner_purged_size=2, inner_embargo_size=2, verbose=True):
    """唯一一次候选搜索：复用 WF+CPCV 内层搜索的 objective / 空间 / purge。

    搜索窗口 = X 前 train_size 天（默认 1008 = 756+252，外层搜索段
    口径，与 notebook search_params['train_size'] 同义；与 WF 第一折
    训练段一致。search_inner_params 内部再截取该段的最近 ts+test_size
    天做 CPCV，ts 为空间内自适应 train_size——本参数与候选参数自带
    的内层 train_size 属不同层次）。

    Returns
    -------
    study : optuna.Study
        含全部 trial 记录（参数、值、状态），供 pick_candidates 使用。
    """
    if space is None:
        space = load_space()
    if len(X) <= train_size:
        raise ValueError(f"search_candidates_once: 数据 {len(X)} 天不足以覆盖"
                         f"搜索段 {train_size} 天")
    X_tr = X.iloc[:train_size]
    _, _, study = search_inner_params(
        X_tr, space, test_size, n_test_folds=n_test_folds, n_jobs=n_jobs,
        cv_n_jobs=cv_n_jobs, n_trials=n_trials, patience=patience,
        min_delta=min_delta, verbose=verbose, seed=seed, sampler=sampler,
        inner_purged_size=inner_purged_size,
        inner_embargo_size=inner_embargo_size)
    return study


def pick_candidates(study, mode="top_k", k=50, seed=42):
    """从 study 生成候选参数列表（只消费搜索结果，不再重搜）。

    Parameters
    ----------
    mode : {"top_k", "random", "all"}
        top_k  : study 有效 trial 按值降序去重后的前 k 个——贴近"研究流程
                 会报告的候选族"，但候选集带搜索段预筛痕迹（PBO 会轻微
                 偏乐观，属标准 CSCV 固有语义）。
        random : 全部去重 trial 中随机抽 k 个——候选集更中性，用于对照。
        all    : 全部去重 trial（成本 = N x 收益流块数 次 fit，谨慎）。
    """
    trials = [t for t in study.trials
              if t.values is not None and np.isfinite(t.values[0])]
    if not trials:
        raise ValueError("pick_candidates: study 无有效 trial")
    if mode == "top_k":
        return [dict(t.params) for t in top_trials(trials, k=k)]

    seen, uniq = set(), []
    for t in trials:
        key = trial_param_key(t.params)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(dict(t.params))
    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(uniq), size=min(k, len(uniq)), replace=False)
        return [uniq[i] for i in sorted(idx)]
    if mode == "all":
        return uniq
    raise ValueError(f"pick_candidates: 未知 mode={mode!r}（top_k/random/all）")


def build_returns_matrix(X, params_list, start, block_size=252, verbose=True):
    """固定参数在评估区 [start, T] 上逐块滚动 fit/predict，拼接收益矩阵。

    每块与 run_params_on_fold 同构：fit 最近 train_size 天（候选自带的内层窗口）→
    predict 下一块（block_size 天）。块间不重叠、无前视；fit 窗口允许
    回看搜索段（拟合不是选择）。尾部不足一整块的残余丢弃。

    Returns
    -------
    M : pd.DataFrame
        index = 评估区日期（n_blocks x block_size 天），columns = 候选
        序号（0..N-1，与 params_list 顺序一致）。
    """
    n_blocks = (len(X) - start) // block_size
    if n_blocks < 2:
        raise ValueError(f"build_returns_matrix: 评估区不足 2 块"
                         f"（{len(X) - start} 天 / {block_size} 天 = {n_blocks}）")
    axis = X.index[start:start + n_blocks * block_size]
    if verbose:
        print(f"  收益流构建：{len(params_list)} 参数 x {n_blocks} 块 x "
              f"{block_size} 天 = {n_blocks * block_size} 天")
    cols = {}
    iterator = enumerate(params_list)
    if verbose:
        iterator = tqdm(iterator, total=len(params_list),
                        desc="收益流构建", unit="候选")
    for j, p in iterator:
        ts = int(p["train_size"])
        parts = []
        for b in range(n_blocks):
            b0 = start + b * block_size
            w = X.iloc[b0 - ts: b0]
            model = build_pipeline(p)
            model.fit(w)
            parts.append(model.predict(X.iloc[b0: b0 + block_size]))
        ret = MultiPeriodPortfolio(parts).returns_df
        if getattr(ret, "ndim", 1) > 1:
            ret = ret.iloc[:, 0]
        ret = ret.reindex(axis)
        miss = int(ret.isna().sum())
        if verbose and miss:
            print(f"  候选 {j + 1}/{len(params_list)}: 缺失 {miss} 天已置 0")
        cols[j] = ret.fillna(0.0)
    return pd.DataFrame(cols, index=axis)


def cscv_core(m, S=16):
    """标准 CSCV 内核：对称重排 -> lambda 分布 -> PBO。

    Parameters
    ----------
    m : array-like, shape (T, N)
        T x N 收益矩阵（行 = 时间，列 = 候选策略）。
    S : int
        等分块数（偶数），组合数 = C(S, S/2)。

    公式：每组合 IS Sharpe argmax 选 n*；OOS 排名 r（升序 1=最差）→
    omega = r / (N+1)；lambda = ln(omega / (1-omega))；
    PBO = freq(lambda <= 0)。Sharpe 按块拼接收益的 sum/sumsq 增量计算
    （可加性，避免逐组合重拼序列），无风险利率取 0。

    Returns
    -------
    dict
        pbo / lambdas / omega / n_star / sel_sharpe_is / sel_sharpe_oos /
        n_combos / S / block_len / T_used / N。
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
    t_use = (T // S) * S
    L = t_use // S
    blocks = m[:t_use].reshape(S, L, N)
    sum_b = blocks.sum(axis=1)                 # (S, N) 每块收益和
    sq_b = np.square(blocks).sum(axis=1)       # (S, N) 每块平方和

    cmat = np.array(list(itertools.combinations(range(S), S // 2)))
    nc = cmat.shape[0]
    n_is = cmat.shape[1] * L
    n_oos = t_use - n_is

    is_sums = sum_b[cmat].sum(axis=1)          # (nc, N)
    is_sq = sq_b[cmat].sum(axis=1)
    mean_is = is_sums / n_is
    var_is = (is_sq - n_is * mean_is ** 2) / (n_is - 1)
    std_is = np.sqrt(np.maximum(var_is, 0.0))
    sharpe_is = np.divide(mean_is, std_is, out=np.full_like(mean_is, -np.inf),
                          where=std_is > 0)
    n_star = sharpe_is.argmax(axis=1)          # (nc,) 每组合 IS 冠军

    oos_sums = sum_b.sum(axis=0) - is_sums
    oos_sq = sq_b.sum(axis=0) - is_sq
    mean_oos = oos_sums / n_oos
    var_oos = (oos_sq - n_oos * mean_oos ** 2) / (n_oos - 1)
    std_oos = np.sqrt(np.maximum(var_oos, 0.0))
    sharpe_oos = np.divide(mean_oos, std_oos, out=np.full_like(mean_oos, -np.inf),
                           where=std_oos > 0)

    ranks = rankdata(sharpe_oos, method="average", axis=1)   # 升序 1=最差
    r_sel = ranks[np.arange(nc), n_star]
    omega = r_sel / (N + 1.0)
    lambdas = np.log(omega / (1.0 - omega))
    pbo = float(np.mean(lambdas <= 0))
    sel_is = sharpe_is[np.arange(nc), n_star]
    sel_oos = sharpe_oos[np.arange(nc), n_star]

    return {"pbo": pbo, "lambdas": lambdas, "omega": omega, "n_star": n_star,
            "sel_sharpe_is": sel_is, "sel_sharpe_oos": sel_oos,
            "n_combos": int(nc), "S": int(S), "block_len": int(L),
            "T_used": int(t_use), "N": int(N)}


def cscv_pbo(X, space=None, *, train_size=1008, test_size=252, n_test_folds=2,
             n_jobs=4, cv_n_jobs=10, n_trials=2000, patience=100, min_delta=1e-4,
             sampler="tpe", seed=42, inner_purged_size=2, inner_embargo_size=2,
             candidate_mode="top_k", top_k=50, S=16, block_size=None,
             verbose=True, plot=False):
    """端到端标准 CSCV PBO 诊断（单次搜索 -> 静态收益矩阵 -> 对称重排）。

    参数与 WF+CPCV+PS notebook 当前配置对齐（train_size=1008=756+252、
    test_size=252、n_trials=2000、sampler=tpe 等）；train_size 为外层
    搜索段口径（与 search_params['train_size'] 同义），评估区自搜索段
    末尾开始（含 fold0 测试段，该段从未参与搜索）。

    Returns
    -------
    dict
        {pbo, lambdas, omega, n_star, sel_sharpe_is, sel_sharpe_oos,
         n_combos, S, block_len, T_used, N, candidates, study, matrix, meta}
    """
    if block_size is None:
        block_size = test_size
    if len(X) <= train_size + 2 * block_size:
        raise ValueError(f"cscv_pbo: 数据 {len(X)} 天不足以覆盖搜索段 "
                         f"{train_size} + 2 块 {block_size}")

    study = search_candidates_once(
        X, space=space, train_size=train_size, test_size=test_size,
        n_test_folds=n_test_folds, n_jobs=n_jobs, cv_n_jobs=cv_n_jobs,
        n_trials=n_trials, patience=patience, min_delta=min_delta,
        sampler=sampler, seed=seed, inner_purged_size=inner_purged_size,
        inner_embargo_size=inner_embargo_size, verbose=verbose)
    cands = pick_candidates(study, mode=candidate_mode, k=top_k, seed=seed)
    if len(cands) < 5:
        raise ValueError(f"cscv_pbo: 候选数 {len(cands)} 过少（<5），"
                         f"排名统计无意义")

    n_blocks = (len(X) - train_size) // block_size
    M = build_returns_matrix(X, cands, start=train_size,
                             block_size=block_size, verbose=verbose)
    core = cscv_core(M.to_numpy(), S=S)
    result = {**core, "candidates": cands, "study": study, "matrix": M,
              "meta": {"train_size": train_size, "test_size": test_size,
                       "block_size": block_size, "n_blocks": n_blocks,
                       "n_trials": n_trials, "sampler": sampler,
                       "candidate_mode": candidate_mode,
                       "eval_start": str(M.index[0]),
                       "eval_end": str(M.index[-1])}}
    if verbose:
        _report(result)
    if plot:
        plot_cscv(result)
    return result


def _report(res):
    """打印 CSCV PBO 诊断报告。"""
    meta = res["meta"]
    lam = np.asarray(res["lambdas"])
    pct = np.percentile(lam, [5, 50, 95])
    print("\n" + "=" * 64)
    print("       CSCV PBO 诊断（标准版：单次搜索 + 静态参数族）")
    print("=" * 64)
    print(f"  搜索：trials={meta['n_trials']} | sampler={meta['sampler']} | "
          f"候选 {meta['candidate_mode']} = {res['N']} | 搜索段 {meta['train_size']} 天")
    print(f"  收益流：{res['N']} 参数 x {meta['n_blocks']} 块 x "
          f"{meta['block_size']} 天 = {meta['n_blocks'] * meta['block_size']} 天")
    print(f"  评估区：{meta['eval_start']} ~ {meta['eval_end']}")
    print(f"  CSCV：S={res['S']} 块（每块 {res['block_len']} 天）| "
          f"组合数 C({res['S']},{res['S'] // 2}) = {res['n_combos']}")
    print(f"  PBO = freq(lambda<=0) = {res['pbo']:.1%}"
          f"（{int(round(res['pbo'] * res['n_combos']))}/{res['n_combos']}）")
    print(f"  lambda 分位：5%={pct[0]:+.2f} | 50%={pct[1]:+.2f} | "
          f"95%={pct[2]:+.2f}")
    print(f"  IS 冠军多样性：{len(np.unique(res['n_star']))}/{res['N']} "
          f"个不同候选被选为冠军")
    print("  解读：PBO 越接近 0 排名传递性越好；>50% 严重预警"
          "（IS 最优比随机挑还不可信）")
    print("=" * 64 + "\n")


def plot_cscv(result, bins=60, figsize=(10, 4), save_path=None):
    """lambda 经验分布图（0 线 + PBO 标注）。"""
    from IPython.display import display
    lam = np.asarray(result["lambdas"])
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(lam, bins=bins, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(0.0, color="#C44E52", linestyle="--", linewidth=1.6)
    ax.set_title(f"CSCV lambda 分布 | PBO = {result['pbo']:.1%}（lambda<=0 占比）")
    ax.set_xlabel("lambda = ln(omega/(1-omega))，omega 为 IS 冠军的 OOS 相对排名")
    ax.set_ylabel("组合频数")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    display(fig)
    plt.close(fig)
