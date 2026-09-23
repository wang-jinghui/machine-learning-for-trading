# -*- coding: utf-8 -*-
"""标准 CSCV PBO 诊断 v2：收益流构建改用 skfolio cross_val_predict(折级并行)。

定位：cscv_pbo.py(v1，保持不动)的加速替代版。搜索 / 候选 / CSCV 内核 /
报告全部复用 v1 组件(零代码分叉)，仅将"收益矩阵 M"的构建方式从手工双重
循环(N 候选 x K 块逐次串行 fit/predict)替换为 cross_val_predict +
WalkForward——折间调度与拼接(时间序、不重叠、无重复 test)由库保证，
不再手写循环。

与 v1 的逐块口径映射(WalkForward 取默认值即 v1 语义):

    v1(build_returns_matrix)                v2(WalkForward)
    ------------------------------------------------------------------
    fit 窗 [b0-ts, b0)，固定窗滚动           train_size=ts(expand_train=False 默认)
    test 块 [b0, b0+block_size)，不重叠       test_size=block_size
    紧邻、无 purge                          purged_size=0(默认)
    尾部残余块丢弃                           reduce_test=False(默认)
    评估区自 start 起                        X 裁剪为 X.iloc[start-ts:](前缀 ts 天
                                            仅供首折 fit 暖机；fit 回看是允许的)

并行与内存(设计约束，务必遵守):
  * cross_val_predict(n_jobs=mat_n_jobs) 是折级并行：逐候选提交一批任务，
    峰值并发 = min(mat_n_jobs, 块数)，批间同步(同一候选内各块窗口等长，
    批同步损耗小)；
  * joblib worker 各自持有 X 切片副本与筛选器中间对象(筛选器在全资产池上
    计算)，并发越高内存峰值越高。默认 mat_n_jobs=5 保守取值——建议不超过
    物理核数一半，留余量给其他任务；
  * 进程池跨调用复用由 joblib 管理，首次调用有暖机开销。

用法(与 v1 接口一致)::

    from cscv_pbo_v2 import cscv_pbo
    out = cscv_pbo(X, space=search_space, **cscv_params)                # 同 v1
    out = cscv_pbo(X, space=search_space, mat_n_jobs=6, **cscv_params)  # 调并行度

首次切换建议对拍(逐日收益一致性，取 2~3 个不同 ts 的候选)::

    from cscv_pbo import build_returns_matrix as build_v1
    from cscv_pbo_v2 import build_returns_matrix as build_v2
    M1 = build_v1(X, cands[:3], start=1008)
    M2 = build_v2(X, cands[:3], start=1008)
    assert (M1 - M2).abs().max().max() < 1e-12
"""
from __future__ import annotations

import sys

import pandas as pd
from skfolio.model_selection import WalkForward, cross_val_predict
from tqdm.auto import tqdm

from cpcv_search_base import build_pipeline
from cscv_pbo import (
    build_report,
    cscv_core,
    pick_candidates,
    plot_cscv,
    search_candidates_once,
)

# Windows GBK 控制台打印中文安全兜底(与同目录模块一致)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

__all__ = [
    "cscv_pbo",
    "build_returns_matrix",
    "search_candidates_once",
    "pick_candidates",
    "cscv_core",
    "build_report",
    "plot_cscv",
]


def build_returns_matrix(X, params_list, start, block_size=252, n_jobs=5,
                         verbose=True):
    """(v2)固定参数在评估区 [start, T] 上逐块滚动，用 cross_val_predict 构建收益矩阵。

    与 v1 逐日等价：每候选按 ts 裁剪输入窗(评估区 + 前置 ts 天)，
    WalkForward(test_size=block_size, train_size=ts) 全默认即 v1 的
    "fit 最近 ts 天 -> predict 下一块"口径；cross_val_predict 返回
    MultiPeriodPortfolio(各折 test 段组合，时间序)，returns_df 即拼接收益流。

    Parameters
    ----------
    n_jobs : int, default=5
        fold 级并行度(单候选内块并行)；峰值并发 = min(n_jobs, 块数)。
        joblib worker 内存开销大，建议不超过物理核数一半。

    Returns
    -------
    M : pd.DataFrame
        index = 评估区日期(n_blocks x block_size 天)，columns = 候选
        序号(0..N-1，与 params_list 顺序一致)。
    """
    n_blocks = (len(X) - start) // block_size
    if n_blocks < 2:
        raise ValueError(f"build_returns_matrix: 评估区不足 2 块"
                         f"({len(X) - start} 天 / {block_size} 天 = {n_blocks})")
    axis = X.index[start:start + n_blocks * block_size]
    if verbose:
        print(f"  收益流构建(v2 cross_val_predict，fold 并行 n_jobs={n_jobs})："
              f"{len(params_list)} 参数 x {n_blocks} 块 x {block_size} 天 "
              f"= {n_blocks * block_size} 天")
    cols = {}
    iterator = enumerate(params_list)
    if verbose:
        iterator = tqdm(iterator, total=len(params_list),
                        desc="收益流构建", unit="候选")
    for j, p in iterator:
        ts = int(p["train_size"])
        if start - ts < 0:
            raise ValueError(f"build_returns_matrix: 候选 {j} 的 train_size={ts} "
                             f"超过评估区起点 {start}，无法回看足量历史")
        # 裁剪输入 = 评估区 + 前置 ts 天：WalkForward 首折 test 恰从 start 开始
        x_view = X.iloc[start - ts:]
        cv = WalkForward(test_size=block_size, train_size=ts)
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


def cscv_pbo(X, space=None, *, train_size=1008, test_size=252, n_test_folds=2,
             n_jobs=4, cv_n_jobs=10, n_trials=2000, patience=100, min_delta=1e-4,
             sampler="tpe", seed=42, inner_purged_size=2, inner_embargo_size=2,
             candidate_mode="top_k", top_k=50, S=16, block_size=None,
             mat_n_jobs=5, verbose=True, plot=False):
    """(v2)端到端标准 CSCV PBO 诊断(单次搜索 -> 静态收益矩阵 -> 对称重排)。

    与 cscv_pbo.cscv_pbo(v1)接口一致(current 参数逐个同义；mat_n_jobs 为
    新增可选关键字参数，带默认值，不影响既有调用方式)。唯一实现差异：
    收益矩阵构建使用 cross_val_predict 折级并行(见 build_returns_matrix
    的 v2 口径说明)。

    Parameters
    ----------
    与 v1 同义：train_size / test_size / n_test_folds / n_jobs(trial 级) /
    cv_n_jobs(单 trial 内 CPCV 并行) / n_trials / patience / min_delta /
    sampler / seed / inner_purged_size / inner_embargo_size / candidate_mode /
    top_k / S / block_size / verbose / plot。

    mat_n_jobs : int, default=5
        收益流构建的 fold 级并行度(内存敏感，保守默认；详见模块 docstring)。

    Returns
    -------
    dict
        {pbo, lambdas, omega, n_star, sel_sharpe_is, sel_sharpe_oos,
         n_combos, S, block_len, T_used, N, candidates, study, matrix,
         meta, report}(与 v1 同构；meta 额外记录 mat_n_jobs)
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
        raise ValueError(f"cscv_pbo: 候选数 {len(cands)} 过少(<5)，"
                         f"排名统计无意义")

    n_blocks = (len(X) - train_size) // block_size
    M = build_returns_matrix(X, cands, start=train_size,
                             block_size=block_size, n_jobs=mat_n_jobs,
                             verbose=verbose)
    core = cscv_core(M.to_numpy(), S=S)
    result = {**core, "candidates": cands, "study": study, "matrix": M,
              "meta": {"train_size": train_size, "test_size": test_size,
                       "block_size": block_size, "n_blocks": n_blocks,
                       "n_trials": n_trials, "sampler": sampler,
                       "candidate_mode": candidate_mode,
                       "mat_n_jobs": mat_n_jobs,
                       "eval_start": str(M.index[0]),
                       "eval_end": str(M.index[-1])}}
    result["report"] = build_report(result)
    if plot:
        plot_cscv(result)
    return result
