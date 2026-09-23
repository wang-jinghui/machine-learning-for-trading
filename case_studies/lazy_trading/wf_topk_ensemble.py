# -*- coding: utf-8 -*-
"""Top-K 参数"资产池融合"评估（ensemble）——搜索完成后的独立后处理。

架构定位：与 topk_paths_eval（每 rank 一条独立 OOS 路径，参数高原印证）
互补的评估视角——本模块把每折 study 的 top-k 候选参数各自的筛选结果
（资产集）合并去重为一个"融合资产池"，以等权（EW）组合在 IS / OOS 上
应用，产出单一融合策略路径并与 top1 生产参数逐折对比：

* 每折：top_trials(study, k) → 每组参数在训练段**各自 train_size 窗口**
  执行纯筛选链（build_pipeline 去掉 EqualWeighted 的前 5 步）→ 资产集；
* 融合池：资产出现票数 ≥ min_votes 后按 X 列序过滤（默认 1 = 纯并集
  去重；>1 为频次阈值投票，过滤仅被少数参数组选中的偶发资产）；
* 组合：融合池上 EqualWeighted，fit top1 拟合窗齐备行（S 为多窗口并集，
  可能含较晚上市资产的前导 NaN；EW 无拟合参数，仅取无 NaN 行段满足
  API 校验）→ predict top1 拟合窗（最近 ts 天）与纯净 OOS test 段
  （与 run_params_on_fold 同构，唯一差别是资产池）；
* 对比：top1 的 fold_results 现成 train/test（引用不重算）vs 融合——
  fold 级指标表 + 全折 test 段拼接的完整 OOS 路径指标。

纪律（与 topk_paths_eval 一致）：只评估、不选参——融合构造仅用 IS 侧
信息（top-k 参数本身来自 IS 搜索），OOS 段保持纯净；即使融合 OOS 更优
也不得据此换参（换参 = 用 OOS 选择 = 消耗该段）。

消费端：两套搜索框架（wf_adaptive / wf_cpcv）的 fold_results 同构，均可
直接消费；wf_kwargs 未给时从 folds 推导折位参数（防静默错位）。

调用形态::

    from wf_topk_ensemble import topk_ensemble_eval

    ens = topk_ensemble_eval(X, folds, k=10)                 # 默认纯并集去重
    ens = topk_ensemble_eval(X, folds, k=10, min_votes=2)    # 频次阈值投票
    ens["summary"]          # fold 级 top1 vs 融合 对比表
    ens["paths_metrics"]    # 完整 OOS 路径指标（top1 / ensemble）
    ens["audit"]            # summarize_top_k_params 的 IS 侧候选审计
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd
from skfolio import MultiPeriodPortfolio
from skfolio.optimization import EqualWeighted

from cpcv_analysis import top_trials
from cpcv_search_base import build_pipeline
from wf_cpcv_robustness import derive_wf_kwargs
from wf_cpcv_search import _outer_wf, summarize_top_k_params

# Windows GBK 控制台打印中文安全兜底（与同目录脚本一致）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


# ---------------------------------------------------------------------------
# 单组候选筛选 / 单折融合
# ---------------------------------------------------------------------------
def _selected_assets(params, w, builder):
    """单组参数在训练窗口 w 上执行纯筛选链，返回筛选后的资产名列表。

    params 为 top_trials 的 trial.params（fitness_measures 为名称字符串，
    build_pipeline 直接可解析）；切片 [:-1] 去掉 EqualWeighted，只保留
    complete→variance→extremes→nondomin→correlate 的筛选语义。w 为该
    参数自己的 train_size 窗口（与搜索时该参数的评估窗口一致）。
    """
    chain = builder(params)[:-1]
    return list(chain.fit_transform(w).columns)


def _ensemble_fold(X, fold_id, tr_idx, te_idx, cands, min_votes, builder,
                   verbose=True):
    """单折融合：top-k 候选各自筛选 → 资产并集（min_votes 阈值）→ EW → IS/OOS。

    每组候选参数用**自己的 train_size 窗口**执行筛选链（忠于搜索语义）；
    融合池按 X 列序过滤（列序稳定，predict 切片一致）；EW 无拟合参数，
    fit 仅需无 NaN 的齐备行段（S 为多窗口筛选并集，可能含在 top1 拟合
    窗口前段尚未上市的资产——前导 NaN 会被 validate_data 拒绝；取窗口内
    全部资产齐备的行段即可，组合语义完全由各组筛选链承担）。

    空池折（min_votes 过严时可能发生）返回 train/test=None 并显式告警
    （不静默、不自动降级——降级会静默改变规则，告警后由调用方决定）。
    """
    X_tr = X.iloc[tr_idx]
    sel_lists = []
    for t in cands:
        w = X_tr.iloc[-int(t.params["train_size"]):]
        sel_lists.append(_selected_assets(t.params, w, builder))
    votes = Counter()
    for sel in sel_lists:
        votes.update(sel)
    S = [c for c in X_tr.columns if votes.get(c, 0) >= min_votes]
    n_top1 = len(sel_lists[0]) if sel_lists else 0
    overlap = len(set(S) & set(sel_lists[0])) if sel_lists else 0
    if not S:
        if verbose:
            print(f"Fold {fold_id}: 融合池为空（min_votes={min_votes} 过严，"
                  f"候选 {len(cands)} 组无一资产达到票数阈值）")
        return {"fold": fold_id, "train": None, "test": None, "assets": [],
                "votes": pd.Series(dtype="int64"), "n_union": 0,
                "n_top1": n_top1, "overlap": 0, "n_cands": len(cands)}
    ts = int(cands[0].params["train_size"])   # top1 拟合窗口口径（与 run_params_on_fold 对齐）
    w = X_tr.iloc[-ts:][S]
    # EW 无拟合参数：fit 仅需无 NaN 齐备行段（S 可能含上线较晚资产，其前导
    # NaN 会被 validate_data 拒绝）；predict 阶段的 NaN 处理与原 pipeline
    # 同机制（新上市资产的 NaN 段由 skfolio 组合层承担）。
    ew = EqualWeighted().fit(w.dropna())
    # IS 段 = top1 拟合窗（fit 什么就评什么；此前用整段 X_tr 会把未参与
    # 搜索/拟合的更早天混进 IS 口径，与 run_params_on_fold 的 train 侧同步修正）
    ens_train = ew.predict(w)
    ens_test = ew.predict(X.iloc[te_idx][S])
    ens_train.name = f"EnsTrain{fold_id}"   # predict 不接受 portfolio_params(0.20.x), 预测后设置名称
    ens_test.name = f"EnsFold{fold_id}"
    # 段长度对齐校验：融合与 top1 必须覆盖同一 IS / OOS 段（防静默错位）
    assert len(ens_train.returns) == len(w), "融合 IS 段长度与 top1 拟合窗不一致"
    assert len(ens_test.returns) == len(te_idx), "融合 OOS 段长度与 test 段不一致"
    return {"fold": fold_id, "train": ens_train, "test": ens_test,
            "assets": S,
            "votes": pd.Series(dict(votes)).sort_values(ascending=False),
            "n_union": len(S), "n_top1": n_top1, "overlap": overlap,
            "n_cands": len(cands)}


# ---------------------------------------------------------------------------
# 主入口：Top-K 参数资产池融合评估
# ---------------------------------------------------------------------------
def topk_ensemble_eval(X, folds, k=10, min_votes=1,
                       metrics=("annualized_sharpe_ratio", "annualized_mean",
                                "max_drawdown"),
                       wf_kwargs=None, purged_size=1, reduce_test=True,
                       verbose=True, pipeline_builder=None):
    """每折 Top-K 候选参数资产池融合（EW），与 top1 生产参数 IS/OOS 对比。

    Parameters
    ----------
    X : pd.DataFrame，资产收益（与搜索同一份数据；X / X_net 均可）
    folds : list，搜索返回的 fold_results（每折含 study；wf_adaptive /
        wf_cpcv 两框架同构可直接消费）
    k : int，每折取 top-k 候选参数（top_trials 同 calc_dsr 去重口径；
        候选不足 k 时用实际数量——每折独立融合，无跨折路径对齐约束）
    min_votes : int，融合池票数阈值——1 = 纯并集去重（默认）；>1 = 只
        保留被 >= min_votes 组参数选中的资产（频次投票，过滤偶发资产）
    metrics : tuple，完整 OOS 路径对比指标（skfolio 现成属性名）
    wf_kwargs : dict | None，run_params_on_fold 折位参数（test_size /
        train_size / purged_size / reduce_test）；None 时优先读 folds 的
        "outer_wf" 显式外层窗口（旧数据回退段长众数推导，与嵌套搜索
        1:1 对齐，防静默错位），须与搜索配置核对
    purged_size / reduce_test : 旧数据回退推导时的默认（出 "outer_wf"
        时被忽略；与搜索侧默认 1 / True 一致，含缩短尾折）
    verbose : bool，逐折进度与空池告警打印
    pipeline_builder : 可选的管道构建器（默认 build_pipeline），签名与
        build_pipeline 一致（params → Pipeline）；供消融变体注入

    Returns
    -------
    dict :
        folds  : list，每折 {fold, train, test, assets(融合池资产名列表),
            votes(Series, 票数降序), n_union, n_top1, overlap, n_cands}；
            空池折的组合为 None、assets 为空列表
        summary : DataFrame，fold 级对比——列 = fold | n_cands | n_top1
            | n_union | overlap | top1 train ASR | ens train ASR
            | top1 test ASR | ens test ASR | Δtest ASR（融合 − top1）
        paths : {"top1": MultiPeriodPortfolio, "ensemble": MultiPeriod
            Portfolio | None}——全折 test 段拼接的完整 OOS 路径（融合
            缺折时不拼接，避免断裂路径）
        paths_metrics : DataFrame，行 = top1 / ensemble，列 = metrics
            （skfolio 现成属性直取，不自算）
        audit : DataFrame，summarize_top_k_params(folds, k=k) 的 IS 侧
            候选审计（score / gap2top1 / 参数快照）
    """
    if k < 1:
        raise ValueError("k 必须 >= 1")
    if min_votes < 1:
        raise ValueError("min_votes 必须 >= 1（1 = 纯并集去重，>1 = 频次投票）")
    if wf_kwargs is None:
        wf_kwargs = derive_wf_kwargs(folds, purged_size=purged_size,
                                     reduce_test=reduce_test)
    outer_cv = _outer_wf(X, **wf_kwargs)
    splits = list(outer_cv.split(X))    # fold i 与搜索同一枚举口径
    builder = pipeline_builder if pipeline_builder is not None else build_pipeline

    ens_folds, rows = [], []
    for f in folds:
        i = f["fold"]
        tr_idx, te_idx = splits[i]
        cands = top_trials(f["study"].trials, k=k)
        r = _ensemble_fold(X, i, tr_idx, te_idx, cands, min_votes, builder,
                           verbose=verbose)
        ens_folds.append(r)
        ens_tr = (r["train"].annualized_sharpe_ratio
                  if r["train"] is not None else np.nan)
        ens_te = (r["test"].annualized_sharpe_ratio
                  if r["test"] is not None else np.nan)
        rows.append({
            "fold": i, "n_cands": r["n_cands"], "n_top1": r["n_top1"],
            "n_union": r["n_union"], "overlap": r["overlap"],
            "top1 train ASR": round(f["train ASR"], 4),
            "ens train ASR": round(ens_tr, 4),
            "top1 test ASR": round(f["test ASR"], 4),
            "ens test ASR": round(ens_te, 4),
            "Δtest ASR": round(ens_te - f["test ASR"], 4),
        })
        if verbose and r["test"] is not None:
            print(f"Fold {i}: 候选={r['n_cands']} | top1资产={r['n_top1']} | "
                  f"融合池={r['n_union']}（重叠={r['overlap']}）| "
                  f"top1_test={f['test ASR']:.4f} | ens_test={ens_te:.4f}")
    summary = pd.DataFrame(rows)

    # 完整 OOS 路径：全折 test 段拼接（top1 直接引用 folds 现成组合，不重算）
    oos_days = sum(len(f["test"].returns) for f in folds)
    top1_mpp = MultiPeriodPortfolio([f["test"] for f in folds])
    if all(r["test"] is not None for r in ens_folds):
        ens_mpp = MultiPeriodPortfolio([r["test"] for r in ens_folds])
    else:
        ens_mpp = None
        if verbose:
            print("融合存在空池折：完整 OOS 路径不拼接（避免断裂路径），"
                  "仅输出 fold 级对比")
    labels, mpps = ["top1"], [top1_mpp]
    if ens_mpp is not None:
        labels.append("ensemble")
        mpps.append(ens_mpp)
    paths_metrics = pd.DataFrame(
        [[getattr(m, met) for met in metrics] for m in mpps],
        index=labels, columns=list(metrics))
    if verbose:
        print(f"完整 OOS 路径: {len(folds)} 折拼接, 共 {oos_days} 天")
    return {"folds": ens_folds, "summary": summary,
            "paths": {"top1": top1_mpp, "ensemble": ens_mpp},
            "paths_metrics": paths_metrics,
            "audit": summarize_top_k_params(folds, k=k)}


# ---------------------------------------------------------------------------
# CLI（冒烟/小预算；完整检验请在 notebook 会话复用已完成的 fold_results）
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Top-K 参数资产池融合评估冒烟 CLI")
    parser.add_argument("--search", choices=("adaptive", "cpcv"), default="adaptive",
                        help="搜索框架（adaptive=wf_adaptive_search 单层 WF；cpcv=嵌套）")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="每折 optuna trial 数（小预算冒烟）")
    parser.add_argument("--tail-days", type=int, default=None,
                        help="只取数据尾部 N 天控制折数（None=全量）")
    parser.add_argument("--k", type=int, default=10, help="每折 Top-K 候选数")
    parser.add_argument("--min-votes", type=int, default=1,
                        help="融合池票数阈值（1=纯并集；>1=频次投票）")
    parser.add_argument("--data", choices=("X", "X_net"), default="X",
                        help="数据口径（默认 X）")
    parser.add_argument("--n-jobs", type=int, default=8, help="optuna trial 级并行度")
    parser.add_argument("--cv-n-jobs", type=int, default=1,
                        help="cpcv 模式单 trial 内 CPCV 并行度")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-verbose", action="store_true",
                        help="搜索阶段打印折进度/optuna 进度条（默认静默，"
                             "冒烟输出只保留融合评估）")
    args = parser.parse_args(argv)

    import walkforward_parameter_search as wfps
    from log_result import init_logger, log_print, log_result

    init_logger("wf_topk_ensemble_smoke")
    X_all, X_net, info = wfps.load_data()
    X_use = X_net if args.data == "X_net" else X_all
    if args.tail_days:
        X_use = X_use.tail(args.tail_days)
    log_print(f"数据: {args.data} | 行数={len(X_use)} | inf={info['inf_cols']}",
              section="数据加载", echo=True)

    if args.search == "adaptive":
        from wf_adaptive_search import adaptive_wf_search, load_adaptive_config
        cfg = load_adaptive_config()
        search_kwargs = {**cfg["search_kwargs"], "n_trials": args.n_trials,
                         "n_jobs": args.n_jobs, "seed": args.seed,
                         "verbose": args.search_verbose}
        folds = adaptive_wf_search(X_use, space=cfg["space"], **search_kwargs)
    else:
        from wf_cpcv_search import nested_adaptive_search, load_nested_config
        cfg = load_nested_config()
        search_kwargs = {**cfg["search_kwargs"], "n_trials": args.n_trials,
                         "n_jobs": args.n_jobs, "cv_n_jobs": args.cv_n_jobs,
                         "seed": args.seed, "verbose": args.search_verbose}
        folds = nested_adaptive_search(X_use, space=cfg["space"], **search_kwargs)

    out = topk_ensemble_eval(X_use, folds, k=args.k, min_votes=args.min_votes)
    log_print(out["summary"].to_string(), section="融合对比（fold 级）", echo=True)
    log_print(out["paths_metrics"].to_string(),
              section="完整 OOS 路径指标（top1 vs 融合）", echo=True)
    log_result(out["audit"], section="Top-K 候选审计（IS 侧）")


if __name__ == "__main__":
    main()
