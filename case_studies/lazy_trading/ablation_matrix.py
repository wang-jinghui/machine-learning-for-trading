# -*- coding: utf-8 -*-
"""Extremes 消融矩阵驱动器：批量运行 ablation_extremes.py 的多组实验。

每组 = (test_size, train_size, objective)，组内跑全部 5 个档位（--arms 默认）。
全部组的输出累积写入同一个日志文件 logs/<--log-name>.log
（默认 Extremes+Ablation.log，追加模式），每组以"矩阵组开始 / 矩阵组完成"
段做边界标记（段头带时间戳），回读时按组名检索即可。

切分铺满 train × test 全网格（3×3=9）：train ∈ {756, 504, 252} × test ∈ {252, 126, 63}，
顺序按手动日志信息价值排序（非几何序），截断时优先保住高价值切分。
每档 trial 数默认 = 参数空间网格规模的一半（动态，见 ablation_extremes.py）。

背景：此前 WalkForward+ParameterSearch(+Extremes).log 是手动逐轮执行的，
存在配置混杂（corr 空间、fitness 集、k 口径不一致）与人为出错风险。
本矩阵用同一自动化脚本把 切分 × 目标函数 × 档位 系统补齐，
日志段落风格与 walkforward_parameter_search.py 一致，可直接回读对照。

特性：
* 按 SPLITS/OBJECTIVES 展开顺序（优先级从高到低）串行执行；
* --max-hours 总时限：超时在组间优雅截断；
* 断点续跑：日志已含"矩阵组完成: <组名>"的组自动跳过；
* 中断的组（有组头无完成标记）重跑时整组再追加一轮（段带时间戳可区分，
  以最新一轮为准），不清除历史内容。

用法::

    G:\\Anaconda3\\envs\\ml4t\\python.exe ablation_matrix.py --max-hours 15
    G:\\Anaconda3\\envs\\ml4t\\python.exe ablation_matrix.py --max-hours 0.2 --n-trials 3   # 冒烟
    G:\\Anaconda3\\envs\\ml4t\\python.exe ablation_matrix.py --dry-run                    # 只看计划
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ablation_extremes as ab

# Windows GBK 控制台打印中文安全兜底
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
DEFAULT_LOG_NAME = "Extremes+Ablation"

# 切分全网格 train ∈ {756, 504, 252} × test ∈ {252, 126, 63}，按信息价值排序（高→低）：
#   (126, 504) 消融原设计，8/31 日志无筛选 vs k200 混杂点所在
#   (63, 504)  手动日志三目标一致贴 k=0.5 上边界的切分
#   (63, 756)  手动日志 MDD 钉死 16.53%（回撤系统性）结论的源头
#   (252, 504) 9 折快切分，手动有对应轮
#   (63, 252)  37 折最长 OOS，手动日志唯一选小 k（0.2）的例外切分
#   (126, 756) 手动 B-2 高分轮（Score 1.063）所在
#   (126, 252) 手动 B-2 高收益轮（年化 15.22%）所在
#   (252, 756) 手动首轮所在（k0.5 贴边、OOS MDD 高）
#   (252, 252) 手动极端轮所在（skew -149.8%）
SPLITS = [(126, 504), (63, 504), (63, 756), (252, 504), (63, 252),
          (126, 756), (126, 252), (252, 756), (252, 252)]
# 目标函数（与手动日志三阶段对应）：ASR 基线 / ASR−MDD / ASR−MDD+skew
OBJECTIVE_NAMES = ["annualized_sharpe_ratio", "asr_mdd", "asr_mdd_skew"]


def cell_name(test: int, train: int, obj: str) -> str:
    """组名 = {test}_{train}_{objective}（日志内检索键）。"""
    return f"{test}_{train}_{obj}"


def cell_done(log_stem: str, name: str) -> bool:
    """组是否已完成：日志文件中存在该组的"矩阵组完成"标记段。"""
    path = LOG_DIR / f"{log_stem}.log"
    if not path.is_file():
        return False
    return f"矩阵组完成: {name}" in path.read_text(encoding="utf-8")


def run_cell(log_stem: str, test: int, train: int, obj: str, extra: list[str]) -> None:
    """跑一组（组内 5 档由 ablation_extremes.main 的 --arms 默认驱动）。"""
    argv = ["--test-size", str(test), "--train-size", str(train),
            "--objective", obj, "--log-name", log_stem, *extra]
    ab.main(argv)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extremes 消融矩阵驱动器")
    parser.add_argument("--log-name", default=DEFAULT_LOG_NAME,
                        help=f"日志文件名（logs/ 下，不含扩展名；默认 {DEFAULT_LOG_NAME}，全部组累积写入）")
    parser.add_argument("--max-hours", type=float, default=15.0,
                        help="总运行时限（小时），超时在组间截断（默认 15）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 组（0=全部）")
    parser.add_argument("--skip-objective", choices=OBJECTIVE_NAMES, action="append", default=[],
                        help="跳过的目标函数（可多次），用于缩减矩阵")
    parser.add_argument("--cell-prefix", default=None,
                        help="只跑组名以该前缀开头的组（如 63_756）")
    parser.add_argument("--dry-run", action="store_true", help="只打印运行计划，不执行")
    # 以下透传到 ablation_extremes.main
    parser.add_argument("--n-trials", type=int, default=None,
                        help="每档 optuna trial 数（默认 None：由 ablation 按网格规模一半动态计算）")
    parser.add_argument("--n-jobs", type=int, default=12, help="trial 级并行度")
    parser.add_argument("--cv-n-jobs", type=int, default=4, help="单 trial 内 cross_val_predict 并行度")
    parser.add_argument("--patience", type=int, default=100, help="optuna 提前停止阈值")
    parser.add_argument("--seed", type=int, default=42, help="optuna TPE 随机种子")
    args = parser.parse_args(argv)

    cells = [(t, tr, obj) for t, tr in SPLITS for obj in OBJECTIVE_NAMES
             if obj not in args.skip_objective]
    if args.cell_prefix:
        cells = [c for c in cells if cell_name(*c).startswith(args.cell_prefix)]
    if args.limit:
        cells = cells[: args.limit]
    if not cells:
        print("[driver] 没有待运行的组")
        return

    extra = ["--n-jobs", str(args.n_jobs), "--cv-n-jobs", str(args.cv_n_jobs),
             "--patience", str(args.patience), "--seed", str(args.seed)]
    if args.n_trials is not None:
        extra.insert(0, "--n-trials")
        extra.insert(1, str(args.n_trials))

    total = len(cells)
    t0 = time.time()
    deadline = t0 + args.max_hours * 3600
    done_before = sum(1 for c in cells if cell_done(args.log_name, cell_name(*c)))
    log_path = LOG_DIR / f"{args.log_name}.log"
    print(f"[driver] 消融矩阵：共 {total} 组（{done_before} 组已完成将跳过），"
          f"日志 {log_path.name}，时限 {args.max_hours}h，"
          f"截断时刻 {time.strftime('%m-%d %H:%M', time.localtime(deadline))}")
    if args.dry_run:
        for i, (test, train, obj) in enumerate(cells, 1):
            mark = "[done] " if cell_done(args.log_name, cell_name(test, train, obj)) else "[todo] "
            print(f"  {i:2d}. {mark}{cell_name(test, train, obj)} ({train} 训练窗 / {test} 测试段)")
        return

    ab.init_logger(args.log_name)
    for i, (test, train, obj) in enumerate(cells, 1):
        name = cell_name(test, train, obj)
        if cell_done(args.log_name, name):
            print(f"[driver][{i}/{total}] {name}：已完成，跳过")
            continue
        if time.time() > deadline:
            print(f"[driver][{i}/{total}] {name}：已达时限，截断退出"
                  f"（该组及后续组待续跑；重跑本命令即可续）")
            break
        ab.log_print(f"train={train} test={test} | objective={obj} | 5 档消融",
                     section=f"矩阵组开始: {name}", echo=True)
        t_start = time.time()
        print(f"[driver][{i}/{total}] {name}：开始 @ {time.strftime('%H:%M:%S')}")
        run_cell(args.log_name, test, train, obj, extra)
        ab.log_print("完成", section=f"矩阵组完成: {name}", echo=False)
        el = (time.time() - t_start) / 60
        acc = (time.time() - t0) / 3600
        print(f"[driver][{i}/{total}] {name}：完成，耗时 {el:.1f} 分钟"
              f"（累计 {acc:.1f}h / 上限 {args.max_hours}h）")


if __name__ == "__main__":
    main()
