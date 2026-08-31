# -*- coding: utf-8 -*-
"""指标结果日志工具：把 DataFrame 指标结果写入本目录 logs/ 下的日志文件。

用途：notebook 的 cell stdout 输出无法被外部工具读取，把每次分析结果
落盘为与 notebook 同名的 .log（每条记录带时间戳），写报告时直接读日志
即可，无需手动复制粘贴输出。

用法（在 lazy_trading 目录下的任一 notebook 中）：

    from log_result import init_logger, log_result, log_print, read_log, set_log_echo

    init_logger("WalkForward+CPCV+PS")                        # 只初始化一次, 绑定文件名
    log_result(median_tbl, section="2.1 核心指标中位数")       # DataFrame 结果落盘
    log_print(f"沪深300 年化收益: {ann:.2%}", section="4.3 基准统计")  # 打印并落盘
    set_log_echo(False)                                       # 关闭控制台打印, 仅落盘
    log_print("批量结果…", section="4.4 批量")                 # 只写日志, 不打印
    set_log_echo(True)                                        # 恢复打印
    print(read_log())                                          # 读全部日志

不调用 init_logger 时, log_result 会尝试自动推断 notebook 名
(JPY_SESSION_NAME 或 cwd 下唯一 .ipynb), 推断成功则自动绑定; 推断
失败需先调用 init_logger 显式绑定。

日志文件：logs/<notebook 同名>.log，例如 logs/WalkForward+CPCV+PS.log。
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent / "logs"
_SEP = "=" * 72
_STEM: str | None = None   # 已绑定（初始化或自动推断缓存）的 notebook 名
_ECHO = True               # log_print 控制台打印全局开关（set_log_echo 可改）


def _detect_notebook_stem() -> str | None:
    """自动推断当前 notebook 文件名（不含扩展名）。

    优先取 Jupyter 运行时环境变量 JPY_SESSION_NAME（JupyterLab 格式为
    "<文件名>-<kernel_id>"，经典 notebook 为完整路径）；回退到 cwd 下
    唯一 .ipynb；仍失败返回 None，调用方需显式传 nb_stem。
    """
    session = os.environ.get("JPY_SESSION_NAME")
    if session:
        name = Path(session).stem if ".ipynb" in session else session
        # JupyterLab: 去掉尾部 "-<kernel_id>"（标准 uuid 形如 8-4-4-4-12 hex）
        name = re.sub(
            r"-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            "", name,
        )
        if name:
            return name
    ipynbs = list(Path.cwd().glob("*.ipynb"))
    if len(ipynbs) == 1:
        return ipynbs[0].stem
    return None


def _get_stem(nb_stem: str | None) -> str:
    """解析生效的 notebook 名：显式传参 > 已绑定 > 自动推断；解析成功即缓存。"""
    global _STEM
    if nb_stem is None:
        nb_stem = _STEM
    if nb_stem is None:
        nb_stem = _detect_notebook_stem()
    if nb_stem:
        _STEM = nb_stem
    if not nb_stem:
        raise ValueError(
            "无法推断 notebook 文件名：请先 init_logger('notebook名') 或传 nb_stem=..."
        )
    return nb_stem


def init_logger(nb_stem: str | None = None, log_dir=None) -> Path:
    """初始化日志器，绑定 notebook 文件名（整个 notebook 只调用一次）。

    绑定后 log_result()/read_log() 无需再传文件名。
    不传 nb_stem 时自动推断，推断失败需显式传入。

    Parameters
    ----------
    nb_stem : str | None
        notebook 文件名（不含 .ipynb）；None 时自动推断
    log_dir : str | Path | None
        日志目录；None 时用本模块同目录下的 logs/

    Returns
    -------
    Path : 日志文件路径（绑定后打印一次，后续调用不再输出）
    """
    stem = _get_stem(nb_stem)
    log_path = (Path(log_dir) if log_dir else _LOG_DIR) / f"{stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[log] 已绑定: {log_path}")
    return log_path


def set_log_echo(enabled: bool) -> None:
    """设置 log_print 是否同时在控制台打印（默认 True）。

    关闭后 log_print 只写日志、不打印（适合批量落盘时避免刷屏）；
    单次调用可用 log_print(..., echo=...) 临时覆盖，不改变全局状态。

    Parameters
    ----------
    enabled : bool
        True 打印并落盘 / False 仅落盘
    """
    global _ECHO
    _ECHO = bool(enabled)


def log_result(df, section="", mode="a", nb_stem=None, log_dir=None) -> None:
    """把 DataFrame/Series 指标结果写入日志（与 notebook 同名，带时间戳）。

    Parameters
    ----------
    df : pd.DataFrame | pd.Series | str
        指标分析结果；DataFrame/Series 以 to_string 全量落盘，str 原样写入
    section : str
        数据块标题（如报告章节名），用于检索定位
    mode : {"a", "w"}
        "a" 追加（默认，多次结果累积到同一日志）/ "w" 覆盖重写
    nb_stem : str | None
        notebook 文件名（不含 .ipynb）；None 时用 init_logger 绑定的名字，
        未绑定则自动推断
    log_dir : str | Path | None
        日志目录；None 时用本模块同目录下的 logs/

    Returns
    -------
    None : 只落盘不返回；日志路径在 init_logger 时已打印
    """
    nb_stem = _get_stem(nb_stem)

    log_dir = Path(log_dir) if log_dir else _LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{nb_stem}.log"

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = df.to_string() if hasattr(df, "to_string") else str(df)
    record = f"\n{_SEP}\n[{stamp}] {section}\n{_SEP}\n{body}\n"
    with open(log_path, mode, encoding="utf-8") as f:
        f.write(record)


def log_print(*args, section="", sep=" ", end="\n", mode="a", nb_stem=None, log_dir=None,
              echo=None) -> None:
    """把内容写入日志，控制台是否打印由 echo 控制（默认跟随 set_log_echo 全局开关）。

    等价于 echo=True 时的 print(*args, sep=sep, end=end) + 落盘；
    适合记录基准统计、换手率等零散输出。

    Parameters
    ----------
    args : Any
        与 print 相同的可变参数
    section : str
        数据块标题（如报告章节名），用于检索定位
    sep, end : str
        同 print
    mode, nb_stem, log_dir
        同 log_result
    echo : bool | None
        True 打印并落盘 / False 仅落盘；None（默认）跟随 set_log_echo 全局开关

    Returns
    -------
    None : 只落盘不返回；日志路径在 init_logger 时已打印
    """
    if echo is None:
        echo = _ECHO
    if echo:
        print(*args, sep=sep, end=end)
    text = sep.join(str(a) for a in args)
    log_result(text, section=section, mode=mode, nb_stem=nb_stem, log_dir=log_dir)


def read_log(nb_stem=None, log_dir=None) -> str:
    """读取当前 notebook 的完整日志（写报告时直接调用）。"""
    nb_stem = _get_stem(nb_stem)
    log_path = (Path(log_dir) if log_dir else _LOG_DIR) / f"{nb_stem}.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"日志不存在: {log_path}")
    return log_path.read_text(encoding="utf-8")
