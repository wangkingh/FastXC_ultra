# utils.py
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import List


# ──────────────────────────────────────────────────────────────── #
#  1. 读取命令列表                                                 #
# ──────────────────────────────────────────────────────────────── #
def read_cmds_from_file(file_path: str | Path) -> List[str]:
    """
    Read non-empty lines as shell commands.

    Returns empty list and logs error if file not found.
    """
    fp = Path(file_path).expanduser().resolve()
    try:
        return [line.strip() for line in fp.read_text().splitlines() if line.strip()]
    except FileNotFoundError:
        logging.error("Command list file not found: %s", fp)
        return []


# ──────────────────────────────────────────────────────────────── #
#  2. 后台线程：定期统计文件数量                                    #
# ──────────────────────────────────────────────────────────────── #
def check_and_log_file_count(
    base_dir: str | Path,
    stop_event: threading.Event,
    interval: int,
    total_tasks: int,
    logger_name: str = __name__,
) -> None:
    """
    每隔 ``interval`` 秒递归统计 ``base_dir`` 内文件数，并写日志::

        spectrums written out: 123/456

    线程收到 ``stop_event.set()`` 后退出。
    """
    dep_logger = logging.getLogger(logger_name)
    base_dir   = Path(base_dir).expanduser().resolve()

    while not stop_event.is_set():
        file_count = sum(1 for _ in base_dir.rglob("*") if _.is_file())
        dep_logger.info("spectrums written out: %d/%d", file_count, total_tasks)
        time.sleep(interval)
