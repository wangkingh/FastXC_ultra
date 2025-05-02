# cmd_deployer.py
from __future__ import annotations

import logging
from pathlib import Path
from threading import Event, Thread

from fastxc.cmd_executor import MultiDeviceTaskExecutor
from .utils import read_cmds_from_file, check_and_log_file_count


def sac2spec_deployer(
    cmd_list_file: str | Path,
    sac_spec_list_dir: str | Path,
    segspec_dir: str | Path,
    log_file_path: str | Path,
    dry_run: bool,
) -> None:
    """
    执行 sac2spec 命令列表，并实时监控 segspec 目录下生成文件数量。

    所有路径参数可为 str 或 pathlib.Path。
    """
    # ---------- Path 归一化 ---------- #
    cmd_list_file     = Path(cmd_list_file).expanduser().resolve()
    sac_spec_list_dir = Path(sac_spec_list_dir).expanduser().resolve()
    segspec_dir       = Path(segspec_dir).expanduser().resolve()
    log_file_path     = Path(log_file_path).expanduser().resolve()

    dep_logger = logging.getLogger(__name__)

    # ---------- 1) 读取命令 ---------- #
    sac2spec_cmds = read_cmds_from_file(str(cmd_list_file))
    dep_logger.debug("Read %d commands from %s", len(sac2spec_cmds), cmd_list_file)
    if not sac2spec_cmds:
        dep_logger.warning("[sac2spec_deployer] No commands to run.")
        return

    # ---------- 2) 后台线程：文件计数 ---------- #
    total_tasks = 0
    for sac_list in sac_spec_list_dir.glob("*sac_list*"):
        with sac_list.open() as fp:
            total_tasks += sum(1 for line in fp if line.strip())

    stop_event = Event()
    Thread(
        target=check_and_log_file_count,
        args=(segspec_dir, stop_event, 10, total_tasks, __name__),
        daemon=True,
    ).start()

    # ---------- 3) 执行命令 ---------- #
    executor = MultiDeviceTaskExecutor.from_threadpool(
        num_threads=len(sac2spec_cmds),   # 一条命令一个线程
        log_file_path=str(log_file_path),
        task_description="SAC2SPEC",
        queue_size=1,
        max_retry=3,
        enable_progress_bar=False,
    )
    executor.set_command_list(sac2spec_cmds)
    executor.run_all(dry_run=dry_run)

    # ---------- 4) 结束后台线程 ---------- #
    stop_event.set()

    # ---------- 5) 统计最终文件数 ---------- #
    file_count = sum(1 for _ in segspec_dir.rglob("*") if _.is_file())
    dep_logger.info(
        "Done SAC2SPEC! %d/%d files written out.\n",
        file_count,
        total_tasks,
    )
