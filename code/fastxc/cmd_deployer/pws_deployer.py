# cmd_deployer.py  (或相应模块)

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence

from fastxc.cmd_executor import MultiDeviceTaskExecutor
from .utils import read_cmds_from_file


def pws_deployer(
    pws_cmd_file: str | Path,
    gpu_list: Sequence[int],
    gpu_task_num: Sequence[int],
    log_file_path: str | Path,
    cpu_count: int,
    dry_run: bool,
) -> None:
    """
    Deploy PWS / TF-PWS stacking commands.

    - 若 `-S` 检测为 `100` → 仅用 CPU 线程池。
    - 若 `-S` 的第三位为 1（TF-PWS 打开）→ 将每张 GPU 同时任务数强制为 1。
    """
    dep_logger = logging.getLogger(__name__)

    # ---------- 路径归一化 ----------------------------------------- #
    pws_cmd_file  = Path(pws_cmd_file).expanduser().resolve()
    log_file_path = Path(log_file_path).expanduser().resolve()

    # ---------- 1) 读取命令列表 ------------------------------------ #
    stack_cmds = read_cmds_from_file(pws_cmd_file)
    dep_logger.debug("Read %d commands from %s", len(stack_cmds), pws_cmd_file)
    if not stack_cmds:
        dep_logger.warning("[pws_deployer] No commands to run.")
        return

    # ---------- 2) 解析 -S 标志 ------------------------------------ #
    sample_cmd     = stack_cmds[0]
    s_val_match    = re.search(r"-S\s+(\S{3})", sample_cmd)
    need_cpu_only  = False
    need_override  = False

    if s_val_match:
        s_val = s_val_match.group(1)            # 例如 '100', '011'
        if s_val == "100":                      # 纯线性叠加
            need_cpu_only = True
        if s_val[2] == "1" and not need_cpu_only:  # 开启 tf-PWS
            need_override = True

    if need_override:
        dep_logger.info("TF-PWS detected; overriding gpu_task_num -> all 1")
        gpu_task_num = [1] * len(gpu_task_num)

    # ---------- 3) 构造执行器 -------------------------------------- #
    if need_cpu_only:
        dep_logger.info("Using CPU thread-pool executor (-S 100).")
        executor = MultiDeviceTaskExecutor.from_threadpool(
            num_threads=cpu_count,
            log_file_path=str(log_file_path),
            task_description="Stack (CPU only)",
            queue_size=1,
            max_retry=3,
            enable_progress_bar=True,
        )
    else:
        executor = MultiDeviceTaskExecutor.from_gpu_pool(
            gpu_ids=list(gpu_list),
            gpu_workers=list(gpu_task_num),
            log_file_path=str(log_file_path),
            task_description="Stack",
            queue_size=1,
            max_retry=3,
            build_type="no_worker_id",
            enable_progress_bar=True,
        )

    # ---------- 4) 运行 ------------------------------------------- #
    executor.set_command_list(stack_cmds)
    executor.run_all(dry_run=dry_run)

    # ---------- 5) 结束 ------------------------------------------- #
    dep_logger.info("Done Stack.\n")
    print()
