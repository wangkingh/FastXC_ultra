# cmd_deployer.py  (or wherever this lives)
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from fastxc.cmd_executor import MultiDeviceTaskExecutor
from .utils import read_cmds_from_file


def xc_deployer(
    xc_cmd_file: str | Path,
    gpu_list: Sequence[int],
    gpu_task_num: Sequence[int],
    log_file_path: str | Path,
    dry_run: bool,
) -> None:
    """
    Launch cross-correlation commands on multiple GPUs.

    Parameters
    ----------
    xc_cmd_file   : Path | str
        File containing one command per line.
    gpu_list      : Sequence[int]
        Physical GPU IDs, e.g. [0,1,2,3].
    gpu_task_num  : Sequence[int]
        Number of concurrent tasks bound to each GPU.
        Length must equal len(gpu_list).
    log_file_path : Path | str
        Executor log file.
    dry_run       : bool
        If True, only print commands without executing.
    """
    dep_logger = logging.getLogger(__name__)

    # ---------- Path normalisation ---------------------------------- #
    xc_cmd_file   = Path(xc_cmd_file).expanduser().resolve()
    log_file_path = Path(log_file_path).expanduser().resolve()

    # ---------- 1. read cmd list ------------------------------------ #
    xc_cmd_list = read_cmds_from_file(str(xc_cmd_file))
    dep_logger.debug("Read %d commands from %s", len(xc_cmd_list), xc_cmd_file)
    if not xc_cmd_list:
        dep_logger.warning("[xc_deployer] No commands to run.")
        return

    # ---------- 2. create executor ---------------------------------- #
    executor = MultiDeviceTaskExecutor.from_gpu_pool(
        gpu_ids=gpu_list,
        gpu_workers=gpu_task_num,
        log_file_path=str(log_file_path),
        task_description="Cross-Correlation",
        queue_size=1,
        max_retry=3,
        build_type="with_worker_id",
        enable_progress_bar=True,
    )

    # ---------- 3. submit commands ---------------------------------- #
    executor.set_command_list(xc_cmd_list)

    # ---------- 4. run ---------------------------------------------- #
    executor.run_all(dry_run=dry_run)

    # ---------- 5. summary ------------------------------------------ #
    dep_logger.info("Done for Cross Correlation!\n")
    print()  # 保证输出换行
