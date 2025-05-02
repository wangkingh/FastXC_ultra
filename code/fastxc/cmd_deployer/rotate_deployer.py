# cmd_deployer.py  (或对应模块)
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from fastxc.cmd_executor import MultiDeviceTaskExecutor
from .utils import read_cmds_from_file


def rotate_deployer(
    output_dir: str | Path,
    cpu_count: int,
    log_file_path: str | Path,
    dry_run: bool,
) -> None:
    """
    Deploy rotation commands (N‑E‑Z ➜ R‑T) in *output_dir/cmd_list*.

    - 每个 `rotate_cmds_XXX.txt` 作为一批，单独跑一个 thread‑pool 执行器。
    - Path 或 str 均可输入。
    """
    log = logging.getLogger(__name__)

    # ---------- Path 归一化 -------------------------------------- #
    output_dir   = Path(output_dir).expanduser().resolve()
    cmd_list_dir = output_dir / "cmd_list"
    log_file_path = Path(log_file_path).expanduser().resolve()

    # ---------- 1) 找到命令文件 ---------------------------------- #
    if not cmd_list_dir.is_dir():
        log.warning("[rotate_deployer] cmd_list dir not found: %s", cmd_list_dir)
        return

    cmd_files: Sequence[Path] = sorted(
        fp for fp in cmd_list_dir.iterdir()
        if fp.name.startswith("rotate_cmds_") and fp.suffix == ".txt"
    )
    if not cmd_files:
        log.warning("[rotate_deployer] No rotate_cmds_*.txt files found.")
        return

    # ---------- 2) 每个文件一批执行 ------------------------------ #
    for cmd_file in cmd_files:
        label = cmd_file.stem[len("rotate_cmds_"):]    # 取文件名中间那段作为 label
        cmds  = read_cmds_from_file(cmd_file)
        if not cmds:
            log.warning("No commands in %s - skip.", cmd_file)
            continue

        executor = MultiDeviceTaskExecutor.from_threadpool(
            num_threads=cpu_count,
            log_file_path=str(log_file_path),
            task_description=f"Rotate-{label}",
            queue_size=1,
            max_retry=3,
            enable_progress_bar=True,
        )
        executor.set_command_list(cmds)
        executor.run_all(dry_run=dry_run)

    log.info("All rotate command batches finished.\n")
