import os
import logging
from fastxc.cmd_executor import MultiDeviceTaskExecutor
from .utils import read_cmds_from_file


def rotate_deployer(output_dir: str, cpu_count: int, log_file_path: str, dry_run: bool):
    """
    use MultiDeviceTaskExecutor to deploy rotate commands
    """
    dep_logger = logging.getLogger(__name__)

    # 2) read commands from file
    cmd_list_dir = os.path.join(output_dir, "cmd_list")
    if not os.path.isdir(cmd_list_dir):
        dep_logger.warning(
            f"[rotate_cmd_deployer] cmd_list_dir not found: {cmd_list_dir}"
        )
        return
    cmd_files = [
        f
        for f in os.listdir(cmd_list_dir)
        if f.startswith("rotate_cmds_") and f.endswith(".txt")
    ]
    if not cmd_files:
        dep_logger.warning("[rotate_cmd_deployer] No rotate_cmds_*.txt files found.")
        return
    
    for cmd_file_name in sorted(cmd_files):
        file_path = os.path.join(cmd_list_dir, cmd_file_name)
        label = cmd_file_name[12:-4]
        label_cmds = read_cmds_from_file(file_path)
        if not label_cmds:
            continue
        executor = MultiDeviceTaskExecutor.from_threadpool(
            num_threads=cpu_count,
            log_file_path=log_file_path,
            task_description=f"Rotating-{label}",
            queue_size=1,
            max_retry=3,
            enable_progress_bar=True,
        )
        executor.set_command_list(label_cmds)
        executor.run_all(dry_run=dry_run)
