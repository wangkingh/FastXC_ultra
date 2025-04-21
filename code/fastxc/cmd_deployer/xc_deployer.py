import logging
from fastxc.cmd_executor import MultiDeviceTaskExecutor
from .utils import read_cmds_from_file

def xc_deployer(
    xc_cmd_file: str,
    gpu_list: list,
    gpu_task_num: list,
    log_file_path: str,
    dry_run: bool,
):
    """
    multi tasks on multi devices
    params:
        xc_cmd_file: str, path to the command list file
        gpu_list: list, list of GPU ids
        gpu_task_num: list, list of tasks for each GPU
        log_file_path: str, path to the log file
        dry_run: bool, whether to run the commands
    """
    dep_logger = logging.getLogger(__name__)

    build_type = "with_worker_id"  # e.g. "with_worker_id"
    enable_progress_bar = True  # e.g. True

    # ========== 1. read cmd list ==========
    xc_cmd_list = read_cmds_from_file(xc_cmd_file)
    dep_logger.debug(f"Read {len(xc_cmd_list)} commands from {xc_cmd_file}")
    if not xc_cmd_list:
        dep_logger.warning("[xc_cmd_deployer] No commands to run.")
        return

    # ========== 2. create executor ==========
    executor = MultiDeviceTaskExecutor.from_gpu_pool(
        gpu_ids=gpu_list,
        gpu_workers=gpu_task_num,
        log_file_path=log_file_path,
        task_description="Cross-Correlation",
        queue_size=1,
        max_retry=3,
        build_type=build_type,
        enable_progress_bar=enable_progress_bar,
    )

    # ========== 3. set command list ==========
    executor.set_command_list(xc_cmd_list)

    # ========== 4. run all tasks ==========
    executor.run_all(dry_run=dry_run)

    # ========== 5. wait for all tasks to complete ==========
    print('\n')
    message = f"Done for Cross Correlation!\n"
    dep_logger.info(message)