import logging
import re

from fastxc.cmd_executor import MultiDeviceTaskExecutor
from .utils import read_cmds_from_file

def pws_deployer(
    pws_cmd_file: str,
    gpu_list: list,
    gpu_task_num: list,
    log_file_path: str,
    cpu_count: int,
    dry_run: bool,
):
    """
    PWS stacking command deployer
      - readin pws_stack_cmds.txt
      - use MultiDeviceTaskExecutor to run commands
      - echo log on completion

      当检测到 -S 为 100 时，仅使用 CPU 执行器(线程池)，
      否则使用 GPU 执行器 (from_gpu_pool)，
      并在第三位字符为 '1' 时，覆盖所有 GPU 并发数为 1。
    """
    dep_logger = logging.getLogger(__name__)

    # 1) 读取所有命令
    tfpws_stack_cmd_list = read_cmds_from_file(pws_cmd_file)
    dep_logger.debug(f"Read {len(tfpws_stack_cmd_list)} commands from {pws_cmd_file}")
    if not tfpws_stack_cmd_list:
        dep_logger.warning("[pws_stack_cmd_deployer] No commands to run.")
        return

    # 2) 检测 -S 参数
    need_override = False
    need_cpu_only = False

    sample_cmd = tfpws_stack_cmd_list[0]
    match = re.search(r'-S\s+(\S{3})', sample_cmd)
    if match:
        s_val = match.group(1)  # 例如 '010'/'011'/'000'/'100'
        # 如果 s_val 为 '100'，使用纯 CPU
        if s_val == '100':
            need_cpu_only = True
        # 如果第三个字符为 '1'，覆盖 GPU 并发数为 1
        if s_val[2] == '1':
            need_override = True

    if need_override and not need_cpu_only:
        dep_logger.info("Detected tf-PWS ON, overriding gpu_task_num to all 1.")
        gpu_task_num = [1 for _ in gpu_task_num]

    # 3) 根据需要创建执行器
    if need_cpu_only:
        dep_logger.info("Detected '-S 100': using CPU thread pool only.")
        executor = MultiDeviceTaskExecutor.from_threadpool(
            num_threads=cpu_count,
            log_file_path=log_file_path,
            task_description="Stack (CPU Only)",
            queue_size=1,
            max_retry=3,
            enable_progress_bar=True,
        )
    else:
        # 默认使用 GPU 执行器
        executor = MultiDeviceTaskExecutor.from_gpu_pool(
            gpu_ids=gpu_list,
            gpu_workers=gpu_task_num,
            log_file_path=log_file_path,
            task_description="Stack",
            queue_size=1,
            max_retry=3,
            build_type="no_worker_id",
            enable_progress_bar=True,
        )

    # 4) 设置命令列表并执行
    executor.set_command_list(tfpws_stack_cmd_list)
    executor.run_all(dry_run=dry_run)

    # 5) 结束日志
    print("\n")
    dep_logger.info("Done Stack.\n")
