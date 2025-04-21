import os
import logging
from threading import Event, Thread
import glob
from fastxc.cmd_executor import MultiDeviceTaskExecutor
from .utils import read_cmds_from_file
from .utils import check_and_log_file_count


def sac2spec_deployer(
    cmd_list_file: str,
    sac_spec_list_dir: str,
    segspec_dir: str,
    log_file_path: str,
    dry_run: bool,
):
    """
    Use MultiDeviceTaskExecutor to deploy commands (instead of ThreadPoolExecutor).
    Also keep a background thread to periodically check the file count.
    """
    dep_logger = logging.getLogger(__name__)

    # 1) read commands from file
    sac2sepc_cmds = read_cmds_from_file(cmd_list_file)
    dep_logger.debug(f"Read {len(sac2sepc_cmds)} commands from {cmd_list_file}")
    if not sac2sepc_cmds:
        dep_logger.warning("[sac2spec_cmd_deployer] No commands to run.")
        return

    # 2) start background thread for checking file count
    sac_lists = glob.glob(os.path.join(sac_spec_list_dir, "*sac_list*"))
    total_tasks = 0
    for sac_list in sac_lists:
        with open(sac_list, "r") as f:
            total_tasks += sum(1 for line in f if line.strip())
    stop_event = Event()
    check_thread = Thread(
        target=check_and_log_file_count,
        args=(segspec_dir, stop_event, 10, total_tasks, __name__),
        daemon=True,
    )
    check_thread.start()

    # 3) create Executor and run
    executor = MultiDeviceTaskExecutor.from_threadpool(
        num_threads=len(sac2sepc_cmds),  # use one thread per task
        log_file_path=log_file_path,
        task_description="SAC2SPEC",
        queue_size=1,
        max_retry=3,
        enable_progress_bar=False,
    )
    executor.set_command_list(sac2sepc_cmds)
    executor.run_all(dry_run=dry_run)

    # 4) stop background thread
    stop_event.set()
    check_thread.join()

    # 5) final file count
    file_count = sum([len(files) for _, _, files in os.walk(segspec_dir)])
    print('\n')
    message = f"Done SAC2SPEC! {file_count}/{total_tasks} files written out.\n"
    dep_logger.info(message)