import subprocess
from logging.handlers import RotatingFileHandler
from tqdm import tqdm
from queue import Queue, Empty
from collections import deque          # 顶部 import
from typing import Dict, List, Tuple
import threading
import logging
import time

logger = logging.getLogger(__name__)


# ------------------ A Simple Progress Tracker Helper ------------------
class ProgressTracker:
    """
    Wraps a tqdm progress bar for counting total tasks.
    """

    def __init__(self, total_tasks: int, description: str = "Progress"):
        self.total_tasks = total_tasks
        self.tqdm_bar = tqdm(total=total_tasks, desc=f"[{description}]", unit="task")

    def update(self):
        self.tqdm_bar.update(1)

    def close(self):
        self.tqdm_bar.close()


class DummyProgressTracker:
    """
    A no-op progress tracker that does nothing, for when we disable progress bar.
    """

    def __init__(self, total_tasks: int, description: str = "Progress"):
        self.total_tasks = total_tasks
        self.tqdm_bar = type("FakeTqdmBar", (), {"n": 0})()

    def update(self):
        pass

    def close(self):
        pass


# ------------------ The main class: MultiDeviceTaskExecutor ------------------
class MultiDeviceTaskExecutor:
    """_summary_
    This class is responsible for executing tasks on multiple GPUs or CPU.
    main idea:
    1) use device_config to tell the usage of devices and the number of workers
    2) create corresponding queues (device_type, device_id, worker_id) -> Queue) for each worker
    3) take tasks from the global task queue and put them into the sub_queue by the dispatcher
    4) the progress tracker will track the progress of all tasks
    5) use logger to record the log of each task and use tqdm to display the progress
    6) the main entry is run_all() method

    e.g. only use GPU:
    executor = MultiDeviceTaskExecutor.from_gpu_pool(
            gpu_ids=[0,1],
            gpu_workers=[2,2],
            log_file_path="my_gpu_log.log"
        )

    only use CPU:
    executor = MultiDeviceTaskExecutor.from_threadpool(8)

    mix use CPU and GPU:
    devices_config = {
            "cpu": { 0: 4 },
            "gpu": { 0: 2, 1: 2 }
        }
        executor = MultiDeviceTaskExecutor(devices_config=devices_config)
    """

    def __init__(
        self,
        devices_config: Dict[str, Dict[int, int]],
        log_file_path: str = "task.log",
        task_description: str = "Task",
        queue_size: int = 1,
        max_retry: int = 3,
        build_type: str = "with_worker_id",
        enable_progress_bar: bool = True,
    ):
        """_summary_

        Args:
            gpu_list (List[int]): list of gpu ids to be used.
            gpu_task_num (List[int]): number of tasks to be executed on each
            log_file_path (str, optional): logger file. Defaults to 'my_log.log'.
            max_retry (int, optional): max retry. Defaults to 3.
        """
        self.devices_config = devices_config
        self.log_file_path = log_file_path
        self.queue_size = queue_size
        self.task_description = task_description
        self.max_retry = max_retry

        # 1) Initialize logger
        self.logger = self._setup_logger()

        # 2) Create the global task queue (store all tasks)
        self.global_task_queue = Queue()

        
        # 3) Create the sub_queue, store tasks for each (device_type, device_id, worker_id)
        self._rr_keys = deque()         # ← 新增
        # DEBUG 2015-05-03 给与不同GPU上的worker_id不同的名字，即使是同
        self.worker_gid_map = {}   # ➊
        ##############################################################
        
        self.sub_queue = self._create_sub_queue()

        # 4) Attributes to manage the threads and tasks
        self._dispatcher_thread = None
        self.sub_worker_threads = []
        self.progress_thread = None
        self.command_list = []
        self.total_tasks = 0

        # cmd build type
        self.build_type = build_type

        # progress tracker
        self.progress_tracker = None

        # enable progress bar
        self.enable_progress_bar = enable_progress_bar

        # for success/failure counting
        self.success_count = 0
        self.failure_count = 0
        self.counter_lock = threading.Lock()
        


    # ------------------ Simplified constructor: only GPU ------------------
    @classmethod
    def from_gpu_pool(
        cls,
        gpu_ids: List[int],
        gpu_workers: List[int],
        log_file_path: str = "task.log",
        task_description: str = "Task",
        queue_size: int = 1,
        max_retry: int = 3,
        build_type: str = "with_worker_id",
        enable_progress_bar: bool = True,
    ):
        """
        only use GPU
        :param gpu_ids: list of gpu ids to be used. e.g. [0, 1]
        :param gpu_workers: number of tasks to be executed on each GPU. e.g. [2, 2]
        """
        if len(gpu_ids) != len(gpu_workers):
            raise ValueError("gpu_ids do not match gpu_workers!")

        devices_config = {"gpu": {}}
        for gid, wnum in zip(gpu_ids, gpu_workers):
            devices_config["gpu"][gid] = wnum

        return cls(
            devices_config=devices_config,
            log_file_path=log_file_path,
            task_description=task_description,
            queue_size=queue_size,
            max_retry=max_retry,
            build_type=build_type,
            enable_progress_bar=enable_progress_bar,
        )

    # ------------------ Simplified constructor: only CPU ------------------
    @classmethod
    def from_threadpool(
        cls,
        num_threads: int,
        log_file_path: str = "task.log",
        task_description: str = "Task",
        queue_size: int = 1,
        max_retry: int = 3,
        enable_progress_bar: bool = True,
    ):
        """
        only use CPU, specify the number of threads
        """
        devices_config = {"cpu": {0: num_threads}}
        return cls(
            devices_config=devices_config,
            log_file_path=log_file_path,
            task_description=task_description,
            queue_size=queue_size,
            max_retry=max_retry,
            enable_progress_bar=enable_progress_bar,
        )

    # ------------------ 1) Initialize Logger ------------------
    def _setup_logger(self) -> logging.Logger:
        """
        Create a logger with rotating file handler.
        """
        logger = logging.getLogger("executor_logger")
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            handler = RotatingFileHandler(
                self.log_file_path, maxBytes=1048576, backupCount=5
            )
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    # 2) Build sub_queues
    def _create_sub_queue(self):
        sub_queues = {}
        gid_counter = 0                     # ➋

        for device_type, device_dict in self.devices_config.items():
            for dev_id, worker_num in device_dict.items():
                for worker_id in range(worker_num):
                    key = (device_type, dev_id, worker_id)
                    sub_queues[key] = Queue(maxsize=self.queue_size)
                    self._rr_keys.append(key)
                    self.worker_gid_map[key] = gid_counter
                    gid_counter += 1
        return sub_queues

    # 3) set the command list
    def set_command_list(self, command_list: List[str]):
        self.command_list = command_list
        self.total_tasks = len(command_list)

    # --------------- Public Methods ---------------
    def run_all(self, dry_run=False):
        """
        Main entry: put tasks into global_task_queue, start the dispatcher + sub_workers
        track progress, and wait for all tasks to finish.
        """
        if not self.command_list:
            self.logger.error("Command list is empty.")
            raise ValueError("Command list is empty.")

        # 1) Create a progress tracker (or dummy), dry run will disable progress bar
        if self.enable_progress_bar and not dry_run:
            self.progress_tracker = ProgressTracker(
                self.total_tasks, description=self.task_description
            )
        else:
            self.progress_tracker = DummyProgressTracker(
                self.total_tasks, description=self.task_description
            )

        # 2) Put all tasks into the global_task_queue
        for cmd in self.command_list:
            self.global_task_queue.put((cmd, self.max_retry))

        # 3) Start the dispatcher thread
        self._dispatcher_thread = threading.Thread(target=self._dispatcher, daemon=True)
        self._dispatcher_thread.start()

        # 4) Start the sub_worker threads
        for (device_type, dev_id, worker_id), sub_queue in self.sub_queue.items():
            thread = threading.Thread(
                target=self._sub_worker_thread_fn,
                args=(device_type, dev_id, worker_id, sub_queue, dry_run),
                daemon=True,
            )
            self.sub_worker_threads.append(thread)
            thread.start()

        # 5) Start progress thread
        if not isinstance(self.progress_tracker, DummyProgressTracker):
            self.progress_thread = threading.Thread(
                target=self._periodic_progress_display, daemon=True
            )
            self.progress_thread.start()

        # 6) Wait for everything to finish
        self._wait_for_finish()

    # --------------- Internal Methods ---------------
    def _dispatcher(self):
        while True:
            try:
                cmd, retry = self.global_task_queue.get(timeout=1)
            except Empty:
                if self.global_task_queue.empty():
                    break
                continue

            for _ in range(len(self._rr_keys)):
                key = self._rr_keys[0]
                sub_q = self.sub_queue[key]
                self._rr_keys.rotate(-1)
                if not sub_q.full():
                    sub_q.put((cmd, retry))
                    self.global_task_queue.task_done()   # 成功转移 −1
                    break
            else:
                time.sleep(1)
                self.global_task_queue.put((cmd, retry)) # 放回 +1
                self.global_task_queue.task_done()       # 再 −1，计数归位

    # --------------- Internal Methods: Sub Worker ---------------
    def _sub_worker_thread_fn(
        self,
        device_type: str,
        dev_id: int,
        worker_id: int,
        sub_queue: Queue,
        dry_run=False,
    ):
        """_summary_
        Worker function for each sub_queue.
        """
        while True:
            try:
                cmd, retry = sub_queue.get(timeout=1)
            except Empty:
                if sub_queue.empty():
                    break
                else:
                    continue

            # Execute the command
            extended_cmd = self._build_command_by_device(
                cmd, device_type, dev_id, worker_id
            )
            return_code = self._execute_cmd(extended_cmd, dry_run)

            if return_code == 0:
                # Update the progress tracker
                self.progress_tracker.update()
                with self.counter_lock:
                    self.success_count += 1
            else:
                if retry > 0:
                    # Retry
                    sub_queue.put((cmd, retry - 1))
                else:
                    self.logger.error(f"Task failed: {cmd}")
                    self.progress_tracker.update()
                    with self.counter_lock:  # update failure count
                        self.failure_count += 1
            sub_queue.task_done()

    def _build_command_by_device(
        self, cmd: str, device_type: str, dev_id: int, worker_id: int
    ) -> str:
        """
        generate the command based on the device type + self.build_type
        """
        concurrency = self.devices_config[device_type][dev_id]
        if device_type.lower() == "gpu":
            if self.build_type == "with_worker_id":
                gid = self.worker_gid_map[(device_type, dev_id, worker_id)]
                return f"{cmd} -G {dev_id} -U {concurrency} -Q {gid}"

            elif self.build_type == "no_worker_id":
                # e.g. add -G 0
                return f"{cmd} -G {dev_id}"
            else:
                return f"{cmd} -G {dev_id}"
        elif device_type.lower() == "cpu":
            # add nothing
            return cmd
        else:
            # add nothing
            return cmd

    # --------------- Internal Methods: Execute Command ---------------
    def _execute_cmd(self, cmd: str, dry_run=False) -> int:
        """_summary_
        Execute the command on the specified GPU.
        """
        if dry_run:
            # self.logger.info(f"[DryRun] Would execute: {cmd}")
            print(f"[DryRun] Would execute: {cmd}")
            return 0
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,  # stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,  # stderr=subprocess.DEVNULL,
                text=True,
            )
            #if result.stdout:
            #     self.logger.info(result.stdout.strip())
            #if result.stderr:
            #     self.logger.error(result.stderr.strip())

            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)

            return 0
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error processing command: {cmd}, Error: {e}")
            return -1
        except Exception as e:
            self.logger.error(f"Error processing command: {cmd}, Error: {e}")
            return -1

    # --------------- Internal Methods: Periodic Progress Display ---------------
    def _periodic_progress_display(self):
        """
        A background thread that periodically checks if progress == total_tasks.
        Once all tasks are done, close the progress bar.
        """
        while self.progress_tracker.tqdm_bar.n < self.progress_tracker.total_tasks:
            time.sleep(2)
        self.progress_tracker.close()

    # ------------------ Internal: Wait for all threads & queues ------------------
    def _wait_for_finish(self):
        """
        Wait for dispatcher, sub-workers, global queue, and sub-queues to finish.
        """
        try:
            # Wait dispatcher
            self._dispatcher_thread.join()
            # Wait sub-workers
            for t in self.sub_worker_threads:
                t.join()

            # Wait for queues
            self.global_task_queue.join()
            for sq in self.sub_queue.values():
                sq.join()

            logger.debug(f"All tasks Done.")
            with self.counter_lock:
                logger.debug(
                    f"Success: {self.success_count}, Fail: {self.failure_count}"
                )
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt detected, cleaning up...")
        finally:
            # Ensure progress thread ends
            if self.progress_thread is not None:
                self.progress_thread.join()
            logger.debug("All threads cleaned up.")


# ------------------ Usage Examples ------------------
if __name__ == "__main__":
    import tempfile
    import os

    # 使用 tempfile.TemporaryDirectory() 创建一个临时文件夹
    # 在这个 with 块结束后，文件夹与其中的文件会自动删除
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Temporary directory created: {tmp_dir}")

        # ==================== 1) 只想用 GPU 并行 ====================
        # 这里假设机器上有 GPU，可以用 nvidia-smi 获取信息
        executor_gpu = MultiDeviceTaskExecutor.from_gpu_pool(
            gpu_ids=[0, 1],
            gpu_workers=[1, 1],
            log_file_path=os.path.join(tmp_dir, "gpu_executor.log"),
            task_description="GPU Tasks",
            max_retry=2,
            build_type="no_worker_id",  # 不带 worker_id
        )

        # 准备一些命令：获取系统信息 (nvidia-smi)，或者简单 echo
        gpu_cmd1 = f"echo 'GPU Hello 0' >> {os.path.join(tmp_dir, 'gpu_hello0.txt')}"
        gpu_cmd2 = f"nvidia-smi >> {os.path.join(tmp_dir, 'gpu_info.txt')}"
        gpu_cmd3 = (
            f"echo 'Another GPU Info' >> {os.path.join(tmp_dir, 'gpu_info2.txt')}"
        )

        executor_gpu.set_command_list([gpu_cmd1, gpu_cmd2, gpu_cmd3])
        executor_gpu.run_all()

        # ==================== 2) 只想用 CPU 并行 (8 个线程) ====================
        executor_cpu = MultiDeviceTaskExecutor.from_threadpool(
            num_threads=8,
            log_file_path=os.path.join(tmp_dir, "cpu_executor.log"),
            task_description="CPU Tasks",
            max_retry=1,
            enable_progress_bar=True,
        )

        # 准备一些命令：获取 CPU 系统信息, uname -a, echo
        cpu_cmd1 = f"echo 'CPU Hello 1' >> {os.path.join(tmp_dir, 'cpu_hello_1.txt')}"
        cpu_cmd2 = f"uname -a >> {os.path.join(tmp_dir, 'cpu_sysinfo.txt')}"
        cpu_cmd3 = (
            f"echo 'Some additional info' >> {os.path.join(tmp_dir, 'cpu_info2.txt')}"
        )

        executor_cpu.set_command_list([cpu_cmd1, cpu_cmd2, cpu_cmd3])
        executor_cpu.run_all()

        # ==================== 3) 混合 CPU + GPU ====================
        devices_config = {
            "cpu": {0: 2},  # 2个CPU并发
            "gpu": {0: 1, 1: 1},  # GPU0开1并发，GPU1开1并发
        }
        hybrid_log_path = os.path.join(tmp_dir, "hybrid_executor.log")

        executor_hybrid = MultiDeviceTaskExecutor(
            devices_config=devices_config,
            log_file_path=hybrid_log_path,
            task_description="Hybrid CPU+GPU Tasks",
            queue_size=1,
            max_retry=2,
            build_type="with_worker_id",  # GPU 命令带 worker_id
            enable_progress_bar=True,
        )

        # 简单的混合命令：一些放CPU跑 (echo, uname -a)，一些放GPU跑 (nvidia-smi)
        hybrid_cmds = [
            f"echo 'Hybrid Task #1' >> {os.path.join(tmp_dir, 'hybrid_output1.txt')}",
            f"nvidia-smi >> {os.path.join(tmp_dir, 'hybrid_gpu_info.txt')}",
            f"uname -a >> {os.path.join(tmp_dir, 'hybrid_cpu_info.txt')}",
            f"echo 'Hybrid Task #4' >> {os.path.join(tmp_dir, 'hybrid_output2.txt')}",
        ]
        executor_hybrid.set_command_list(hybrid_cmds)
        executor_hybrid.run_all()

        # 如果需要查看这些临时文件的内容，可以在这里读取并打印
        print("\n====== Check generated files in temp dir ======")
        for filename in sorted(os.listdir(tmp_dir)):
            file_path = os.path.join(tmp_dir, filename)
            print(f"\n=== Contents of {file_path} ===")
            with open(file_path, "r", encoding="utf-8") as f:
                print(f.read())

    # 离开 with 块后，tmp_dir 及其内容会自动被删除
    print("All temporary files removed. Script end.")
