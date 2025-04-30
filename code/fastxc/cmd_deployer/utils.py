import os
from typing import List
import logging
import threading
import time


def read_cmds_from_file(file_path: str) -> List[str]:
    try:
        with open(file_path, "r") as f:
            cmds = [line.strip() for line in f if line.strip()]
        return cmds
    except FileNotFoundError:
        logging.error(f"The file {file_path} was not found.")
        return []


def check_and_log_file_count(
    base_dir: str,
    stop_event: threading.Event,
    interval: int,
    total_tasks: int,
    logger_name: str = __name__,
):
    """
    Periodically check how many files are in 'base_dir' (recursively)
    and log the count with the given logger.
    """
    dep_logger = logging.getLogger(logger_name)

    while not stop_event.is_set():
        file_count = sum(len(files) for _, _, files in os.walk(base_dir))
        dep_logger.info(f"spectrums written out: " f"{file_count}/{total_tasks}")
        time.sleep(interval)
