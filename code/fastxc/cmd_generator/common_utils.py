# cmd_generator/_common_utils.py
import os
import shutil
import logging
from typing import List

logger = logging.getLogger(__name__)


def write_cmd_list(cmd_list: List[str], xc_cmd_list: str) -> None:
    """
    将生成的命令列表写入文本文件。
    """
    os.makedirs(os.path.dirname(xc_cmd_list), exist_ok=True)
    with open(xc_cmd_list, "w") as f:
        f.write("\n".join(cmd_list))