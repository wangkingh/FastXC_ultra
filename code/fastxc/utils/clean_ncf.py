# cmd_generator/_common_utils.py
import os
import shutil
import logging

logger = logging.getLogger(__name__)

def prepare_ncf_dir(ncf_dir: str, clean_flag: bool) -> None:
    """
    准备输出目录。如果 clean_flag=True, 则删除并重建；否则复用现有目录。
    """
    if clean_flag:
        if os.path.exists(ncf_dir):
            shutil.rmtree(ncf_dir)
            logger.info(f"Output directory '{ncf_dir}' has been removed.")
        os.makedirs(ncf_dir, exist_ok=True)
        logger.info(f"Output directory '{ncf_dir}' has been created.")
    else:
        logger.info(f"Output directory '{ncf_dir}' has been used.")