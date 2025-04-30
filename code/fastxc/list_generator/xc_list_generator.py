# Purpose: Generating the xc_list dir for spec cross-correlation
from fastxc.SeisHandler import SeisArray
from datetime import datetime
import os
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def _write_speclist(files_group: dict, xc_list_dir: str):
    """
    files_group looks like:
    {
        (network_val, time_obj): {
            "station": [...],
            "year": [...],
            "jday": [...],
            "hour": [...],
            "minute": [...],
            "component": [...],
            "suffix": [...],
            "path": [file1, file2, ...]
        },
        ...
    }

    This function writes a .speclist file for each (network_val, time_obj) key in files_group.
    Each .speclist contains the file paths from the 'path' list, one path per line.
    The output files are stored in:
        xc_list_dir/<network_val>/<YYYY.jjj.HHMM>.speclist
    Where:
        - <network_val> is the network (array) name
        - <YYYY.jjj.HHMM> is year + Julian day + hour-minute
    """

    # print("files_group:", files_group)  # 可视化检查字典结构，必要时可移除

    for (network_val, time_obj), info_dict in files_group.items():
        paths = info_dict["path"]
        paths = sorted(paths)

        # 将 time_obj 转为字符串(YYYY.jjj.HHMM)，或直接转为 str(time_obj)
        time_info = time_obj.strftime("%Y.%j.%H%M")

        # 构造输出文件名和路径
        target_name = f"{time_info}.speclist"
        target_dir = os.path.join(xc_list_dir, network_val)
        os.makedirs(target_dir, exist_ok=True)

        target_xc_list_path = os.path.join(target_dir, target_name)

        # 写入 speclist 文件
        try:
            with open(target_xc_list_path, "w") as f:
                for p in paths:
                    f.write(p + "\n")
        except Exception as e:
            logger.error(f"Error writing file '{target_xc_list_path}': {str(e)}")


def gen_xc_list(segspec_dir: str, xc_list_dir: str, num_thread: int):
    """
    use SeisArray to generate xc_list dir for cross-correlation
    """
    message = f"Generating SEGSPEC Lists For Cross-Correlation\n"
    logger.info(message)

    extra_field = {
        "arrayID":  r"[A-Za-z0-9]+",
    }
    
    #  initialize SeisArray
    segspec_dir = os.path.abspath(segspec_dir)
    seis_array = SeisArray(
        array_dir=segspec_dir,
        pattern="{home}/{arrayID}/{*}/{network}.{station}.{YYYY}.{JJJ}.{HH}{MI}.{component}.{suffix}",
        custom_fields=extra_field,
    )
    # match files
    seis_array.match(threads=num_thread)

    # 5) write out spec_list based on dual_flag
    seis_array.group(labels=["arrayID", "time"], filtered=False)
    _write_speclist(seis_array.files_group, xc_list_dir)

    logger.debug(f"xc_list dir generated at {xc_list_dir}\n")
