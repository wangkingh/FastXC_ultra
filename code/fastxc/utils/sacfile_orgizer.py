# 假设 dataclass 已定义
from __future__ import annotations
from ..config_parser.schema import ArrayInfo
from typing import Optional, Tuple, List
from pathlib import Path
from fastxc.SeisHandler import SeisArray
from pandas import Timestamp
from typing import List
import logging

logger = logging.getLogger(__name__)


def read_station_list(sta_list_path: str) -> List[str]:
    station_list = []
    if sta_list_path != "NONE":
        with open(sta_list_path, "r") as f:
            for line in f:
                line = line.strip()
                # 如果是空行或以 # 开头的行都跳过
                if not line or line.startswith("#"):
                    continue
                station_list.append(line)
    return station_list


def read_time_list(time_list_path: str) -> List[Timestamp]:
    t_list = []
    if time_list_path != "NONE":
        with open(time_list_path, "r") as f:
            for line in f:
                line = line.strip()
                # 如果是空行或以 # 开头的行都跳过
                if not line or line.startswith("#"):
                    continue
                t_list.append(line)
        t_list = [Timestamp(x) for x in t_list]
    return t_list

def gen_seis_file_group(
    sac_dir,
    pattern,
    sta_list_path,
    time_list_path,
    component_list,
    time_range,
    cpu_count,
):
    """
    generating seis_file_group_list based on info_list (parsed from congig.ini file)
    """
    if sac_dir.upper() == "NONE":
        return [], [], []

    criteria = {}

    sta_list = read_station_list(sta_list_path)

    if sta_list:
        criteria["station"] = {"type": "list", "data_type": "str", "value": sta_list}
    else:
        logger.warning("Sta_list is empty, will not use as a criteria")

    time_list = read_time_list(time_list_path)

    if time_list:
        criteria["time"] = {"type": "list", "data_type": "datetime", "value": time_list}
        logger.info(f"Time_list {time_list} is used as a criteria")
    else:
        time_range = [Timestamp(time) for time in time_range]
        criteria["time"] = {
            "type": "range",
            "data_type": "datetime",
            "value": time_range,
        }
        logger.warning("Time_list is empty, will use time_range as a criteria")
        logger.info(f"Time_range {time_range} is used as a criteria")

    criteria["component"] = {
        "type": "list",
        "data_type": "str",
        "value": component_list,
    }

    # core function genrating seis_file_group_list
    seis_array = SeisArray(sac_dir, pattern)
    seis_array.match()

    seis_array.filter(criteria, threads=cpu_count, verbose=True)
    group_labels = ["station", "time"]
    
    # —— 动态决定 group 维度 ——
    group_labels = ["station", "time"]
    if "{network}" in pattern.lower():
        group_labels.append("network")          # 新增 network
        logger.info('Pattern contains "{network}", grouping by station-time-network.')
    else:
        logger.info('Pattern has no "{network}", grouping by station-time.')
        
    sorted_labels = ["component"]
    seis_array.group(labels=group_labels, sort_labels=sorted_labels, filtered=True)
    stas = seis_array.get_stations(filtered=True)
    times = seis_array.get_times(filtered=True)
    return seis_array.files_group, stas, times





# ----------------------------------------------------------------------------- #
def orgnize_sacfile(
    array1: ArrayInfo | dict,
    array2: Optional[ArrayInfo | dict],
    cpu_count: int,
) -> Tuple[
    List[str], List[str],          # stas1,   stas2
    List[str], List[str],          # times1,  times2
    List[str], List[str],          # files_group1, files_group2
]:
    """
    组织 SAC 文件并按阵列拆分。
    现在首选传入 ArrayInfo; 若仍传 dict 会自动转换，保持向后兼容。
    """

    # ---------------- 0) 构造 / 回退兼容 ---------------- #
    if isinstance(array1, dict):                        # ★ 兼容旧代码
        array1 = ArrayInfo.from_cfg(array1)

    if array2 is not None and isinstance(array2, dict): # ★
        array2 = ArrayInfo.from_cfg(array2)

    # ---------------- 1) 阵列-1 处理 ------------------- #
    sac_dir1: Path        = Path(array1.sac_dir).expanduser()
    pattern1              = array1.pattern
    sta_list_path1        = array1.sta_list
    time_list_path1       = array1.time_list
    component_list1       = array1.component_list
    time_range_1          = [array1.time_start, array1.time_end]

    files_group1, stas1, times1 = gen_seis_file_group(
        str(sac_dir1),               # 旧函数仍吃 str -> 显式转换
        pattern1,
        sta_list_path1,
        time_list_path1,
        component_list1,
        time_range_1,
        cpu_count,
    )

    # ---------------- 2) 阵列-2（可选） ---------------- #
    if array2 is None or array2.sac_dir == "NONE":
        stas2, times2, files_group2 = [], [], []
    else:
        sac_dir2: Path        = Path(array2.sac_dir).expanduser()
        pattern2              = array2.pattern
        sta_list_path2        = array2.sta_list
        time_list_path2       = array2.time_list
        component_list2       = array2.component_list
        time_range_2          = [array2.time_start, array2.time_end]

        files_group2, stas2, times2 = gen_seis_file_group(
            str(sac_dir2),
            pattern2,
            sta_list_path2,
            time_list_path2,
            component_list2,
            time_range_2,
            cpu_count,
        )

    # ---------------- 3) 返回 -------------------------- #
    return (
        stas1,
        stas2,
        times1,
        times2,
        files_group1,
        files_group2,
    )
