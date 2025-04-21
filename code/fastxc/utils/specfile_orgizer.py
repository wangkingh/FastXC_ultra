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

def gen_spec_group(
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
    sorted_labels = ["component"]
    seis_array.group(labels=group_labels, sort_labels=sorted_labels, filtered=True)
    stas = seis_array.get_stations(filtered=True)
    times = seis_array.get_times(filtered=True)
    return seis_array.files_group, stas, times


def orgnize_sacfile(array_info1, array_info2, cpu_count):
    """
    process sacfiles
    """
    # step 0: extract array parameters from input
    sac_dir1 = array_info1["sac_dir"]
    pattern1 = array_info1["pattern"]
    sta_list_path1 = array_info1["sta_list"]
    time_start1 = array_info1["time_start"]
    time_end1 = array_info1["time_end"]
    time_list_path1 = array_info1["time_list"]
    component_list1 = array_info1["component_list"]

    sac_dir2 = array_info2["sac_dir"]
    pattern2 = array_info2["pattern"]
    sta_list_path2 = array_info2["sta_list"]
    time_start2 = array_info2["time_start"]
    time_end2 = array_info2["time_end"]
    time_list_path2 = array_info2["time_list"]
    component_list2 = array_info2["component_list"]

    time_range_1 = [time_start1, time_end1]
    time_range_2 = [time_start2, time_end2]

    # step 1: generating sac file list group
    files_group1, stas1, times1 = gen_seis_file_group(
        sac_dir1,
        pattern1,
        sta_list_path1,
        time_list_path1,
        component_list1,
        time_range_1,
        cpu_count,
    )
    files_group2, stas2, times2 = gen_seis_file_group(
        sac_dir2,
        pattern2,
        sta_list_path2,
        time_list_path2,
        component_list2,
        time_range_2,
        cpu_count,
    )

    return (
        stas1,
        stas2,
        times1,
        times2,
        files_group1,
        files_group2,
    )
