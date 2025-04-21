from typing import List, Optional
import os
import glob
import logging
from .common_utils import write_cmd_list

logger = logging.getLogger(__name__)

##############################################################################
# 1) 构造公共参数（对应 C 代码中的 -A/-B/-O/-C/-D/-Z/-S/-G/-Q/-U/-T/-M）
##############################################################################


def build_param_set(
    ncf_dir: str,  # -O
    cclength: float,  # -C
    dist_range: str = None,  # -D (distmin/distmax)
    az_range: str = None,  # -Z (azmin/azmax)
    srcinfo_file: str = None,  # -S
    cpu_count: int = None,  # -T
    write_mode: str = None,  # -M (0=append, 1=aggregate)
) -> List[str]:
    """
    构造命令行选项列表，仅在参数非 None 时写入对应选项；
    若为 None，则使用与 C 代码相同的默认值，但不会在命令行中显式写出。
    """
    # print write_mode
    if write_mode == "APPEND":
        mode_val = 1
    elif write_mode == "AGGREGATE":
        mode_val = 2
    else:
        mode_val = 1
    param_set = []

    # --------------------------------------------
    # -O ncf_dir
    # --------------------------------------------
    if ncf_dir is not None:
        param_set.extend(["-O", ncf_dir])

    # --------------------------------------------
    # -C cclength
    # --------------------------------------------
    if cclength is not None:
        param_set.extend(["-C", str(cclength)])
    else:
        # 如果用户没指定，就用默认值，但不在命令行写
        # C 端自己会用 0.0f 做内部默认
        pass

    # --------------------------------------------
    # -D distmin/distmax
    # --------------------------------------------
    if dist_range is not None:
        param_set.extend(["-D", dist_range])
    else:
        # 默认 "0.0/400000.0"
        pass

    # --------------------------------------------
    # -Z azmin/azmax
    # --------------------------------------------
    if az_range is not None:
        param_set.extend(["-Z", az_range])
    else:
        # 默认 "0.0/360.0"
        pass

    # --------------------------------------------
    # -S source_info.txt
    # --------------------------------------------
    if srcinfo_file is not None:
        param_set.extend(["-S", srcinfo_file])

    # --------------------------------------------
    # -T cpu_count
    # --------------------------------------------
    if cpu_count is not None:
        param_set.extend(["-T", str(cpu_count)])
    else:
        # 默认 1
        pass

    # --------------------------------------------
    # -M write_mode
    # --------------------------------------------
    if mode_val is not None:
        param_set.extend(["-M", str(mode_val)])
    else:
        # 默认 0
        pass

    return param_set


##############################################################################
# 2) 构造输入文件对：保留你的“单/双阵列”逻辑，用于 -A/-B
##############################################################################


def build_input_sets(
    single_array: bool,
    array1_dir: Optional[str] = None,
    array2_dir: Optional[str] = None,
) -> List[List[str]]:
    """
    原先的业务逻辑：
    - single_array=True: A=A, B=A (自互相关)
    - single_array=False: 逐个匹配 array1 和 array2 中的同名 .speclist
      A=file_in_array1, B=file_in_array2
    若用户想手动指定 -A/-B，则可绕过这个函数。
    """
    # 如果你不想再用这种自动扫描逻辑，也可去掉本函数。
    input_sets = []

    if not array1_dir or not os.path.isdir(array1_dir):
        logger.warning(f"array1_dir not specified or not a directory: {array1_dir}")
        return input_sets

    speclist_1 = sorted(glob.glob(os.path.join(array1_dir, "*.speclist")))
    if not speclist_1:
        logger.warning(f"No .speclist found in {array1_dir}")
        return input_sets

    if single_array:
        # A=A, B=A
        for src in speclist_1:
            input_sets.append(["-A", src, "-B", src])
    else:
        # 双阵列匹配
        if not array2_dir or not os.path.isdir(array2_dir):
            logger.warning(f"array2_dir not specified or not a directory: {array2_dir}")
            return input_sets

        for f1 in speclist_1:
            fname = os.path.basename(f1)
            f2 = os.path.join(array2_dir, fname)
            if os.path.exists(f2):
                input_sets.append(["-A", f1, "-B", f2])

    return input_sets


##############################################################################
# 3) 生成最终命令行并写入文件
##############################################################################


def gen_xc_cmd(
    single_array: bool,
    xc_list_dir: str,
    xc_cmd_list: str,
    xc_exe: str,
    ncf_dir: str,
    cclength: float,
    dist_range: str,
    azimuth_range: str,
    srcinfo_file: str,
    cpu_count: int,
    write_mode: str,
) -> List[str]:
    """
    生成互相关命令列表，并写入到 xc_cmd_list 文件中。
    单/双阵列构建 -A/-B 文件对后，再与 build_param_set 得到的公共参数合并。
    """

    array1_dir = os.path.join(xc_list_dir, "array1")
    array2_dir = os.path.join(xc_list_dir, "array2")
    input_sets = build_input_sets(
        single_array=single_array,
        array1_dir=array1_dir,
        array2_dir=array2_dir,
    )

    # 若没能构建出任何 A/B 对，则返回空
    if not input_sets:
        logger.warning("No valid input sets constructed; no commands generated.")
        return []

    # 1) 构造公共参数（除 -A/-B）
    param_set = build_param_set(
        ncf_dir=ncf_dir,
        cclength=cclength,
        dist_range=dist_range,
        az_range=azimuth_range,
        srcinfo_file=srcinfo_file,
        cpu_count=cpu_count,
        write_mode=write_mode,
    )

    # 3) 合并每组 -A/-B 与公共参数
    cmd_list = []
    for each_input in input_sets:
        cmd = xc_exe + " " + " ".join(each_input + param_set)
        cmd_list.append(cmd)

    # 4) 写出到文件
    write_cmd_list(cmd_list, xc_cmd_list)
    return cmd_list
