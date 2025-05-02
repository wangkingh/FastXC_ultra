# xc_cmd_generator.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import glob
import logging

from .common_utils import write_cmd_list

logger = logging.getLogger(__name__)

# ───────────────────────────────────────── build_param_set ──────────────────────────────────
def build_param_set(
    ncf_dir: str | Path,                    # -O
    cclength: float | None = None,          # -C
    dist_range: str | None = None,          # -D  形如 "0/500"
    az_range: str | None = None,            # -Z  形如 "0/360"
    srcinfo_file: str | Path | None = None, # -S
    cpu_count: int | None = None,           # -T
    write_mode: str | None = "APPEND",      # -M  (APPEND / AGGREGATE)
    save_segment: bool | None = None,       # -R  True→1  False→0  None→不写
) -> List[str]:
    """
    根据参数构造互相关命令行公共选项列表。None 表示不显式写出，让 C 端自行取默认值。
    """
    p: list[str] = ["-O", str(Path(ncf_dir).expanduser().resolve())]

    if cclength is not None:
        p += ["-C", str(cclength)]

    if dist_range:
        p += ["-D", dist_range]

    if az_range:
        p += ["-Z", az_range]

    if srcinfo_file:
        p += ["-S", str(Path(srcinfo_file).expanduser().resolve())]

    if cpu_count:
        p += ["-T", str(cpu_count)]

    mode_val = {"APPEND": 1, "AGGREGATE": 2}.get(str(write_mode).upper(), 1)
    p += ["-M", str(mode_val)]

    if save_segment is not None:
        p += ["-R", "1" if save_segment else "0"]

    return p


# ───────────────────────────────────────── build_input_sets ────────────────────────────────
def build_input_sets(
    single_array: bool,
    array1_dir: str | Path | None = None,
    array2_dir: str | Path | None = None,
) -> List[List[str]]:
    """
    自动扫描 *.speclist，拼出 (-A, -B) 组合。
      · 单阵列:  A=list  B=list  (自互相关)
      · 双阵列:  按文件名匹配 array1 与 array2
    """
    in_sets: list[list[str]] = []

    array1_dir = Path(array1_dir or "").expanduser()
    if not array1_dir.is_dir():
        logger.warning("array1_dir not found: %s", array1_dir)
        return in_sets

    speclist_1 = sorted(array1_dir.glob("*.speclist"))
    if not speclist_1:
        logger.warning("No .speclist in %s", array1_dir)
        return in_sets

    if single_array:
        for f in speclist_1:
            in_sets.append(["-A", str(f), "-B", str(f)])
    else:
        array2_dir = Path(array2_dir or "").expanduser()
        if not array2_dir.is_dir():
            logger.warning("array2_dir not found: %s", array2_dir)
            return in_sets

        for f1 in speclist_1:
            f2 = array2_dir / f1.name
            if f2.exists():
                in_sets.append(["-A", str(f1), "-B", str(f2)])

    return in_sets


# ───────────────────────────────────────── gen_xc_cmd ──────────────────────────────────────
def gen_xc_cmd(
    *,
    single_array: bool,
    xc_list_dir: str | Path,
    xc_cmd_list: str | Path,
    xc_exe: str | Path,
    ncf_dir: str | Path,
    cclength: float,
    dist_range: str,
    azimuth_range: str,
    srcinfo_file: str | Path,
    cpu_count: int,
    write_mode: str,
    save_segment: bool,          # ← 新增：写 -R
) -> List[str]:
    """
    生成互相关命令并写入文件；返回命令字符串列表。
    """
    xc_list_dir = Path(xc_list_dir).expanduser()
    array1_dir  = xc_list_dir / "array1"
    array2_dir  = xc_list_dir / "array2"

    input_sets = build_input_sets(
        single_array=single_array,
        array1_dir=array1_dir,
        array2_dir=array2_dir,
    )
    if not input_sets:
        logger.warning("No input sets; xc_cmd generation skipped.")
        return []

    param_set = build_param_set(
        ncf_dir      = ncf_dir,
        cclength     = cclength,
        dist_range   = dist_range,
        az_range     = azimuth_range,
        srcinfo_file = srcinfo_file,
        cpu_count    = cpu_count,
        write_mode   = write_mode,
        save_segment = save_segment,     # 传入 -R
    )

    cmd_list = [
        f"{xc_exe} " + " ".join(each_input + param_set) for each_input in input_sets
    ]

    write_cmd_list(cmd_list, Path(xc_cmd_list))
    return cmd_list
