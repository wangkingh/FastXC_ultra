# list_generator/xc_list_generator.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from fastxc.SeisHandler import SeisArray
from pandas import Timestamp
from tqdm import tqdm  # 仍在其它地方用到，可保留

log = logging.getLogger(__name__)


# ────────────────────────────────────────── #
#  写单个 <YYYY.jjj.HHMM>.speclist          #
# ────────────────────────────────────────── #
def _write_speclist(
    files_group: Dict[Tuple[str, Timestamp], Dict[str, List[str]]],
    xc_list_dir: Path,
) -> None:
    """
    Parameters
    ----------
    files_group : {(arrayID, Timestamp): {"path": [...], ...}, ...}
        来自 ``SeisArray.group(labels=["arrayID","time"])`` 的结果。
    xc_list_dir : Path
        输出根目录，最终文件位于::

            xc_list_dir/<arrayID>/<YYYY.jjj.HHMM>.speclist
    """
    for (array_id, time_obj), info in files_group.items():
        paths = sorted(info["path"])
        time_str = time_obj.strftime("%Y.%j.%H%M")

        target_dir  = xc_list_dir / array_id
        target_file = target_dir / f"{time_str}.speclist"
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            with target_file.open("w") as fp:
                fp.write("\n".join(paths) + "\n")
        except OSError as e:
            log.error("Write <%s> failed: %s", target_file, e)


# ────────────────────────────────────────── #
#  公用入口                                  #
# ────────────────────────────────────────── #
def gen_xc_list(
    segspec_dir: str | Path,
    xc_list_dir: str | Path,
    num_thread: int = 4,
) -> None:
    """
    读取 ``segspec_dir`` 下的谱文件，生成交叉相关用的 ``*.speclist`` 路径集合。

    Notes
    -----
    * 调用方保证 `fastxc.SeisHandler.SeisArray` 可用。
    * 输出目录若存在将追加/覆盖同名文件。
    """
    segspec_dir = Path(segspec_dir).expanduser().resolve()
    xc_list_dir = Path(xc_list_dir).expanduser().resolve()
    xc_list_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating SEGSPEC lists for cross-correlation …")

    # ------ 1. 组装 SeisArray ----------------------------------- #
    extra_field = {"arrayID": r"[A-Za-z0-9]+"}
    seis = SeisArray(
        array_dir   = segspec_dir,
        pattern     = "{home}/{arrayID}/{*}/{network}.{station}.{YYYY}.{JJJ}.{HH}{MI}.{component}.{suffix}",
        custom_fields = extra_field,
    )

    # ------ 2. 扫描匹配 ----------------------------------------- #
    seis.match(threads=num_thread)

    # ------ 3. 分组 & 写文件 ------------------------------------ #
    seis.group(labels=["arrayID", "time"], filtered=False)
    _write_speclist(seis.files_group, xc_list_dir)

    log.info("xc_list generated under %s\n", xc_list_dir)
