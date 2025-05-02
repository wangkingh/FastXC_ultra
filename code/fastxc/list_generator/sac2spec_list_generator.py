# list_generator/sac2spec_list_generator.py
from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import numpy as np
from pandas import Timestamp

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────── #
#  0. GPU 任务分配                                               #
# ─────────────────────────────────────────────────────────────── #
def distribute_tasks(
    gpu_list: Sequence[int],
    gpu_memory: Sequence[int],
    num_tasks: int,
) -> Dict[int, int]:
    """按显存大小比率把任务分配给 GPU。"""
    if len(gpu_list) != len(gpu_memory):
        raise ValueError("gpu_list and gpu_memory must have the same length")

    total_mem = sum(gpu_memory)
    gpu_tasks = {gid: int(np.floor(num_tasks * m / total_mem))
                 for gid, m in zip(gpu_list, gpu_memory)}

    # 把余数任务给显存最大那块 GPU
    remaining = num_tasks - sum(gpu_tasks.values())
    if remaining:
        largest_gpu = gpu_list[int(np.argmax(gpu_memory))]
        gpu_tasks[largest_gpu] += remaining

    assert sum(gpu_tasks.values()) == num_tasks
    return gpu_tasks


# ─────────────────────────────────────────────────────────────── #
#  1. 构造 SAC↔SPEC 路径对                                        #
# ─────────────────────────────────────────────────────────────── #
def build_sac_spec_pair(
    station: str,
    time: Timestamp,
    components: Sequence[str],
    paths: Sequence[str | Path],
    spec_dir: Path,
    array_id: str,
    network: str = "VV",
) -> Dict[str, List[str]]:
    """返回 {'sac': [...], 'spec': [...]} 两个等长列表。"""
    time_str   = time.strftime("%Y.%j.%H%M")
    sac_paths  = [str(Path(p).expanduser().resolve()) for p in paths]
    spec_paths = []

    for comp in components:
        spec_name   = f"{network}.{station}.{time_str}.{comp}.segspec"
        array_flag  = f"array{array_id}"
        full_path   = spec_dir / array_flag / time_str / spec_name
        spec_paths.append(str(full_path))

    return {"sac": sac_paths, "spec": spec_paths}


# ─────────────────────────────────────────────────────────────── #
#  2. 批量构造单阵列 SAC↔SPEC Pair                                #
# ─────────────────────────────────────────────────────────────── #
def build_sac_spec_pairs_for_group(
    seis_file_group: Dict[Tuple, Dict[str, List[str]]],
    spec_dir: Path,
    array_id: str,
    component_flag: int,
    placeholder_net: str = "VV",
) -> List[Dict[str, List[str]]]:
    """遍历 group 字典生成 pair 列表。"""
    results: list[Dict[str, List[str]]] = []
    if not seis_file_group:
        return results

    # key 有 2 或 3 元： (station, time[, network])
    sample_key = next(iter(seis_file_group))
    has_net    = len(sample_key) == 3

    for key, info in seis_file_group.items():
        station, time, *rest = key
        network  = rest[0] if has_net else placeholder_net
        comps    = info["component"]
        paths    = info["path"]

        if len(comps) != component_flag:
            log.warning(
                "[Group %s] station=%s time=%s: %d comps (expect %d). Skip.",
                array_id, station, time, len(comps), component_flag,
            )
            continue

        pair = build_sac_spec_pair(
            station, time, comps, paths,
            spec_dir=spec_dir, array_id=array_id, network=network,
        )
        results.append(pair)

    return results


# ─────────────────────────────────────────────────────────────── #
#  3. 外部入口：生成 sac_list_*.txt / spec_list_*.txt            #
# ─────────────────────────────────────────────────────────────── #
def gen_sac2spec_list(
    files_group1: Dict,
    files_group2: Dict | None,
    gpu_list: Sequence[int],
    gpu_memory: Sequence[int],
    component_list1: Sequence[str],
    component_list2: Sequence[str] | None,
    output_dir: str | Path,
) -> None:
    """
    根据两阵列文件分组，生成 *sac/spec 路径列表*，写入::

        <output_dir>/sac_spec_list/{sac|spec}_list_<gpu>.txt
    """
    if not files_group1 and not files_group2:
        log.warning("No SAC files in both groups – skip writing specs.")
        return
    if not files_group1:
        log.error("files_group1 is empty – abort.")
        sys.exit(1)

    output_dir      = Path(output_dir).expanduser().resolve()
    sac_spec_dir    = output_dir / "sac_spec_list"
    spec_dir        = output_dir / "segspec"
    sac_spec_dir.mkdir(parents=True, exist_ok=True)

    # ---------- build pairs --------------------------------------- #
    pairs1 = build_sac_spec_pairs_for_group(
        files_group1, spec_dir, "1", len(component_list1)
    )
    pairs2 = []
    if files_group2:
        pairs2 = build_sac_spec_pairs_for_group(
            files_group2, spec_dir, "2", len(component_list2 or [])
        )
    all_pairs = pairs1 + pairs2
    if not all_pairs:
        log.warning("No valid SAC/SPEC pairs generated – nothing to write.")
        return

    # ---------- distribute to GPUs -------------------------------- #
    gpu_tasks = distribute_tasks(gpu_list, gpu_memory, len(all_pairs))

    start = 0
    for gpu_id, n in gpu_tasks.items():
        sub_pairs = all_pairs[start : start + n]
        start += n

        sac_file  = sac_spec_dir / f"sac_list_{gpu_id}.txt"
        spec_file = sac_spec_dir / f"spec_list_{gpu_id}.txt"

        with sac_file.open("w") as fsac, spec_file.open("w") as fspec:
            for pair in sub_pairs:
                for s, p in zip(pair["sac"], pair["spec"]):
                    fsac.write(s + "\n")
                    fspec.write(p + "\n")

    log.info("Generated sac/spec list for %d GPUs under %s", len(gpu_tasks), sac_spec_dir)
