# utils/concatenate_ncf.py
from __future__ import annotations

import concurrent.futures
import logging
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

log = logging.getLogger(__name__)


def concatenate_ncf(output_dir: str | Path, cpu_count: int) -> None:
    """
    把 ``<output_dir>/ncf/queue_*/*.bigsac`` 按文件名汇总合并到
    ``<output_dir>/ncf``，多 GPU 产出的碎片一次性拼接。

    Parameters
    ----------
    output_dir : Path | str
        FastXC 的 output 根目录。
    cpu_count  : int
        并行线程数。
    """
    output_dir   = Path(output_dir).expanduser().resolve()
    ncf_dir      = output_dir / "ncf"
    temp_ncf_dir = output_dir / "temp_ncf"

    if not ncf_dir.is_dir():
        log.warning("'ncf' directory not found: %s – skip concatenate.", ncf_dir)
        return

    # ---------- (re)create temp dir --------------------------------- #
    if temp_ncf_dir.exists():
        temp_ncf_dir.unlink(missing_ok=True) if temp_ncf_dir.is_file() else None
        temp_ncf_dir.rmdir() if temp_ncf_dir.is_dir() and not any(temp_ncf_dir.iterdir()) else None
    temp_ncf_dir.mkdir(exist_ok=True)

    # ---------- collect queue_* dirs -------------------------------- #
    gpu_subdirs: Sequence[Path] = [p for p in ncf_dir.iterdir()
                                   if p.is_dir() and p.name.startswith("queue_")]
    if not gpu_subdirs:
        log.info("No queue_* subdirectories under %s; nothing to do.", ncf_dir)
        return

    # ---------- map filename -> all chunks -------------------------- #
    log.info("Collecting .bigsac files from %d GPU subdirs …", len(gpu_subdirs))
    ncf_map: dict[str, list[Path]] = defaultdict(list)

    for gpu_dir in gpu_subdirs:
        for sfile in gpu_dir.glob("*.bigsac"):
            ncf_map[sfile.name].append(sfile)

    if not ncf_map:
        log.info("No .bigsac files found; skip concatenate.")
        return

    log.info("Found %d unique .bigsac names to merge.", len(ncf_map))

    # ---------- merge helper ---------------------------------------- #
    def merge_one(name: str, chunks: Sequence[Path]) -> None:
        target = temp_ncf_dir / name
        with target.open("ab") as out_f:
            for chunk in chunks:
                out_f.write(chunk.read_bytes())
                chunk.unlink()   # remove source

    # ---------- parallel merge -------------------------------------- #
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as pool:
        futures = [pool.submit(merge_one, fname, paths)
                   for fname, paths in ncf_map.items()]

        for _ in tqdm(concurrent.futures.as_completed(futures),
                      total=len(futures),
                      desc="[Concatenating NCF]",
                      unit="file"):
            pass

    # ---------- clean up queue dirs -------------------------------- #
    for gpu_dir in gpu_subdirs:
        try:
            gpu_dir.rmdir()      # 空目录直接删；若仍有残留文件，可用 shutil.rmtree
        except OSError:
            import shutil
            shutil.rmtree(gpu_dir, ignore_errors=True)

    # ---------- replace ncf with merged ----------------------------- #
    log.info("Replacing old 'ncf' with merged results.")
    backup_dir = output_dir / "ncf_old"
    if ncf_dir.exists():
        ncf_dir.rename(backup_dir)
    temp_ncf_dir.rename(ncf_dir)
    if backup_dir.exists():
        # 若需要保留备份，可注释下一行
        import shutil
        shutil.rmtree(backup_dir, ignore_errors=True)

    log.info("NCF concatenation finished.\n")
