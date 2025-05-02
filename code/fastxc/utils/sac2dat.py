from __future__ import annotations
"""Batch-convert cross-correlation SAC stacks to plain-text DAT files.

Differences to the original (ObsPy ≥1.3 tested)
-----------------------------------------------
*  **Dynamic branch selection** — if any `stack/*rtz*` dirs exist (case-insensitive),
   only those are processed; else all first-level dirs under `stack/` are processed.
*  **No global dat_list.txt** — each `dat/<branch>/` gets its own `dat_list.txt`.
*  **Pathlib only** — platform-independent path handling.
*  **Safe Windows launch** — guarded by ``if __name__ == "__main__"``.
"""

from pathlib import Path
import multiprocessing as mp
import logging
from functools import partial

import numpy as np
from obspy import read
from tqdm import tqdm


# ----------------------------------------------------------------------------- helpers
def _split_ccf_trace(
    data: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time axis (t >= 0), negative and positive halves of a symmetrical CCF.

    For *odd* npts the central sample is assigned to the *positive* half.
    """
    npts = data.size
    half = npts // 2  # integer division

    neg = data[:half][::-1]
    pos = data[half:]
    t = np.arange(pos.size) * dt
    return t, neg, pos


# ------------------------------------------------------------------- SAC → DAT pair generator
def gen_sac_dat_pairs(output_dir: Path, *, skip_self: bool = False):
    """Yield (sac_path, dat_path) tuples for every processing branch.

    If at least one branch name contains 'rtz' (case-insensitive),
    only those branches are processed.
    """
    stack_root = output_dir / "stack"
    dat_root = output_dir / "dat"

    # collect first-level branch dirs
    branches = [d for d in stack_root.iterdir() if d.is_dir()]
    rtz_branches = [d for d in branches if "rtz" in d.name.lower()]
    if rtz_branches:  # prefer *rtz* dirs if present
        branches = rtz_branches

    for proc_dir in branches:
        branch_dat_root = dat_root / proc_dir.name  # <out>/dat/<branch>

        for sac_path in proc_dir.rglob("*.sac"):
            if skip_self:
                sta_pair = sac_path.stem.split(".")[1]
                if "-" in sta_pair:
                    sta1, sta2 = sta_pair.split("-", 1)
                    if sta1 == sta2:
                        continue

            rel = sac_path.relative_to(proc_dir)
            dat_path = branch_dat_root / rel.with_suffix(".dat")
            dat_path.parent.mkdir(parents=True, exist_ok=True)
            yield sac_path, dat_path


# ----------------------------------------------------------------------------- conversion worker
def sac_to_dat(pair):  # path typing omitted for Pool speed
    """Convert a single SAC file to DAT format."""
    sac_path, dat_path = pair

    try:
        tr = read(str(sac_path))[0]
    except Exception as exc:
        logging.error("Failed to read %s – %s", sac_path, exc)
        return

    dt = tr.stats.delta

    try:
        sac_hdr = tr.stats.sac
        stla, stlo = sac_hdr.stla, sac_hdr.stlo or 0.0
        evla, evlo = sac_hdr.evla, sac_hdr.evlo or 0.0
        evel = 0.0
        stel = 0.0
    except AttributeError:
        logging.warning("Missing SAC header(s) in %s - skipped", sac_path)
        return

    t, neg, pos = _split_ccf_trace(tr.data.astype(float), dt)

    with dat_path.open("w") as fh:
        fh.write(f"{evlo:.7e} {evla:.7e} {evel:.7e}\n")
        fh.write(f"{stlo:.7e} {stla:.7e} {stel:.7e}\n")
        for tt, nn, pp in zip(t, np.pad(neg, (0, pos.size - neg.size)), pos):
            fh.write(f"{tt:.7e} {nn:.7e} {pp:.7e}\n")


def _update_progress(_: None, bar: tqdm):
    bar.update()


# --------------------------------------------------------------------------- list writers
def build_dat_lists(output_dir: Path):
    """Write per-branch dat_list.txt (excludes self-correlations)."""

    dat_root = output_dir / "dat"

    # pick branches with same 'rtz-only' rule as gen_sac_dat_pairs
    branches = [d for d in dat_root.iterdir() if d.is_dir()]
    rtz_branches = [d for d in branches if "rtz" in d.name.lower()]
    if rtz_branches:
        branches = rtz_branches

    for branch_dir in branches:
        list_file = branch_dir / "dat_list.txt"
        with list_file.open("w") as lf:
            for dat_path in branch_dir.rglob("*.dat"):
                sta_pair = dat_path.stem.split(".")[1]
                if "-" in sta_pair:
                    sta1, sta2 = sta_pair.split("-", 1)
                    if sta1 == sta2:  # skip self-correlations
                        continue
                rel = dat_path.relative_to(branch_dir)  # path relative to branch root
                lf.write(f"{rel}\n")


# ------------------------------------------------------------------------------- public API
def sac2dat_deployer(xc_param: dict):
    """Entry point - run the whole conversion pipeline."""

    output_dir = Path(xc_param["output_dir"]).resolve()
    cpu_count = int(xc_param.get("cpu_count", mp.cpu_count()))

    pairs = list(gen_sac_dat_pairs(output_dir))
    if not pairs:
        logging.warning("No SAC files found under %s/stack", output_dir)
        return

    bar = tqdm(total=len(pairs), desc="Converting SAC→DAT")
    with mp.Pool(processes=cpu_count) as pool:
        for pair in pairs:
            pool.apply_async(
                sac_to_dat, args=(pair,), callback=partial(_update_progress, bar=bar)
            )
        pool.close()
        pool.join()
    bar.close()

    build_dat_lists(output_dir)


# --------------------------------------------------------------------------- CLI launcher
if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2 or not Path(sys.argv[1]).is_file():
        print("Usage: python sac2dat.py <xc_param.json>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        params = json.load(f)

    sac2dat_deployer(params)
