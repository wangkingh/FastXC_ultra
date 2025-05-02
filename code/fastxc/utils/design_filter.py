# utils.py  (或专门的 filter 模块)
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from scipy.signal import butter

log = logging.getLogger(__name__)


def design_filter(
    delta: float,
    bands: str | Sequence[str],
    output_file: str | Path,
    order: int = 2,
) -> None:
    """
    设计 *整体 + 分段* 带通滤波器，并把系数写入文本文件。

    Parameters
    ----------
    delta : float
        采样间隔 (秒)，Fs = 1/delta.
    bands : str | Sequence[str]
        形如 ``"0.1/0.5 0.6/1"`` 或 ``["0.1/0.5","0.6/1"]``。
    output_file : Path | str
        写入文件路径；父目录不存在则自动创建。
    order : int, default 2
        butterworth 滤波器阶数。
    """
    output_file = Path(output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # -------- 1. 解析频带 ------------------------------------------ #
    if isinstance(bands, str):
        band_tokens = bands.split()
    else:
        band_tokens = list(bands)

    try:
        band_pairs = [tuple(map(float, b.split("/"))) for b in band_tokens]
    except ValueError as e:
        raise ValueError(f"Bad band specification: {bands}") from e

    fs   = 1.0 / delta
    fnyq = fs / 2.0

    # -------- 2. 校验 & 归一化 ------------------------------------- #
    overall_min = min(lo for lo, _ in band_pairs)
    overall_max = max(hi for _, hi in band_pairs)

    if not (0 < overall_min < overall_max < fnyq):
        raise ValueError(
            f"Invalid overall band {overall_min}/{overall_max} "
            f"for Nyquist={fnyq:.4f} Hz"
        )

    norm = lambda f: f / fnyq  # noqa: E731

    # -------- 3. 设计滤波器 --------------------------------------- #
    coeffs: list[tuple[str, Sequence[float], Sequence[float]]] = []

    # 3-1 整体带通
    b_all, a_all = butter(order, [norm(overall_min), norm(overall_max)], "bandpass")
    coeffs.append((f"{overall_min}/{overall_max}", b_all, a_all))

    # 3-2 各子带
    for lo, hi in band_pairs:
        b, a = butter(order, [norm(lo), norm(hi)], "bandpass")
        coeffs.append((f"{lo}/{hi}", b, a))

    # -------- 4. 写文件 ------------------------------------------- #
    try:
        with output_file.open("w") as fp:
            for tag, b, a in coeffs:
                fp.write(f"# {tag}\n")
                fp.write("\t".join(f"{x:.18e}" for x in b) + "\n")
                fp.write("\t".join(f"{x:.18e}" for x in a) + "\n")
        log.info("Filter written to %s", output_file)
    except OSError as e:
        log.error("Error writing filter file %s: %s", output_file, e)
        raise


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # quick test
    design_filter(
        delta=0.01,
        bands="0.2/0.5 0.6/0.8",
        output_file="./filter.txt",
    )
