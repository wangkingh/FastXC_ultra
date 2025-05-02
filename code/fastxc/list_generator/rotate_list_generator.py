# list_generator/rotate_list_generator.py
from __future__ import annotations

import logging
from itertools import product
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)


# ────────────────────────────────────────── #
#  0. 依据 stack_flag 返回需旋转的 ENZ 目录   #
# ────────────────────────────────────────── #
def prepare_enz_dirs(stack_flag: str, output_dir: Path) -> list[Path]:
    """根据 3 位 stack_flag 组合出 linear / pws / tfpws ENZ 目录列表。"""
    flags = tuple(int(x) for x in stack_flag[:3])  # 只看前三位
    labels = ("linear", "pws", "tfpws")
    return [
        output_dir / "stack" / lbl
        for lbl, f in zip(labels, flags)
        if f
    ]


# ────────────────────────────────────────── #
#  1. 为单个 ENZ 目录生成 in/out 列表文件     #
# ────────────────────────────────────────── #
def _gen_rotate_list(
    enz_dir: Path,
    label: str,
    comp1: List[str],
    comp2: List[str],
    output_dir: Path,
) -> None:
    """
    对 enz_dir 下每个 <sta1_sta2> 目录生成::

        rotate_list/<label>/<sta_pair>/{enz_list.txt, rtz_list.txt}
    """
    rtz_ncf_dir   = output_dir / "stack" / f"rtz_{label}"
    rotate_root   = output_dir / "rotate_list" / label
    mapping       = {c: ch for c, ch in zip(comp1, "ENZ")}
    mapping.update({c: ch for c, ch in zip(comp2, "ENZ")})

    ENZ_order = [
        "E-E", "E-N", "E-Z",
        "N-E", "N-N", "N-Z",
        "Z-E", "Z-N", "Z-Z",
    ]
    RTZ_order = [
        "R-R", "R-T", "R-Z",
        "T-R", "T-T", "T-Z",
        "Z-R", "Z-T", "Z-Z",
    ]

    for sta_pair_dir in enz_dir.iterdir():
        if not sta_pair_dir.is_dir():
            continue

        sta_pair = sta_pair_dir.name
        if sum(1 for _ in sta_pair_dir.iterdir()) != 9:
            # 不是 9 个分量 → 跳过
            continue

        # ---------- 构造 ENZ 路径字典 -------------------------------- #
        enz_group: dict[str, str] = {}
        for c1, c2 in product(comp1, comp2):
            tag = f"{mapping[c1]}-{mapping[c2]}"
            fname = f"{sta_pair}.{c1}-{c2}.{label}.sac"
            enz_path = sta_pair_dir / fname
            enz_group[tag] = str(enz_path)

        # ---------- 输出文件 ---------------------------------------- #
        rotate_dir = rotate_root / sta_pair
        rotate_dir.mkdir(parents=True, exist_ok=True)

        in_list  = rotate_dir / "enz_list.txt"
        out_list = rotate_dir / "rtz_list.txt"

        # in_list
        with in_list.open("w") as fp:
            for tag in ENZ_order:
                fp.write(enz_group[tag] + "\n")

        # out_list
        with out_list.open("w") as fp:
            for tag in RTZ_order:
                out_path = rtz_ncf_dir / sta_pair / f"{sta_pair}.{tag}.ncf.sac"
                fp.write(str(out_path) + "\n")


# ────────────────────────────────────────── #
#  2. 外部入口                               #
# ────────────────────────────────────────── #
def gen_rotate_list(
    component_list1: List[str],
    component_list2: List[str],
    stack_flag: str,
    output_dir: str | Path,
) -> bool:
    """
    生成 `rotate_list/` 目录所需的 {enz_list, rtz_list} 文件。

    Returns
    -------
    bool
        True 生成成功 / False 无需生成。
    """
    output_dir = Path(output_dir).expanduser().resolve()
    enz_dirs   = prepare_enz_dirs(stack_flag, output_dir)

    if not enz_dirs:
        log.warning("No ENZ directories matched stack_flag '%s'.", stack_flag)
        return False

    for enz_dir in enz_dirs:
        if not enz_dir.is_dir():
            log.warning("ENZ directory not found: %s", enz_dir)
            continue

        label = enz_dir.name  # linear / pws / tfpws
        log.info("Generating rotate list for %s …", label)
        _gen_rotate_list(enz_dir, label, component_list1, component_list2, output_dir)

    return True
