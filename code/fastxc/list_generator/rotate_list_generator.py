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
    if comp2 == []:
        comp2 = comp1
    if len(comp1) != 3 or len(comp2) != 3:
        log.error("Component list must be of length 3.")

    labels1 = list("ENZ")          # 第 0/1/2 个 → E/N/Z
    labels2 = list("ENZ")
    ENZ_order = [f"{a}-{b}" for a in labels1 for b in labels2]

    RTZ_order = [
        "R-R", "R-T", "R-Z",
        "T-R", "T-T", "T-Z",
        "Z-R", "Z-T", "Z-Z",
    ]
        
    rtz_ncf_dir   = output_dir / "stack" / f"rtz_{label}"
    rotate_root   = output_dir / "rotate_list" / label

    POS2CHR = ("E", "N", "Z")
    for net_sta_pair_dir in enz_dir.iterdir():
        if not net_sta_pair_dir.is_dir():
            continue

        net_sta_pair = net_sta_pair_dir.name
        if sum(1 for _ in net_sta_pair_dir.iterdir()) != 9:
            # 不是 9 个分量 → 跳过
            continue

        # ---------- 构造 ENZ 路径字典 -------------------------------- #
        enz_group: dict[str, str] = {}
        for i, c1 in enumerate(comp1):         # 外层：comp1 的顺序
            for j, c2 in enumerate(comp2):     # 内层：comp2 的顺序
                tag   = f"{POS2CHR[i]}-{POS2CHR[j]}"
                fname = f"{net_sta_pair}.{c1}-{c2}.{label}.sac"
                enz_path = net_sta_pair_dir / fname
                enz_group[tag] = str(enz_path)  # 写入字典

        # ---------- 输出文件 ---------------------------------------- #
        rotate_dir = rotate_root / net_sta_pair
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
                out_path = rtz_ncf_dir / net_sta_pair / f"{net_sta_pair}.{tag}.sac"
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
