# cmd_generator/stack_cmd.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)


def gen_stack_cmd(
    stack_exe: str | Path,
    output_dir: str | Path,
    stack_flag: str = "100",
    sub_stack_size: int = 1,        # ← 新增：对应 -B，<2 表示关闭
) -> List[str]:
    """
    生成 *stack* 可执行程序的命令列表，并写入::

        <output_dir>/cmd_list/stack_cmds.txt

    Parameters
    ----------
    stack_exe
        `ncf_pws` / `specxc_mg` 等可执行文件绝对路径。
    output_dir
        FastXC 主输出目录。
    stack_flag
        3 位字符串，对应线性 / PWS / TF-PWS，例如 ``"110"``。
    sub_stack_size
        对应命令行 ``-B``；取值:
        - ``>=2``  → 启用子叠加，先对每 *N* 道做 Linear Stack；
        - ``<2``   → 关闭（等价不写 ``-B``）。

    Returns
    -------
    list[str]
        生成的命令行列表；若 ``ncf/*.bigsac`` 不存在则返回空列表。
    """
    stack_exe = str(stack_exe)                       # 直接拼接进 cmd
    out_root  = Path(output_dir).expanduser().resolve()

    ncf_dir   = out_root / "ncf"
    stack_dir = out_root / "stack"
    cmd_dir   = out_root / "cmd_list"

    # ---------- 基本检查 ----------
    if not ncf_dir.is_dir():
        log.warning("NCF directory %s not found – nothing to stack.", ncf_dir)
        return []

    big_sacs = list(ncf_dir.glob("*.bigsac"))
    if not big_sacs:
        log.warning("No *.bigsac files under %s.", ncf_dir)
        return []

    # ---------- 确保输出目录 ----------
    stack_dir.mkdir(parents=True, exist_ok=True)
    cmd_dir.mkdir(parents=True,   exist_ok=True)

    # ---------- 组装命令 ----------
    base_opt = f"-O {stack_dir} -S {stack_flag}"
    if sub_stack_size >= 2:
        base_opt += f" -B {sub_stack_size}"

    cmd_list: list[str] = [
        f"{stack_exe} -I {bs} {base_opt}" for bs in big_sacs
    ]

    # ---------- 写入文件 ----------
    cmd_file = cmd_dir / "stack_cmds.txt"
    cmd_file.write_text("\n".join(cmd_list))
    log.info("Stack command list saved to %s (%d cmds)", cmd_file, len(cmd_list))

    return cmd_list
