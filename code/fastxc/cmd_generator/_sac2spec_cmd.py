from __future__ import annotations
from typing import List
import glob
from pathlib import Path
from typing import List, Union
import glob


# find min and max value in a string like '0.1/0.5 0.5/1.0 1.0/2.0'
def find_min_max_in_string(s):
    elements = s.split()

    numbers = []
    for element in elements:
        numbers.extend(map(float, element.split("/")))

    min_value = min(numbers)
    max_value = max(numbers)

    return f"{min_value}/{max_value}"

# ────────────────────────────────────────────────────────────── #
#  工具：把 "0,5,10,-1" / "0/5/10/-1" 统一转成 "0/5/10/-1"       #
# ────────────────────────────────────────────────────────────── #
def _normalize_skip(skip: str) -> str:
    skip = skip.strip()
    # 把逗号换成斜杠；去掉重复分隔符
    skip = skip.replace(",", "/").replace("//", "/")
    # 去掉可能的首尾空分隔
    parts = [p for p in skip.split("/") if p]
    return "/".join(parts)


# ────────────────────────────────────────────────────────────── #
def gen_sac2spec_cmd(
    component_num: int,                 # 1 或 3
    sac2spec_exe: Union[str, Path],     # 可传 Path
    output_dir: Union[str, Path],       # 工作目录
    win_len: int,                       # -L
    shift_len: int,                     # -S
    normalize: str,                     # 'OFF' | 'RUN-ABS-MF' | ...
    bands: str,                         # '0.1/0.5 0.5/1' …
    gpu_num: int,                       # GPU 任务数
    cpu_per_thread: int,                # 每 GPU CPU 线程
    whiten: str,                        # BEFORE / AFTER / BOTH / OFF
    skip_step: str,                     # "-1" 或 "0,5,10,-1" / "0/5/10/-1"
) -> List[str]:
    """
    生成 sac2spec 命令行并写 cmd 文件（cmd_list/sac2spec_cmds.txt）。

    Parameters
    ----------
    component_num : int
        1(只 Z)或 3(E,N,Z) 分量；
    sac2spec_exe : Path | str
        sac2spec 可执行文件路径
    output_dir : Path | str
        工作目录；将搜索 ./sac_spec_list/sac_list_*.txt
    win_len, shift_len : int
        片段长度与滑动步长 (秒)
    normalize : str
        'OFF' | 'RUN-ABS-MF' | 'ONE-BIT' | 'RUN-ABS'
    bands : str
        漂白/归一化频带；如 "0.1/0.5 0.5/1"
    gpu_num : int
        GPU 任务并发数；会写入 -U
    cpu_per_thread : int
        每个 GPU 绑定的 CPU 线程数；写入 -T
    whiten : str
        BEFORE / AFTER / BOTH / OFF
    skip_step : str
        形如 "-1" 或 "0,5,10,-1"；内部统一转成 "0/5/10/-1"

    Returns
    -------
    list[str] : 生成的全部命令行
    """

    # ---------- 路径和常量 ---------- #
    exe_path  = str(sac2spec_exe)
    out_dir   = Path(output_dir).expanduser().resolve()
    sac_dir   = out_dir / "sac_spec_list"
    filter_fp = out_dir / "filter.txt"

    whiten_code  = {"OFF": 0, "BEFORE": 1, "AFTER": 2, "BOTH": 3}[whiten]
    norm_code    = {"OFF": 0, "RUN-ABS-MF": 1, "ONE-BIT": 2, "RUN-ABS": 3}[normalize]

    whiten_band  = find_min_max_in_string(bands)   # 你已有的工具函数
    skip_fixed   = _normalize_skip(skip_step)      # ★ 新规范统一转旧格式

    common_flags = [
        "-L", str(win_len),
        "-S", str(shift_len),
        "-W", str(whiten_code),
        "-N", str(norm_code),
        "-F", whiten_band,
        "-Q", skip_fixed,
        "-B", str(filter_fp),
        "-U", str(gpu_num),
        "-T", str(cpu_per_thread),
    ]

    # ---------- 构造命令行 ---------- #
    cmd_lines: List[str] = []
    for sac_list in sorted(glob.glob(str(sac_dir / "sac_list_*.txt"))):
        gpu_id   = Path(sac_list).stem.split("_")[-1]
        spec_out = sac_dir / f"spec_list_{gpu_id}.txt"

        cmd_parts = [
            exe_path,
            "-I", sac_list,
            "-O", str(spec_out),
            "-C", str(component_num),
            "-G", gpu_id,
            *common_flags,
        ]
        cmd_lines.append(" ".join(cmd_parts))

    # ---------- 写出 cmd 文件 ---------- #
    cmd_dir  = out_dir / "cmd_list"
    cmd_dir.mkdir(parents=True, exist_ok=True)
    cmd_file = cmd_dir / "sac2spec_cmds.txt"
    cmd_file.write_text("\n".join(cmd_lines))

    return cmd_lines