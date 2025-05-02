# config_parser/schema.py
# ---------------------------------------------------------------------- #
"""Dataclass schemas for every INI section – *100 % dict-friendly*."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Mapping, Any

# ---------------------------------------------------------------------- #
# 共用工具
# ---------------------------------------------------------------------- #
def _as_bool(val: str | bool | Any, *, default: bool = False) -> bool:
    """将 'yes/true/1/on' 等字符串安全转换为 bool。"""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "on", "y"}
    return default


# ---------------------------------------------------------------------- #
# 1. ArrayInfo
# ---------------------------------------------------------------------- #
@dataclass
class ArrayInfo:
    sac_dir: str
    pattern: str
    time_start: str
    time_end: str
    time_list: str = "NONE"
    sta_list: str = "NONE"
    component_list: List[str] = field(default_factory=list)

    # ---------- factory ----------
    @classmethod
    def from_cfg(cls, g: Mapping[str, str]) -> "ArrayInfo":
        comps = [c.strip() for c in g.get("component_list", "").split(",") if c.strip()]
        return cls(
            sac_dir        = g.get("sac_dir", "NONE"),
            pattern        = g["pattern"],
            time_start     = g["time_start"],
            time_end       = g["time_end"],
            time_list      = g.get("time_list", "NONE"),
            sta_list       = g.get("sta_list", "NONE"),
            component_list = comps,
        )

    # ---------- validation ----------
    def validate(self) -> None:
        if self.sac_dir != "NONE" and not Path(self.sac_dir).exists():
            raise FileNotFoundError(f"SAC dir not found: {self.sac_dir}")

        for ts, lab in [(self.time_start, "time_start"),
                        (self.time_end,   "time_end")]:
            try:
                datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            except ValueError as e:
                raise ValueError(f"{lab} format error: {ts}") from e

        if not 1 <= len(self.component_list) <= 3:
            raise ValueError("component_list must contain 1–3 items")

    # ---------- helpers ----------
    def to_dict(self): return asdict(self)


# ---------------------------------------------------------------------- #
# 2. Preprocess
# ---------------------------------------------------------------------- #
@dataclass
class Preprocess:
    win_len:    int
    shift_len:  int
    delta:      float
    normalize:  str
    bands:      str
    whiten:     str = "OFF"
    skip_step:  str = "-1"

    @classmethod
    def from_cfg(cls, g: Mapping[str, str]) -> "Preprocess":
        return cls(
            win_len   = int(g["win_len"]),
            shift_len = int(g["shift_len"]),
            delta     = float(g["delta"]),
            normalize = g.get("normalize", "OFF").upper(),
            bands     = g.get("bands", ""),
            whiten    = g.get("whiten", "OFF").upper(),
            skip_step = g.get("skip_step", "-1"),
        )

    def validate(self) -> None:
        if self.normalize not in {"OFF", "RUN-ABS", "ONE-BIT", "RUN-ABS-MF"}:
            raise ValueError("invalid normalize value")

        if self.whiten not in {"OFF", "BEFORE", "AFTER", "BOTH"}:
            raise ValueError("invalid whiten value")

        if not re.fullmatch(r"-1|(\d+(,\d+)*)", self.skip_step):
            raise ValueError("skip_step must be -1 or comma-separated ints")

    def to_dict(self): return asdict(self)


# ---------------------------------------------------------------------- #
# 3. Xcorr   (-C/-M/-D/-Z/-S/-R)
# ---------------------------------------------------------------------- #
@dataclass
class Xcorr:
    max_lag:          int
    write_mode:       str = "APPEND"      # APPEND / AGGREGATE
    distance_range:   str = "-1/50000"
    azimuth_range:    str = "-1/360"
    source_info_file: str = "NONE"
    write_segment:    bool = False        # -R  True→1

    @classmethod
    def from_cfg(cls, g: Mapping[str, str]) -> "Xcorr":
        return cls(
            max_lag          = int(g["max_lag"]),
            write_mode       = g.get("write_mode", "APPEND").upper(),
            distance_range   = g.get("distance_range", "-1/50000"),
            azimuth_range    = g.get("azimuth_range", "-1/360"),
            source_info_file = g.get("source_info_file", "NONE"),
            write_segment    = _as_bool(g.get("write_segment", False)),
        )

    def validate(self) -> None:
        if self.write_mode not in {"APPEND", "AGGREGATE"}:
            raise ValueError("write_mode must be APPEND or AGGREGATE")
        self._check_range(self.distance_range, "distance_range")
        self._check_range(self.azimuth_range,  "azimuth_range")

        if self.source_info_file != "NONE" and not Path(self.source_info_file).is_file():
            raise FileNotFoundError(f"source_info_file not found: {self.source_info_file}")

    @staticmethod
    def _check_range(r: str, name: str):
        try:
            lo, hi = map(float, r.split("/"))
        except Exception:
            raise ValueError(f"{name} must be 'low/high', got '{r}'")
        if lo > hi:
            raise ValueError(f"{name}: lower {lo} > upper {hi}")

    def to_dict(self): return asdict(self)


# ---------------------------------------------------------------------- #
# 4. Stack   (-S/-B/-F)
# ---------------------------------------------------------------------- #
@dataclass
class Stack:
    stack_flag:       str
    sub_stack_size:   int = 1
    source_info_file: str = "NONE"

    @classmethod
    def from_cfg(cls, g: Mapping[str, str]) -> "Stack":
        return cls(
            stack_flag       = g["stack_flag"],
            sub_stack_size   = int(g.get("sub_stack_size", 1)),
            source_info_file = g.get("source_info_file", "NONE"),
        )

    def validate(self) -> None:
        if not re.fullmatch(r"[01]{3}", self.stack_flag):
            raise ValueError("stack_flag must be three binary digits")
        if self.sub_stack_size < 1:
            raise ValueError("sub_stack_size must be ≥1")
        if self.source_info_file != "NONE" and not Path(self.source_info_file).is_file():
            raise FileNotFoundError(f"source_info_file not found: {self.source_info_file}")

    def to_dict(self): return asdict(self)


# ---------------------------------------------------------------------- #
# 5. Executables
# ---------------------------------------------------------------------- #
@dataclass
class Executables:
    sac2spec: str
    xc:       str
    stack:    str
    rotate:   str

    @classmethod
    def from_cfg(cls, g): return cls(**g)

    def validate(self):
        for p in [self.sac2spec, self.xc, self.stack, self.rotate]:
            if not Path(p).is_file():
                raise FileNotFoundError(f"executable not found: {p}")

    def to_dict(self): return asdict(self)


# ---------------------------------------------------------------------- #
# 6. Device
# ---------------------------------------------------------------------- #
@dataclass
class Device:
    gpu_list:     List[int]
    gpu_task_num: List[int]
    gpu_mem_info: List[int]
    cpu_count:    int = 1

    @classmethod
    def from_cfg(cls, g):
        return cls(
            gpu_list     = [int(x) for x in g.get("gpu_list", "").split(",")     if x],
            gpu_task_num = [int(x) for x in g.get("gpu_task_num", "").split(",") if x],
            gpu_mem_info = [int(x) for x in g.get("gpu_mem_info", "").split(",") if x],
            cpu_count    = int(g.get("cpu_count", 1)),
        )

    def validate(self):
        if not (len(self.gpu_list) == len(self.gpu_task_num) == len(self.gpu_mem_info)):
            raise ValueError("gpu_list / gpu_task_num / gpu_mem_info length mismatch")
        if self.cpu_count < 1:
            raise ValueError("cpu_count must be ≥1")

    def to_dict(self): return asdict(self)


# ---------------------------------------------------------------------- #
# 7. Storage
# ---------------------------------------------------------------------- #
@dataclass
class Storage:
    output_dir: str
    overwrite:  bool = True
    clean_ncf:  bool = True
    write_mode: str = "APPEND"        # 供外部引用

    @classmethod
    def from_cfg(cls, g):
        return cls(
            output_dir = Path(g["output_dir"]).expanduser().resolve(),  # ★ 转 Path
            overwrite  = _as_bool(g.get("overwrite", True)),
            clean_ncf  = _as_bool(g.get("clean_ncf", True)),
            write_mode = g.get("write_mode", "APPEND").upper(),
        )

    def validate(self):
        Path(self.output_dir).expanduser().mkdir(parents=True, exist_ok=True)

    def to_dict(self): return asdict(self)


# ---------------------------------------------------------------------- #
# 8. Debug
# ---------------------------------------------------------------------- #
@dataclass
class Debug:
    dry_run: bool = False
    log_file_path: str = "NONE"

    # ---------- factory ----------
    @classmethod
    def from_cfg(cls, g: Mapping[str, str]) -> "Debug":
        return cls(
            dry_run       = _as_bool(g.get("dry_run", False)),
            log_file_path = g.get("log_file_path", "NONE"),
        )

    # ---------- validation ----------
    def validate(self, *, output_dir: str | Path) -> None:
        """
        `output_dir` 由 loader 在调用时塞进来，避免猜路径。
        只在 log_file_path == "NONE" 且 **非 dry_run** 时才真正创建文件。
        """
        out_dir = Path(output_dir).expanduser().resolve()

        if self.log_file_path == "NONE":
            ts   = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = out_dir / "log"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file_path = str(log_dir / f"fastxc-{ts}.log")

        if not self.dry_run:
            Path(self.log_file_path).touch(exist_ok=True)

    # ---------- helper ----------
    def to_dict(self): return asdict(self)
