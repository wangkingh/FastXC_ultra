# config_parser/loader.py
# -----------------------------------------------------------------------------
"""High-level loader that turns an INI file into a validated Config object.

Usage
-----
>>> from config_parser.loader import Config
>>> cfg = Config("config.ini")   # 读取
>>> cfg.validate_all()           # 全局校验
>>> cfg.stack.sub_stack_size = 8 # 直接修改
>>> cfg.to_ini("new.ini")        # 回写

Attributes
----------
Each INI section becomes a dataclass attribute, e.g.::

    cfg.array1     -> ArrayInfo
    cfg.preprocess -> Preprocess
    cfg.xcorr      -> Xcorr
    cfg.stack      -> Stack
    ...
You may also use dict-like access: ``cfg["stack"]``.
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Mapping, Optional, Iterator

import configparser

# 导入所有 dataclass
from .schema import (
    ArrayInfo, Preprocess, Xcorr, Stack,
    Executables, Device, Storage, Debug,
)

log = logging.getLogger(__name__)


# ----------------------------- custom exception ---------------------------- #
class ConfigError(RuntimeError):
    """Raised when configuration loading / validation fails."""


# ----------------------------- section registry --------------------------- #
# 在这里注册“节名 → 数据类”映射。一行搞定扩展：新增节 ⇒ 加到 dict。
SECTION_MAP: Dict[str, type] = {
    "array1":      ArrayInfo,
    "array2":      ArrayInfo,       # array2 与 array1 复用同一数据类
    "preprocess":  Preprocess,
    "xcorr":       Xcorr,
    "stack":       Stack,
    "executables": Executables,
    "device":      Device,
    "storage":     Storage,
    "debug":       Debug,
}



# ------------------------------- Config ----------------------------------- #
class Config(Mapping[str, Any]):
    """主配置对象，可当成 *mapping* (只读) 也可用 *属性* 访问。"""

    # ---------- construction ----------
    def __init__(self, ini_path: str | Path, *, env_expand: bool = True) -> None:
        """
        Parameters
        ----------
        ini_path : path-like
            The INI file to read.
        env_expand : bool, default True
            Whether to expand ``${ENV}`` tokens in values.
        """
        path = Path(ini_path).expanduser()
        if not path.is_file():
            raise ConfigError(f"INI file not found: {path}")

        self._cp = cp = configparser.ConfigParser(interpolation=None)
        cp.read(path)  # 若失败仍返回 []，但 path 已存在，上面检查过

        # 逐节解析
        self._sections: Dict[str, Any] = {}
        missing_secs: list[str] = []

        for sec, cls in SECTION_MAP.items():
            if sec not in cp:
                missing_secs.append(sec)
                continue
            kv = dict(cp[sec].items())
            if env_expand:
                kv = {k: self._expandenv(v) for k, v in kv.items()}
            try:
                obj = cls.from_cfg(kv)  # type: ignore[arg-type]
            except Exception as e:
                raise ConfigError(f"[{sec}] parsing error: {e}") from None
            self._sections[sec] = obj

        if missing_secs:
            raise ConfigError(f"Missing required section(s): {', '.join(missing_secs)}")

        # array2 可选：若 sac_dir==NONE 则删除
        arr2 = self._sections["array2"]
        if isinstance(arr2, ArrayInfo) and arr2.sac_dir == "NONE":
            del self._sections["array2"]

        # 将各节设为 attribute，IDE 可补全
        for name, obj in self._sections.items():
            setattr(self, name, obj)

        log.debug("Config build complete. Sections: %s", list(self._sections))

    @property
    def is_double_array(self) -> bool:
        """
        True  ⇒ 配置文件包含 [array2] 且其 sac_dir 不是 'NONE'
        False ⇒ 单阵列
        """
        arr2 = self._sections.get("array2")
        return isinstance(arr2, ArrayInfo) and arr2.sac_dir != "NONE"

    # ---------- public API ----------
    def validate_all(self) -> None:
        """
        调用每个 section 的 validate()。
        Debug.validate() 需要拿到 output_dir, 其他节保持原样。
        """
        for name, obj in self._sections.items():
            if not hasattr(obj, "validate"):
                continue

            try:
                if name == "debug":
                    # 关键：把 storage.output_dir 作为实参传进去
                    obj.validate(output_dir=self.storage.output_dir)
                else:
                    obj.validate()
            except Exception as e:
                raise ConfigError(f"[{name}] validation failed: {e}") from None

        log.info("All configuration checks passed.")

    def override(self, section: str, **updates: Any) -> None:
        """在运行期批量修改某节字段。"""
        if section not in self._sections:
            raise KeyError(f"Config has no section '{section}'")
        obj = self._sections[section]
        for k, v in updates.items():
            if not hasattr(obj, k):
                raise AttributeError(f"[{section}] has no field '{k}'")
            setattr(obj, k, v)
        log.debug("Override <%s>: %s", section, updates)

    def to_ini(self, path: str | Path, *, include_defaults: bool = True) -> None:
        """把当前内存配置写回 INI 文件。"""
        cp_out = configparser.ConfigParser()
        for name, obj in self._sections.items():
            data = obj.to_dict() if hasattr(obj, "to_dict") else obj.__dict__
            # 转成字符串；可根据 include_defaults 决定是否过滤默认值
            cp_out[name] = {k: str(v) for k, v in data.items()}
        with Path(path).open("w") as fp:
            cp_out.write(fp)
        log.info("Config written to %s", path)

    # ---------- Mapping interface ----------
    def __getitem__(self, key: str) -> Any:
        return self._sections[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._sections)

    def __len__(self) -> int:
        return len(self._sections)

    # ---------- helpers ----------
    @staticmethod
    def _expandenv(value: str) -> str:
        """Expand ${ENV} variables and ~ in the given string."""
        if "${" in value:
            value = os.path.expandvars(value)
        if value.startswith("~"):
            value = os.path.expanduser(value)
        return value

    def __repr__(self) -> str:  # pragma: no cover
        secs = ", ".join(self._sections)
        return f"<Config [{secs}]>"

    # ---------- quality-of-life shortcuts ----------
    # 便于 IDE 补全：cfg.arrays -> List[ArrayInfo]
    @property
    def arrays(self) -> list[ArrayInfo]:
        arrs = [self.array1]
        if "array2" in self._sections:
            arrs.append(self.array2)  # type: ignore[attr-defined]
        return arrs


# ------------------------------ quick CLI ---------------------------------- #
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Validate a SeisXC INI file.")
    parser.add_argument("ini", help="path to configuration INI")
    parser.add_argument("-q", "--quiet", action="store_true", help="mute INFO logs")
    args = parser.parse_args()

    lvl = logging.ERROR if args.quiet else logging.INFO
    logging.basicConfig(level=lvl, format="%(levelname)s: %(message)s")

    try:
        cfg = Config(args.ini)
        cfg.validate_all()
    except ConfigError as e:
        log.error(e)
        sys.exit(1)

    log.info("INI OK. Sections: %s", list(cfg))
