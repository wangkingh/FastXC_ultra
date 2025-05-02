# config_parser/__init__.py
# -----------------------------------------------------------------------------
"""SeisXC configuration parser.

>>> from config_parser import Config
>>> cfg = Config("config.ini")
>>> cfg.validate_all()
"""

from importlib.metadata import version, PackageNotFoundError

# 公共 API：直接把 loader 的两个关键对象暴露出去
from .loader import Config, ConfigError  # noqa: F401

# 动态获取包版本；若未打包发行则设置为 "dev"
try:
    __version__: str = version(__name__)
except PackageNotFoundError:            # pragma: no cover
    __version__ = "dev"

__all__ = ["Config", "ConfigError", "__version__"]
