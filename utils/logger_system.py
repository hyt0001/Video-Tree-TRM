"""
日志系统模块
============
提供双通道日志（system.log 文本 + metrics.json 结构化），
以及全局便捷函数 log_msg / log_json / ensure / log_exception。

使用方式::

    from utils.logger_system import log_msg, log_json, ensure, log_exception

    log_msg("INFO", "模型加载完成", model="bert-base")
    log_json("train_loss", {"epoch": 1, "loss": 0.35})
    ensure(tensor.shape[0] > 0, "batch 不能为空")
"""

from __future__ import annotations

import json
import logging
import os
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# 日志目录
# ---------------------------------------------------------------------------
_LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_SYSTEM_LOG = _LOG_DIR / "system.log"
_METRICS_LOG = _LOG_DIR / "metrics.json"


# ---------------------------------------------------------------------------
# LoggerSystem
# ---------------------------------------------------------------------------
class LoggerSystem:
    """双通道日志系统。

    通道 1: system.log — 文本格式，记录所有级别的运行日志。
    通道 2: metrics.json — JSON Lines 格式，记录结构化指标数据。

    Attributes:
        _logger: 标准库 Logger 实例，输出到 system.log。
        _metrics_path: metrics.json 文件路径。
    """

    _instance: Optional["LoggerSystem"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, log_dir: Optional[str] = None) -> None:
        """初始化日志系统。

        参数:
            log_dir: 日志输出目录，默认为 'logs/'。
        """
        log_path = Path(log_dir) if log_dir else _LOG_DIR
        log_path.mkdir(parents=True, exist_ok=True)

        self._metrics_path = log_path / "metrics.json"

        # 文本日志
        self._logger = logging.getLogger("video_tree_trm")
        if not self._logger.handlers:
            self._logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(str(log_path / "system.log"), encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)

    @classmethod
    def get(cls) -> "LoggerSystem":
        """获取全局单例（线程安全，双重检查锁定）。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ---- 文本日志 ----

    def msg(self, level: str, message: str, **kwargs: Any) -> None:
        """写入文本日志。

        参数:
            level: 日志级别，如 "INFO", "ERROR", "DEBUG", "WARNING"。
            message: 日志消息。
            **kwargs: 附加键值对，追加到消息末尾。
        """
        extra = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        text = f"{message} | {extra}" if extra else message
        log_fn = getattr(self._logger, level.lower(), self._logger.info)
        log_fn(text)

    # ---- 结构化指标 ----

    def json(self, tag: str, data: dict[str, Any]) -> None:
        """写入结构化 JSON 指标。

        参数:
            tag: 指标标签，如 "train_loss"。
            data: 指标数据字典。
        """
        record = {"ts": datetime.now().isoformat(), "tag": tag, **data}
        with open(self._metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- 断言 ----

    @staticmethod
    def ensure(condition: bool, message: str) -> None:
        """运行时断言，失败时抛出 ValueError。

        参数:
            condition: 断言条件。
            message: 失败时的错误消息。
        """
        if not condition:
            raise ValueError(message)

    # ---- 异常记录 ----

    def exception(self, message: str, exc: BaseException) -> None:
        """记录异常信息到日志。

        参数:
            message: 上下文描述。
            exc: 异常实例。
        """
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        self._logger.error(f"{message} | {''.join(tb)}")


# ---------------------------------------------------------------------------
# 全局便捷函数
# ---------------------------------------------------------------------------


def log_msg(level: str, message: str, **kwargs: Any) -> None:
    """写入文本日志（全局便捷函数）。"""
    LoggerSystem.get().msg(level, message, **kwargs)


def log_json(tag: str, data: dict[str, Any]) -> None:
    """写入结构化 JSON 指标（全局便捷函数）。"""
    LoggerSystem.get().json(tag, data)


def ensure(condition: bool, message: str) -> None:
    """运行时断言（全局便捷函数）。"""
    LoggerSystem.ensure(condition, message)


def log_exception(message: str, exc: BaseException) -> None:
    """记录异常（全局便捷函数）。"""
    LoggerSystem.get().exception(message, exc)
