from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Any, Optional, Iterable


@dataclass
class _Clock:
    start_ns: int = 0

    def start(self) -> None:
        self.start_ns = time.perf_counter_ns()

    def elapsed_ms(self) -> float:
        if self.start_ns == 0:
            return 0.0
        return (time.perf_counter_ns() - self.start_ns) / 1e6


class LogLevel:
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40


@dataclass
class Logger:
    level: int = LogLevel.INFO

    def _log(self, lvl: int, msg: str) -> None:
        if lvl >= self.level:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{ts}] {lvl} {msg}")

    def debug(self, msg: str) -> None:
        self._log(LogLevel.DEBUG, msg)

    def info(self, msg: str) -> None:
        self._log(LogLevel.INFO, msg)

    def warn(self, msg: str) -> None:
        self._log(LogLevel.WARN, msg)

    def error(self, msg: str) -> None:
        self._log(LogLevel.ERROR, msg)


def timeit(logger: Optional[Logger] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            clk = _Clock()
            clk.start()
            try:
                return fn(*args, **kwargs)
            finally:
                ms = clk.elapsed_ms()
                if logger:
                    logger.debug(f"{fn.__name__} took {ms:.2f} ms")
        return wrapper
    return decorator


def retry(times: int = 3, delay_ms: int = 50, logger: Optional[Logger] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[BaseException] = None
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if logger:
                        logger.warn(f"{fn.__name__} failed attempt {attempt}/{times}: {e}")
                    time.sleep(max(0.0, delay_ms / 1000.0))
            if logger:
                logger.error(f"{fn.__name__} exhausted retries")
            if last_exc:
                raise last_exc
            return None
        return wrapper
    return decorator


def moving_average(seq: Iterable[float], window: int = 10) -> float:
    values = list(seq)[-max(1, window):]
    if not values:
        return 0.0
    return sum(values) / len(values)
