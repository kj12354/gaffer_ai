"""Wall-clock and peak-RSS logging for the compute-efficiency report."""

from __future__ import annotations

import platform
import resource
import time
from dataclasses import dataclass


@dataclass
class RunStats:
    wall_clock_s: float
    peak_rss_mb: float

    def format_line(self) -> str:
        return (
            f"[compute] wall_clock_s={self.wall_clock_s:.2f}  "
            f"peak_rss_mb={self.peak_rss_mb:.1f}"
        )


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in megabytes.

    macOS reports ``ru_maxrss`` in bytes; Linux reports kibibytes.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


class Timer:
    """Context manager that records wall time and peak RSS."""

    def __init__(self) -> None:
        self.stats = RunStats(wall_clock_s=0.0, peak_rss_mb=0.0)
        self._t0 = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stats = RunStats(
            wall_clock_s=time.perf_counter() - self._t0,
            peak_rss_mb=peak_rss_mb(),
        )
