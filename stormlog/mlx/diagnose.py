"""Bounded diagnostics of this process's MLX runtime and allocator."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from .artifacts import CaptureBundle, CaptureOptions, Outcome, capture_failure
from .runtime import Runtime


def capture_process(
    *,
    output: str | Path | None,
    duration: float = 0,
    options: CaptureOptions | None = None,
    runtime: Runtime | None = None,
    command: str = "diagnose",
    wait: Callable[[float], None] = time.sleep,
) -> tuple[Path, Outcome]:
    import math

    if not math.isfinite(duration) or duration < 0:
        raise ValueError("duration must be finite and nonnegative")
    bundle = CaptureBundle(output, command, options or CaptureOptions())
    outcome = Outcome()
    try:
        bundle.start(runtime)
        wait(duration)
    except BaseException as exc:
        outcome = capture_failure(exc)
        bundle.environment["runtime_error"] = outcome.error
    finally:
        bundle.finish(outcome)
    return bundle.directory, outcome


def run_diagnose(
    *,
    output: str | Path | None,
    duration: float = 0,
    options: CaptureOptions | None = None,
    runtime: Runtime | None = None,
) -> tuple[Path, Outcome]:
    return capture_process(
        output=output, duration=duration, options=options, runtime=runtime
    )
