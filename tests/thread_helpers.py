"""Threads for test servers that are sure to be running."""

from __future__ import annotations

import threading
import time
from typing import Callable

ATTEMPTS = 3


def start_running(
    target: Callable[[], None], *, timeout: float = 5.0
) -> threading.Thread:
    """Start ``target`` in a daemon thread and return once it is running.

    Under coverage on Python 3.10 a new thread can die in its own start-up
    ("Cannot install a trace function while another trace function is being
    installed") before its target runs. A test server that never runs makes
    its client fail in ways that look like product bugs, so such a thread is
    replaced, up to ``ATTEMPTS`` times.
    """
    for _ in range(ATTEMPTS):
        running = threading.Event()

        def run(running: threading.Event = running) -> None:
            running.set()
            target()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        deadline = time.monotonic() + timeout
        while not running.wait(0.01):
            if not thread.is_alive():
                break  # died before its target ran: start another
            if time.monotonic() > deadline:
                raise RuntimeError("a test server thread did not start in time")
        if running.is_set():
            return thread
    raise RuntimeError(f"a test server thread died at start-up {ATTEMPTS} times")
