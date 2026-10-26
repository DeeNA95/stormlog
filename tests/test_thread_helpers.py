from __future__ import annotations

import threading
from typing import Any

import pytest

import tests.thread_helpers as thread_helpers
from tests.thread_helpers import start_running


def test_a_thread_that_dies_at_start_up_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[int] = []

    class DiesOnce(threading.Thread):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            started.append(1)
            if len(started) == 1:
                # As a coverage trace-hook failure would: the thread starts
                # and ends without running its target.
                self.run = lambda: None  # type: ignore[method-assign]

    monkeypatch.setattr(thread_helpers.threading, "Thread", DiesOnce)
    ran = threading.Event()

    thread = start_running(ran.set)
    thread.join(5)

    assert ran.is_set()
    assert len(started) == 2


def test_a_thread_that_never_runs_its_target_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class AlwaysDies(threading.Thread):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.run = lambda: None  # type: ignore[method-assign]

    monkeypatch.setattr(thread_helpers.threading, "Thread", AlwaysDies)

    with pytest.raises(RuntimeError, match="died at start-up"):
        start_running(lambda: None)
