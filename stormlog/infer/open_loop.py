"""Send requests at scheduled times, bounded by an in-flight limit."""

from __future__ import annotations

import asyncio
import bisect
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

Overflow = Literal["wait", "drop"]

STOPPED_WAITING = "still waiting to be sent when the stop came"


@dataclass(frozen=True)
class Arrival:
    """One scheduled request and how it got onto the wire."""

    index: int
    mode: str
    intended_at_ns: int
    # Every in-flight slot was busy when the dispatcher reached this arrival.
    held_for_slot: bool = False
    in_flight_at_dispatch: int | None = None


class InFlightLimiter:
    """Count outstanding requests and hold new ones at the limit."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("in-flight limit must be >= 1")
        self.limit = limit
        self.active = 0
        self._slots = asyncio.Semaphore(limit)

    @property
    def full(self) -> bool:
        return self.active >= self.limit

    async def acquire(self) -> int:
        """Wait for a free slot; return the in-flight count including this one."""
        await self._slots.acquire()
        self.active += 1
        return self.active

    def release(self) -> None:
        self.active -= 1
        self._slots.release()


@dataclass(frozen=True)
class Dispatch:
    """The requests a schedule sent, and when the schedule started.

    ``tasks`` holds the requests still running and any that failed, whose
    errors are raised by whoever drains them; finished requests are dropped
    so a long schedule holds no more than its in-flight limit.
    """

    started_at_ns: int
    tasks: list[asyncio.Task[None]]
    # Set when ``until`` ended the schedule early: when, and how many
    # arrivals were due by then.
    ended_at_ns: int | None = None
    scheduled: int | None = None


async def dispatch_schedule(
    offsets: Sequence[float],
    *,
    mode: str,
    limiter: InFlightLimiter,
    overflow: Overflow,
    send: Callable[[Arrival], Awaitable[None]],
    drop: Callable[[Arrival, str], None],
    deadline: float | None = None,
    until: asyncio.Event | None = None,
) -> Dispatch:
    """Start each request at its offset from now, in seconds.

    With ``overflow="wait"`` an arrival that finds every slot busy is held
    until one frees up; while it waits, later arrivals fall behind schedule
    too, which their dispatch lag shows. With ``"drop"`` it is handed to
    ``drop`` and never sent, with the reason. ``deadline`` is when the
    phase's drain ends, in seconds from now: an arrival still unsent then is
    dropped rather than held any longer. ``until`` ends the schedule early
    once set: no later arrival is sent, and if it is set while an arrival
    waits for a slot, that arrival and any others already due are dropped.
    Returns once every arrival has been sent or dropped, or ``until`` was
    set; the requests themselves may still be running.
    """
    started = time.perf_counter()
    started_ns = time.time_ns()
    deadline_at = None if deadline is None else started + deadline
    # In creation order, so the first failure is the one raised.
    tasks: dict[asyncio.Task[None], None] = {}
    ended_at_ns: int | None = None
    scheduled: int | None = None

    def forget(task: asyncio.Task[None]) -> None:
        if not _still_matters(task):
            tasks.pop(task, None)

    try:
        for index, offset in enumerate(offsets):
            delay = started + offset - time.perf_counter()
            if await _ended(until, delay):
                ended_at_ns, scheduled = time.time_ns(), index
                break
            arrival = Arrival(
                index,
                mode,
                started_ns + round(offset * 1e9),
                held_for_slot=limiter.full,
            )
            slot = await _slot_for(arrival, limiter, overflow, deadline_at, until)
            if slot == STOPPED_WAITING:
                ended_at_ns = time.time_ns()
                elapsed = time.perf_counter() - started
                scheduled = _drop_due(offsets, arrival, elapsed, started_ns, drop)
                break
            if isinstance(slot, str):
                drop(arrival, slot)
                continue
            arrival = replace(arrival, in_flight_at_dispatch=slot)
            task = asyncio.create_task(_release_after(send(arrival), limiter))
            tasks[task] = None
            task.add_done_callback(forget)
    except asyncio.CancelledError:
        # Stopped part-way: the requests already sent finish cancelling first.
        await cancel_all(list(tasks))
        raise
    # Callbacks for the last requests to finish may not have run yet.
    return Dispatch(
        started_at_ns=started_ns,
        tasks=[task for task in tasks if _still_matters(task)],
        ended_at_ns=ended_at_ns,
        scheduled=scheduled,
    )


async def _ended(until: asyncio.Event | None, delay: float) -> bool:
    """Wait ``delay`` seconds for the next arrival; whether ``until`` was set
    first."""
    if until is None:
        if delay > 0:
            await asyncio.sleep(delay)
        return False
    if until.is_set():
        return True
    try:
        await asyncio.wait_for(until.wait(), timeout=max(delay, 0.0))
    except asyncio.TimeoutError:
        return False
    return True


def _still_matters(task: asyncio.Task[None]) -> bool:
    """Still running, or failed with an error the drain has to raise."""
    if not task.done():
        return True
    return not task.cancelled() and task.exception() is not None


async def _slot_for(
    arrival: Arrival,
    limiter: InFlightLimiter,
    overflow: Overflow,
    deadline: float | None,
    until: asyncio.Event | None,
) -> int | str:
    """Take a slot for ``arrival`` and return the in-flight count with it, or
    return why it is dropped instead."""
    if overflow == "drop" and arrival.held_for_slot:
        return f"in-flight limit of {limiter.limit} reached"
    in_flight = await _acquire_by(limiter, deadline, until)
    if in_flight is not None:
        return in_flight
    if until is not None and until.is_set():
        return STOPPED_WAITING
    return "still waiting to be sent when the drain deadline passed"


async def _acquire_by(
    limiter: InFlightLimiter, deadline: float | None, until: asyncio.Event | None
) -> int | None:
    """Take a slot, or return None once ``deadline`` (a perf_counter time)
    passes or ``until`` is set while waiting for one."""
    timeout = None if deadline is None else deadline - time.perf_counter()
    if timeout is not None and timeout <= 0:
        return None
    if not limiter.full:
        return await limiter.acquire()
    acquiring = asyncio.ensure_future(limiter.acquire())
    waits: list[asyncio.Future[Any]] = [acquiring]
    if until is not None:
        waits.append(asyncio.ensure_future(until.wait()))
    try:
        await asyncio.wait(waits, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for wait in waits:
            wait.cancel()
        await asyncio.gather(*waits, return_exceptions=True)
    return None if acquiring.cancelled() else acquiring.result()


def _drop_due(
    offsets: Sequence[float],
    waiting: Arrival,
    elapsed: float,
    started_ns: int,
    drop: Callable[[Arrival, str], None],
) -> int:
    """Drop the arrival that was waiting for a slot when the stop came, and
    every later one already due ``elapsed`` seconds in; return how many
    arrivals were due."""
    due = max(waiting.index + 1, bisect.bisect_right(offsets, elapsed))
    for index in range(waiting.index, due):
        intended_at_ns = started_ns + round(offsets[index] * 1e9)
        drop(
            replace(waiting, index=index, intended_at_ns=intended_at_ns),
            STOPPED_WAITING,
        )
    return due


async def cancel_all(tasks: Sequence[asyncio.Task[None]]) -> None:
    """Cancel tasks and wait until each has finished cancelling."""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _release_after(request: Awaitable[None], limiter: InFlightLimiter) -> None:
    try:
        await request
    finally:
        limiter.release()
