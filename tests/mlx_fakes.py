"""Small deterministic native-API fake; no framework dependencies."""

from collections import deque
from types import SimpleNamespace
from typing import Any, Iterable

from stormlog.mlx.runtime import MLXRuntime


class FakeCore:
    gpu = "gpu"

    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.values: dict[str, Any] = {
            "active_memory": 100,
            "cache_memory": 20,
            "peak_memory": 1000,
            "memory_limit": 2000,
        }
        self.queued: dict[str, deque[Any]] = {}
        self.metal = SimpleNamespace(is_available=lambda: True)
        self.outputs: list[tuple[int, ...]] = []
        self.sync_error: BaseException | None = None

    def _get(self, name: str) -> Any:
        self.calls.append(name)
        queue = self.queued.get(name)
        value = queue.popleft() if queue else self.values[name]
        if isinstance(value, BaseException):
            raise value
        return value

    def queue(self, name: str, values: Iterable[Any]) -> None:
        self.queued[name] = deque(values)

    def get_active_memory(self) -> Any:
        return self._get("active_memory")

    def get_cache_memory(self) -> Any:
        return self._get("cache_memory")

    def get_peak_memory(self) -> Any:
        return self._get("peak_memory")

    def get_memory_limit(self) -> Any:
        return self._get("memory_limit")

    def device_info(self, device: Any) -> dict[str, Any]:
        self.calls.append(("device_info", device))
        return {"device_name": "Fake Metal", "max_recommended_working_set_size": 5000}

    def default_stream(self, device: Any) -> str:
        self.calls.append(("default_stream", device))
        return "default"

    def eval(self, *roots: Any) -> None:
        self.calls.append("eval")
        # Do not retain objects; real profiling must not retain them either.
        self.outputs.append(tuple(id(root) for root in roots))
        self.values["active_memory"] = 500
        self.values["peak_memory"] = max(500, self.values["peak_memory"])

    def synchronize(self, stream: Any) -> None:
        self.calls.append(("sync", stream))
        if self.sync_error:
            raise self.sync_error

    def reset_peak_memory(self) -> None:
        self.calls.append("reset")
        self.values["peak_memory"] = 0


def make_runtime(core: FakeCore | None = None) -> MLXRuntime:
    return MLXRuntime(
        core=core or FakeCore(),
        runtime_version="0.32.3",
        platform_system="Darwin",
        platform_machine="arm64",
    )
