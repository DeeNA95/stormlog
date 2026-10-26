import functools

from stormlog.mlx import MLXMemoryProfiler
from stormlog.mlx.context_profiler import (
    clear_global_profiler,
    get_global_profiler,
    profile_context,
    profile_function,
    set_global_profiler,
)
from tests.mlx_fakes import make_runtime


def test_decorator_preserves_metadata_and_kwargs() -> None:
    profiler = MLXMemoryProfiler(runtime=make_runtime())

    @profile_function(profiler=profiler)
    def workload(name: str, state_getter: int = 3) -> tuple[str, int]:
        """A workload with kwargs resembling profiler options."""
        return name, state_getter

    assert workload(name="user", state_getter=7) == ("user", 7)
    assert workload.__name__ == "workload"
    assert (workload.__doc__ or "").startswith("A workload")
    assert profiler.get_results()[0].name == "workload"


def test_global_helpers_lifecycle() -> None:
    profiler = MLXMemoryProfiler(runtime=make_runtime())
    set_global_profiler(profiler)
    try:
        assert get_global_profiler() is profiler
        with profile_context("global") as region:
            region.evaluate(None)
        assert profiler.get_results()[0].name == "global"
    finally:
        clear_global_profiler()


def test_decorator_accepts_a_callable_without_a_name() -> None:
    profiler = MLXMemoryProfiler(runtime=make_runtime())

    def add(a: int, b: int) -> int:
        return a + b

    wrapped = profile_function(functools.partial(add, 1), profiler=profiler)
    assert wrapped(2) == 3
    assert profiler.get_results()[0].name == "function"
