import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from stormlog.mlx.profile_artifact import load_profiles, validate_profiles
from stormlog.mlx.profiler import MLXMemoryProfiler
from tests.mlx_fakes import make_runtime


def test_profile_schema_and_validator_agree(tmp_path: Path) -> None:
    profiler = MLXMemoryProfiler(runtime=make_runtime())
    profiler.profile_function(lambda: None, name="schema")
    path = tmp_path / "profile.json"
    profiler.export(str(path))
    payload = load_profiles(path)
    schema = json.loads(
        (
            Path(__file__).parents[1] / "docs/schemas/mlx_profile_v1.schema.json"
        ).read_text()
    )
    jsonschema.validate(payload, schema)
    for key, value in [
        ("elapsed_ns", True),
        ("valid_sample_count", -1),
        ("completion_verified", 1),
        ("peak_mode", "subtract_lifetime_peaks"),
        ("metadata", []),
    ]:
        invalid = copy.deepcopy(payload)
        invalid["profiles"][0][key] = value
        with pytest.raises(ValueError):
            validate_profiles(invalid)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)
    invalid = copy.deepcopy(payload)
    invalid["profiles"][0]["unexpected"] = "field"
    with pytest.raises(ValueError):
        validate_profiles(invalid)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)


def test_nested_artifact_validation_matches_schema(tmp_path: Path) -> None:
    profiler = MLXMemoryProfiler(runtime=make_runtime())
    profiler.profile_function(lambda: None)
    path = tmp_path / "profile.json"
    profiler.export(str(path))
    payload = load_profiles(path)
    schema = json.loads(
        (
            Path(__file__).parents[1] / "docs/schemas/mlx_profile_v1.schema.json"
        ).read_text()
    )
    invalids = []
    for key, value in [
        ("pid", True),
        ("rank", -1),
        ("world_size", 0),
        ("job_id", 3),
        ("unknown", 1),
    ]:
        invalid = copy.deepcopy(payload)
        invalid["profiles"][0]["session_summary"][key] = value
        invalids.append(invalid)
    invalid = copy.deepcopy(payload)
    invalid["profiles"][0]["final"]["unavailable"] = {"rss": 1}
    invalids.append(invalid)
    for invalid in invalids:
        with pytest.raises(ValueError):
            validate_profiles(invalid)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)


def test_a_wall_clock_step_back_still_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stormlog.mlx.profiler as profiler_module

    profiler = MLXMemoryProfiler(runtime=make_runtime(), sampling_interval=100)
    stepped_back = SimpleNamespace(
        perf_counter_ns=time.perf_counter_ns,
        time_ns=lambda: time.time_ns() - 5_000_000_000,
    )
    monkeypatch.setattr(profiler_module, "time", stepped_back)
    with profiler.profile_context("clock", already_evaluated=True):
        pass
    monkeypatch.undo()
    (result,) = profiler.get_results()
    assert result.ended_at_ns < result.started_at_ns
    path = tmp_path / "profile.json"
    profiler.export(str(path))
    assert [item["name"] for item in load_profiles(path)["profiles"]] == ["clock"]
