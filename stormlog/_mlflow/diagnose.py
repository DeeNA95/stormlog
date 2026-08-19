"""Diagnose-bundle MLflow export helpers (mirror of the W&B exporter)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .._wandb.core import (
    read_json_if_exists,
    session_slug,
    session_summary_fields,
    session_summary_from_manifest,
)
from .._wandb.diagnose import diagnose_metrics, diagnose_summary_fields
from .attribution import log_attribution_outputs
from .core import (
    MlflowExportConfig,
    log_directory_artifact,
    resolve_run,
    update_summary,
)


def export_diagnose_bundle_to_mlflow(
    config: MlflowExportConfig,
    *,
    command_name: str,
    artifact_dir: str | Path,
) -> None:
    """Export one diagnose bundle directory to MLflow."""
    if not config.enabled:
        return

    bundle_dir = Path(artifact_dir)
    if not bundle_dir.exists() or not bundle_dir.is_dir():
        raise FileNotFoundError(
            f"Diagnose artifact directory not found: {artifact_dir}"
        )

    manifest = read_json_if_exists(bundle_dir / "manifest.json")
    diagnostic_summary = read_json_if_exists(bundle_dir / "diagnostic_summary.json")
    session_summary = session_summary_from_manifest(manifest)

    mlflow, run, managed = resolve_run(
        config,
        command_name=command_name,
        session_summary=session_summary,
    )
    try:
        update_summary(
            mlflow,
            diagnose_metrics(diagnostic_summary, manifest)
            | session_summary_fields(session_summary)
            | diagnose_summary_fields(bundle_dir, manifest),
        )

        if config.log_tables:
            log_suggestions_table(mlflow, diagnostic_summary)

        if config.log_artifacts:
            log_directory_artifact(
                mlflow,
                artifact_path=f"stormlog-diagnose-{session_slug(session_summary)}",
                path=bundle_dir,
            )

        if config.log_attribution:
            update_summary(
                mlflow,
                log_attribution_outputs(
                    mlflow,
                    root=bundle_dir,
                    session_slug=session_slug(session_summary),
                    allow_artifact_logging=config.log_artifacts,
                ),
            )
    finally:
        if managed:
            mlflow.end_run()


def log_suggestions_table(
    mlflow: Any,
    diagnostic_summary: Mapping[str, Any] | None,
) -> None:
    if not isinstance(diagnostic_summary, Mapping):
        return
    suggestions = diagnostic_summary.get("suggestions")
    if not isinstance(suggestions, list) or not suggestions:
        return
    mlflow.log_table(
        data={
            "index": list(range(1, len(suggestions) + 1)),
            "suggestion": [str(suggestion) for suggestion in suggestions],
        },
        artifact_file="stormlog_diagnostic_suggestions.json",
    )
