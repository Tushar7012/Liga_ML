"""Validation helpers for stable cloud training templates."""

from __future__ import annotations

from typing import Any


def validate_sft_template_request(params: dict[str, Any]) -> list[str]:
    """Return human-readable validation errors for an SFT template request."""

    errors: list[str] = []
    for field in ("dataset_name", "model_name", "hub_model_id"):
        if not str(params.get(field) or "").strip():
            errors.append(f"{field} is required")

    task_type = str(params.get("task_type") or "sft").strip().lower()
    if task_type != "sft":
        errors.append("Only task_type='sft' is supported by the stable SFT template")

    if params.get("packing") is True:
        errors.append("packing=True is not allowed for the stable Vertex SFT template")

    if params.get("attn_implementation"):
        errors.append(
            "attn_implementation is not allowed for the stable Vertex SFT template"
        )

    if params.get("flash_attention") or params.get("use_flash_attention"):
        errors.append(
            "flash attention is not allowed for the stable Vertex SFT template"
        )

    submitted_text = " ".join(
        str(params.get(field) or "")
        for field in (
            "display_name",
            "dataset_name",
            "hub_model_id",
            "trackio_project",
            "trackio_space_id",
        )
    ).lower()
    if "dummy" in submitted_text or "placeholder" in submitted_text:
        errors.append(
            "dummy placeholder values are not allowed for Vertex training jobs"
        )

    target_text = " ".join(
        str(params.get(field) or "")
        for field in (
            "display_name",
            "hub_model_id",
            "trackio_project",
            "trackio_space_id",
        )
    ).lower()
    dataset_name = str(params.get("dataset_name") or "").lower()
    if "finance" in target_text and "medical" in dataset_name:
        errors.append("finance training jobs cannot use an obviously medical dataset")

    return errors
