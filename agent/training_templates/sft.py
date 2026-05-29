"""Stable SFT training script generation.

The agent should choose dataset/model/task values. This module owns the
runtime-sensitive training script so cloud jobs do not rely on ad hoc code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


DEFAULT_PACKAGES = [
    "torch==2.4.0",
    "transformers>=4.45,<5",
    "trl==1.5.1",
    "accelerate>=0.34,<2",
    "datasets>=3.0,<5",
    "peft>=0.13,<1",
    "huggingface_hub>=0.25,<2",
    "google-cloud-storage>=2.18,<4",
    "trackio",
    "sentencepiece",
]


@dataclass(frozen=True)
class SftTemplateConfig:
    dataset_name: str
    model_name: str
    hub_model_id: str
    dataset_config: str | None = None
    dataset_split: str = "train"
    task_type: str = "sft"
    column_mapping: dict[str, Any] = field(default_factory=dict)
    max_train_samples: int | None = None
    num_train_epochs: int = 1
    max_length: int = 1024
    trackio_project: str | None = None
    trackio_space_id: str | None = None


def build_sft_training_script(config: SftTemplateConfig) -> str:
    """Build a conservative, Vertex/HF compatible SFT training script."""

    if config.task_type != "sft":
        raise ValueError("Only task_type='sft' is supported by this template.")
    if not config.dataset_name.strip():
        raise ValueError("dataset_name is required.")
    if not config.model_name.strip():
        raise ValueError("model_name is required.")
    if not config.hub_model_id.strip():
        raise ValueError("hub_model_id is required.")

    payload = {
        "dataset_name": config.dataset_name,
        "dataset_config": config.dataset_config,
        "dataset_split": config.dataset_split,
        "model_name": config.model_name,
        "hub_model_id": config.hub_model_id,
        "column_mapping": config.column_mapping,
        "max_train_samples": config.max_train_samples,
        "num_train_epochs": config.num_train_epochs,
        "max_length": config.max_length,
        "trackio_project": config.trackio_project,
        "trackio_space_id": config.trackio_space_id,
        "packages": DEFAULT_PACKAGES,
    }
    config_json = json.dumps(payload, sort_keys=True)
    packages_source = json.dumps(DEFAULT_PACKAGES, indent=4)

    return f'''"""Generated Liga ML SFT training script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

CONFIG = json.loads({config_json!r})

DATASET_NAME = "{config.dataset_name}"
MODEL_NAME = "{config.model_name}"
HUB_MODEL_ID = "{config.hub_model_id}"
VERTEX_OUTPUT_DIR = os.environ.get("AIP_MODEL_DIR", "/tmp/liga-ml-sft-output")
OUTPUT_DIR = (
    "/tmp/liga-ml-sft-output"
    if VERTEX_OUTPUT_DIR.startswith("gs://")
    else VERTEX_OUTPUT_DIR
)
REQUIRED_PACKAGES = {packages_source}


def install_dependencies() -> None:
    if any(not package for package in REQUIRED_PACKAGES):
        raise RuntimeError("Empty dependency entry is not allowed.")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--upgrade",
        *REQUIRED_PACKAGES,
    ])


install_dependencies()

from datasets import load_dataset
from google.cloud import storage
from huggingface_hub import HfApi
from trl import SFTConfig, SFTTrainer


def upload_folder_to_gcs(folder_path: Path, gcs_uri: str) -> None:
    if not gcs_uri.startswith("gs://"):
        return

    bucket_and_prefix = gcs_uri.removeprefix("gs://").strip("/")
    bucket_name, _, prefix = bucket_and_prefix.partition("/")
    if not bucket_name:
        raise RuntimeError(f"Invalid GCS output path: {{gcs_uri}}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for file_path in folder_path.rglob("*"):
        if not file_path.is_file():
            continue
        relative_path = file_path.relative_to(folder_path).as_posix()
        blob_name = "/".join(part for part in [prefix, "final", relative_path] if part)
        bucket.blob(blob_name).upload_from_filename(str(file_path))


def format_example(example):
    mapping = CONFIG.get("column_mapping") or {{}}
    if "messages" in example:
        return {{"messages": example["messages"]}}
    if "prompt" in example and "completion" in example:
        return {{"prompt": example["prompt"], "completion": example["completion"]}}

    user_column = mapping.get("user") or "input"
    assistant_columns = mapping.get("assistant") or ["output"]
    if isinstance(assistant_columns, str):
        assistant_columns = [assistant_columns]

    if user_column not in example:
        raise KeyError(f"Missing user column: {{user_column}}")
    missing = [column for column in assistant_columns if column not in example]
    if missing:
        raise KeyError(f"Missing assistant columns: {{missing}}")

    assistant_text = "\\n\\n".join(str(example[column]).strip() for column in assistant_columns if example[column])
    return {{
        "messages": [
            {{"role": "user", "content": str(example[user_column]).strip()}},
            {{"role": "assistant", "content": assistant_text}},
        ]
    }}


def main() -> None:
    dataset_kwargs = {{"path": CONFIG["dataset_name"], "split": CONFIG["dataset_split"]}}
    if CONFIG.get("dataset_config"):
        dataset_kwargs["name"] = CONFIG["dataset_config"]

    dataset = load_dataset(**dataset_kwargs)
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    if CONFIG.get("max_train_samples"):
        dataset = dataset.select(range(min(int(CONFIG["max_train_samples"]), len(dataset))))

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=int(CONFIG["num_train_epochs"]),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        save_strategy="epoch",
        fp16=True,
        gradient_checkpointing=True,
        disable_tqdm=True,
        report_to=["trackio"] if CONFIG.get("trackio_space_id") else [],
        project=CONFIG.get("trackio_project"),
        trackio_space_id=CONFIG.get("trackio_space_id"),
        remove_unused_columns=False,
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
        hub_strategy="every_save",
        packing=False,
        max_length={int(config.max_length)},
    )

    trainer = SFTTrainer(
        model=CONFIG["model_name"],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    final_dir = Path(OUTPUT_DIR) / "final"
    trainer.save_model(str(final_dir))
    trainer.processing_class.save_pretrained(str(final_dir))
    trainer.push_to_hub()

    api = HfApi()
    api.upload_folder(folder_path=str(final_dir), repo_id=HUB_MODEL_ID, repo_type="model")
    upload_folder_to_gcs(final_dir, VERTEX_OUTPUT_DIR)

    print(f"Final GCS/Vertex output: {{VERTEX_OUTPUT_DIR}}", flush=True)
    print(f"Final Hugging Face model: https://huggingface.co/{{HUB_MODEL_ID}}", flush=True)


if __name__ == "__main__":
    main()
'''
