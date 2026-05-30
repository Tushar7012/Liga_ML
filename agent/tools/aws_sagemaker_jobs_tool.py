"""Safe AWS SageMaker Jobs tool.

Phase 3 validates AWS readiness and request shape, stages normalized datasets
to S3, and returns conservative cost metadata without submitting SageMaker jobs.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any

from agent.core.aws_dataset_staging import stage_hf_dataset_to_s3
from agent.core.aws_readiness import build_aws_sagemaker_readiness_snapshot
from agent.core.cost_estimation import estimate_aws_sagemaker_job_cost
from agent.core.session import Event
from agent.tools.types import ToolResult

AWS_OUTPUT_POLICIES = {"aws-private", "hf-hub", "cloud-and-hf-hub"}
AWS_REQUIRED_ENV_HELP = (
    "Set AWS_REGION, AWS_S3_BUCKET, and AWS_SAGEMAKER_ROLE_ARN in the backend "
    "environment. Optional defaults: AWS_S3_PREFIX, AWS_DEFAULT_INSTANCE_TYPE, "
    "AWS_DEFAULT_INSTANCE_COUNT, AWS_DEFAULT_MAX_RUN_SECONDS, AWS_OUTPUT_POLICY. "
    "AWS credentials must be discoverable by boto3's default provider chain. "
    "Check /api/health/providers for the current non-sensitive readiness snapshot."
)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9-]+", "-", value.strip()).strip("-").lower()
    return slug[:63] or "liga-ml-sagemaker-job"


def _now_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _positive_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _request_value(args: dict[str, Any], key: str, readiness: dict[str, Any]) -> Any:
    if args.get(key) not in (None, ""):
        return args[key]
    readiness_key = {
        "s3_bucket": "s3_bucket",
        "s3_prefix": "s3_prefix",
        "role_arn": "sagemaker_role_arn",
        "instance_type": "default_instance_type",
        "instance_count": "default_instance_count",
        "max_run_seconds": "default_max_run_seconds",
        "output_policy": "output_policy",
    }.get(key)
    return readiness.get(readiness_key) if readiness_key else None


class AwsSageMakerJobsTool:
    """Validate would-be SageMaker training job requests without AWS calls."""

    def __init__(
        self,
        *,
        session: Any = None,
        tool_call_id: str | None = None,
        sagemaker_client: Any | None = None,
        s3_client: Any | None = None,
    ) -> None:
        self.session = session
        self.tool_call_id = tool_call_id
        self.sagemaker_client = sagemaker_client
        self.s3_client = s3_client

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        operation = str(params.get("operation", "")).lower().strip()
        if not operation:
            return self._error("'operation' parameter is required.")
        if operation == "run":
            return await self._run_job(params)
        if operation == "ps":
            return await self._list_jobs()
        if operation == "inspect":
            return self._inspect_job(params)
        if operation == "logs":
            return self._logs(params)
        if operation == "cancel":
            return self._cancel_job(params)
        return self._error(
            f'Unknown operation: "{operation}". Available operations: run, ps, logs, inspect, cancel.'
        )

    async def _run_job(self, args: dict[str, Any]) -> ToolResult:
        readiness = build_aws_sagemaker_readiness_snapshot()
        if not readiness.get("configured"):
            return self._missing_config_error(readiness)

        errors = self._validate_run_request(args)
        if errors:
            return self._error("; ".join(errors))

        region = str(readiness.get("region") or "us-east-1")
        job_name = _slug(
            str(args.get("job_name") or f"liga-ml-sagemaker-{_now_suffix()}")
        )
        s3_bucket = _request_value(args, "s3_bucket", readiness)
        s3_prefix = str(_request_value(args, "s3_prefix", readiness) or "liga-ml")
        role_arn = _request_value(args, "role_arn", readiness)
        instance_type = str(
            _request_value(args, "instance_type", readiness) or "ml.g5.xlarge"
        )
        instance_count = _positive_int(
            _request_value(args, "instance_count", readiness), 1
        )
        max_run_seconds = _positive_int(
            _request_value(args, "max_run_seconds", readiness), 3600
        )
        output_policy = str(
            _request_value(args, "output_policy", readiness) or "aws-private"
        )
        if output_policy not in AWS_OUTPUT_POLICIES:
            return self._error(
                "output_policy must be one of: "
                + ", ".join(sorted(AWS_OUTPUT_POLICIES))
                + "."
            )

        dataset_name = str(args.get("dataset_name") or "").strip()
        dataset_config = (
            str(args.get("dataset_config")).strip()
            if args.get("dataset_config") not in (None, "")
            else None
        )
        dataset_split = str(args.get("dataset_split") or "train").strip() or "train"
        cost = await estimate_aws_sagemaker_job_cost(
            {
                "operation": "run",
                "instance_type": instance_type,
                "instance_count": instance_count,
                "max_run_seconds": max_run_seconds,
            }
        )
        cost_line = (
            f"**Conservative cost estimate:** ${cost.estimated_cost_usd:.2f}\n"
            if cost.estimated_cost_usd is not None
            else f"**Conservative cost estimate:** unavailable ({cost.block_reason})\n"
        )

        hf_token = self._hf_token()
        try:
            staged = await stage_hf_dataset_to_s3(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                dataset_split=dataset_split,
                s3_bucket=str(s3_bucket),
                s3_prefix=s3_prefix,
                job_name=job_name,
                session_id=getattr(self.session, "session_id", None),
                hf_token=hf_token,
                s3_client=self.s3_client,
            )
        except Exception as exc:
            return self._error(str(exc))

        await self._emit_staged_state(staged, job_name)

        metadata = {
            "state": "staged",
            "job_name": job_name,
            "region": region,
            "s3_train_uri": staged.s3_train_uri,
            "s3_prefix_uri": staged.s3_prefix_uri,
            "s3_output_uri": staged.s3_output_uri,
            "s3_checkpoint_uri": staged.s3_checkpoint_uri,
            "row_count": staged.row_count,
            "bytes_uploaded": staged.bytes_uploaded,
            "dataset_name": staged.dataset_name,
            "dataset_config": staged.dataset_config,
            "dataset_split": staged.dataset_split,
            "instance_type": instance_type,
            "instance_count": instance_count,
            "max_run_seconds": max_run_seconds,
            "output_policy": output_policy,
            "estimated_cost_usd": cost.estimated_cost_usd,
        }

        return {
            "formatted": (
                "AWS SageMaker dataset staging completed; no SageMaker training job submitted.\n\n"
                "Dataset staged to S3. SageMaker job submission is not implemented until a later "
                "AWS phase; no training job was created.\n\n"
                f"**Job name:** `{job_name}`\n"
                f"**Region:** `{region}`\n"
                f"**S3 train URI:** `{staged.s3_train_uri}`\n"
                f"**S3 output URI:** `{staged.s3_output_uri}`\n"
                f"**S3 checkpoint URI:** `{staged.s3_checkpoint_uri}`\n"
                f"**Rows staged:** `{staged.row_count}`\n"
                f"**Bytes uploaded:** `{staged.bytes_uploaded}`\n"
                f"**Instance type:** `{instance_type}`\n"
                f"**Instance count:** `{instance_count}`\n"
                f"**Max run seconds:** `{max_run_seconds}`\n"
                f"**Role ARN configured: {'yes' if role_arn else 'no'}**\n"
                f"**S3 bucket:** `{s3_bucket}`\n"
                f"**S3 prefix:** `{s3_prefix}`\n"
                f"**Output policy:** `{output_policy}`\n"
                f"{cost_line}\n"
                "Phase 3 does not call `create_training_job`, stream CloudWatch logs, "
                "or create SageMaker training resources."
            ),
            "totalResults": 1,
            "resultsShared": 1,
            "metadata": metadata,
        }

    async def _list_jobs(self) -> ToolResult:
        readiness = build_aws_sagemaker_readiness_snapshot()
        return {
            "formatted": (
                "AWS SageMaker read-only listing enabled later; no live AWS call was made.\n\n"
                f"**Configured:** `{bool(readiness.get('configured'))}`\n"
                f"**Region:** `{readiness.get('region')}`"
            ),
            "totalResults": 0,
            "resultsShared": 0,
        }

    def _inspect_job(self, args: dict[str, Any]) -> ToolResult:
        job_name = str(args.get("job_name") or args.get("job_id") or "").strip()
        if not job_name:
            return self._error("job_name is required for inspect.")
        return {
            "formatted": (
                "AWS SageMaker inspect is not implemented until later AWS phases; "
                f"no live AWS call was made for `{job_name}`."
            ),
            "totalResults": 0,
            "resultsShared": 0,
        }

    def _logs(self, args: dict[str, Any]) -> ToolResult:
        job_name = str(args.get("job_name") or args.get("job_id") or "").strip()
        if not job_name:
            return self._error("job_name is required for logs.")
        return {
            "formatted": (
                "AWS SageMaker log streaming is not implemented until later AWS phases; "
                f"no CloudWatch call was made for `{job_name}`."
            ),
            "totalResults": 0,
            "resultsShared": 0,
        }

    def _cancel_job(self, args: dict[str, Any]) -> ToolResult:
        job_name = str(args.get("job_name") or args.get("job_id") or "").strip()
        if not job_name:
            return self._error("job_name is required for cancel.")
        return {
            "formatted": (
                "AWS SageMaker cancellation is not implemented until later AWS phases; "
                f"no `stop_training_job` call was made for `{job_name}`."
            ),
            "totalResults": 0,
            "resultsShared": 0,
        }

    @staticmethod
    def _validate_run_request(args: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        template = str(args.get("template") or "").strip().lower()
        if template and template != "sft":
            errors.append(
                "Unsupported template: " + template + ". Available templates: sft"
            )
        for key in ("dataset_name", "model_name", "output_model_id"):
            if not str(args.get(key) or "").strip():
                errors.append(f"{key} is required for SageMaker dataset staging")
        return errors

    def _hf_token(self) -> str | None:
        session_token = getattr(self.session, "hf_token", None)
        if session_token:
            return str(session_token)
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    async def _emit_staged_state(self, staged: Any, job_name: str) -> None:
        if self.session is None or not self.tool_call_id:
            return
        send_event = getattr(self.session, "send_event", None)
        if send_event is None:
            return
        try:
            await send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": self.tool_call_id,
                        "tool": "aws_sagemaker_jobs",
                        "state": "staged",
                        "jobName": job_name,
                        "s3TrainUri": staged.s3_train_uri,
                        "s3OutputUri": staged.s3_output_uri,
                        "s3CheckpointUri": staged.s3_checkpoint_uri,
                    },
                )
            )
        except Exception:
            return

    @staticmethod
    def _missing_config_error(readiness: dict[str, Any]) -> ToolResult:
        missing = readiness.get("missing_env") or []
        details = []
        if missing:
            details.append(
                "Missing required AWS configuration: " + ", ".join(missing) + "."
            )
        if not readiness.get("credentials_detected"):
            details.append("AWS credentials were not detected by boto3.")
        for error in readiness.get("errors") or []:
            if error not in details:
                details.append(str(error))
        return AwsSageMakerJobsTool._error(" ".join(details + [AWS_REQUIRED_ENV_HELP]))

    @staticmethod
    def _error(message: str) -> ToolResult:
        return {
            "formatted": message,
            "totalResults": 0,
            "resultsShared": 0,
            "isError": True,
        }


AWS_SAGEMAKER_JOBS_TOOL_SPEC = {
    "name": "aws_sagemaker_jobs",
    "description": (
        "Stage normalized datasets to S3 for future AWS SageMaker AI training jobs "
        "without launching SageMaker training. Use this when the session provider is "
        "AWS SageMaker AI or the "
        "user asks for AWS/SageMaker training, fine-tuning, SFT, model adaptation, or "
        "cloud compute. Phase 3 behavior is intentionally limited: run validates "
        "readiness, request fields, and conservative cost metadata, loads the normalized "
        "dataset config, uploads train.jsonl to S3, and reports that SageMaker job "
        "submission is not implemented until later AWS phases. It does not call "
        "create_training_job, stream CloudWatch logs, or create SageMaker training "
        "resources. "
        "Operations: run, ps, inspect, logs, cancel. Run and cancel are approval-gated; "
        "read-only operations are not."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["run", "ps", "inspect", "logs", "cancel"],
                "description": "Operation to execute.",
            },
            "template": {
                "type": "string",
                "enum": ["sft"],
                "description": "Future training template. Phase 3 supports staging only for sft.",
            },
            "dataset_name": {
                "type": "string",
                "description": "Hugging Face dataset id to load and stage to S3.",
            },
            "dataset_config": {
                "type": "string",
                "description": "Optional dataset config.",
            },
            "dataset_split": {
                "type": "string",
                "description": "Dataset split. Default: train.",
            },
            "model_name": {
                "type": "string",
                "description": "Base model id for the future training job.",
            },
            "output_model_id": {
                "type": "string",
                "description": "Intended output model id or artifact label.",
            },
            "output_policy": {
                "type": "string",
                "enum": ["aws-private", "hf-hub", "cloud-and-hf-hub"],
                "description": "Future artifact policy. Default: aws-private.",
            },
            "s3_bucket": {
                "type": "string",
                "description": "Optional S3 bucket override.",
            },
            "s3_prefix": {
                "type": "string",
                "description": "Optional S3 prefix override. Default: liga-ml.",
            },
            "role_arn": {
                "type": "string",
                "description": "Optional SageMaker execution role ARN override.",
            },
            "instance_type": {
                "type": "string",
                "description": "SageMaker training instance type. Default: ml.g5.xlarge.",
            },
            "instance_count": {
                "type": "integer",
                "description": "SageMaker training instance count. Default: 1.",
            },
            "max_run_seconds": {
                "type": "integer",
                "description": "Maximum future runtime in seconds for cost guardrails. Default: 3600.",
            },
            "job_name": {
                "type": "string",
                "description": "Would-be or existing SageMaker job name.",
            },
        },
        "required": ["operation"],
    },
}


async def aws_sagemaker_jobs_handler(
    arguments: dict[str, Any], session: Any = None, tool_call_id: str | None = None
) -> tuple[str, bool]:
    try:
        tool = AwsSageMakerJobsTool(session=session, tool_call_id=tool_call_id)
        result = await tool.execute(arguments)
        return result["formatted"], not result.get("isError", False)
    except Exception as exc:
        return f"Error executing AWS SageMaker Jobs tool: {exc}", False
