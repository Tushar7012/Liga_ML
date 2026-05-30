"""Safe AWS SageMaker Jobs tool skeleton.

Phase 2 validates AWS readiness, request shape, and conservative cost metadata
without submitting SageMaker jobs or calling live AWS APIs.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from agent.core.aws_readiness import build_aws_sagemaker_readiness_snapshot
from agent.core.cost_estimation import estimate_aws_sagemaker_job_cost
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
    ) -> None:
        self.session = session
        self.tool_call_id = tool_call_id
        self.sagemaker_client = sagemaker_client

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

        return {
            "formatted": (
                "AWS SageMaker job execution is not implemented until later AWS phases; "
                "request/readiness validated but no job submitted.\n\n"
                f"**Would-be job name:** `{job_name}`\n"
                f"**Region:** `{region}`\n"
                f"**Instance type:** `{instance_type}`\n"
                f"**Instance count:** `{instance_count}`\n"
                f"**Max run seconds:** `{max_run_seconds}`\n"
                f"**Role ARN configured: {'yes' if role_arn else 'no'}**\n"
                f"**S3 bucket:** `{s3_bucket}`\n"
                f"**S3 prefix:** `{s3_prefix}`\n"
                f"**Output policy:** `{output_policy}`\n"
                f"{cost_line}\n"
                "Phase 2 does not call `create_training_job`, stage data to S3, "
                "stream CloudWatch logs, or create billable AWS resources."
            ),
            "totalResults": 1,
            "resultsShared": 1,
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
                errors.append(f"{key} is required for future SageMaker run preparation")
        return errors

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
        "Validate AWS SageMaker AI training job requests for Liga ML without launching "
        "AWS resources. Use this when the session provider is AWS SageMaker AI or the "
        "user asks for AWS/SageMaker training, fine-tuning, SFT, model adaptation, or "
        "cloud compute. Phase 2 behavior is intentionally safe: run validates readiness, "
        "request fields, and conservative cost metadata, then reports that execution is "
        "not implemented until later AWS phases. It does not call create_training_job, "
        "stage data to S3, stream CloudWatch logs, or create billable AWS resources. "
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
                "description": "Future training template. Phase 2 validates only.",
            },
            "dataset_name": {
                "type": "string",
                "description": "Dataset id for the future SageMaker job.",
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
