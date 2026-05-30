import pytest

from agent.tools.aws_sagemaker_jobs_tool import (
    AwsSageMakerJobsTool,
    aws_sagemaker_jobs_handler,
)


def _ready_snapshot(**overrides):
    return {
        "configured": True,
        "missing_env": [],
        "region": "us-east-1",
        "s3_bucket": "training-bucket",
        "s3_prefix": "liga-ml",
        "sagemaker_role_arn": "arn:aws:iam::123456789012:role/TestRole",
        "default_instance_type": "ml.g5.xlarge",
        "default_instance_count": 1,
        "default_max_run_seconds": 3600,
        "output_policy": "aws-private",
        "credentials_detected": True,
        "warnings": [],
        "errors": [],
        **overrides,
    }


@pytest.mark.asyncio
async def test_run_missing_aws_config_is_actionable(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(
            configured=False,
            missing_env=["AWS_S3_BUCKET", "AWS_SAGEMAKER_ROLE_ARN"],
            s3_bucket=None,
            sagemaker_role_arn=None,
            credentials_detected=False,
            errors=["Missing required AWS environment variables."],
        ),
    )

    tool = AwsSageMakerJobsTool()
    result = await tool.execute({"operation": "run", "template": "sft"})

    assert result["isError"] is True
    assert "AWS_S3_BUCKET" in result["formatted"]
    assert "AWS_SAGEMAKER_ROLE_ARN" in result["formatted"]
    assert "AWS_REGION" in result["formatted"]
    assert "/api/health/providers" in result["formatted"]
    assert "AWS credentials" in result["formatted"]
    assert "secret" not in result["formatted"].lower()


@pytest.mark.asyncio
async def test_run_validates_request_but_does_not_submit_job(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(),
    )

    class ExplodingSageMakerClient:
        def create_training_job(self, **_kwargs):
            raise AssertionError("create_training_job must not be called in Phase 2")

    tool = AwsSageMakerJobsTool(sagemaker_client=ExplodingSageMakerClient())
    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "dataset_name": "owner/dataset",
            "dataset_config": "default",
            "dataset_split": "train",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_model_id": "owner/aws-output",
            "instance_type": "ml.g5.xlarge",
            "instance_count": 1,
            "max_run_seconds": 3600,
            "job_name": "custom-job-name",
        }
    )

    assert not result.get("isError")
    formatted = result["formatted"]
    assert (
        "AWS SageMaker job execution is not implemented until later AWS phases"
        in formatted
    )
    assert "request/readiness validated but no job submitted" in formatted
    assert "custom-job-name" in formatted
    assert "us-east-1" in formatted
    assert "ml.g5.xlarge" in formatted
    assert "1" in formatted
    assert "3600" in formatted
    assert "Role ARN configured: yes" in formatted
    assert "training-bucket" in formatted
    assert "liga-ml" in formatted
    assert "aws-private" in formatted
    assert "arn:aws:iam::123456789012:role/TestRole" not in formatted


@pytest.mark.asyncio
async def test_run_requires_future_job_fields(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(),
    )

    tool = AwsSageMakerJobsTool()
    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "dataset_name": "owner/dataset",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        }
    )

    assert result["isError"] is True
    assert "output_model_id" in result["formatted"]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["inspect", "logs", "cancel"])
async def test_job_name_operations_require_job_name(operation, monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(),
    )

    tool = AwsSageMakerJobsTool()
    result = await tool.execute({"operation": operation})

    assert result["isError"] is True
    assert "job_name is required" in result["formatted"]


@pytest.mark.asyncio
async def test_read_only_and_cancel_skeletons_do_not_call_live_aws(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(),
    )

    class ExplodingSageMakerClient:
        def list_training_jobs(self, **_kwargs):
            raise AssertionError("list_training_jobs must not be called in Phase 2")

        def describe_training_job(self, **_kwargs):
            raise AssertionError("describe_training_job must not be called in Phase 2")

        def stop_training_job(self, **_kwargs):
            raise AssertionError("stop_training_job must not be called in Phase 2")

    tool = AwsSageMakerJobsTool(sagemaker_client=ExplodingSageMakerClient())

    ps = await tool.execute({"operation": "ps"})
    inspect = await tool.execute({"operation": "inspect", "job_name": "job-1"})
    logs = await tool.execute({"operation": "logs", "job_name": "job-1"})
    cancel = await tool.execute({"operation": "cancel", "job_name": "job-1"})

    assert "read-only listing enabled later" in ps["formatted"]
    assert "not implemented until later AWS phases" in inspect["formatted"]
    assert "not implemented until later AWS phases" in logs["formatted"]
    assert "not implemented until later AWS phases" in cancel["formatted"]
    assert not ps.get("isError")
    assert not inspect.get("isError")
    assert not logs.get("isError")
    assert not cancel.get("isError")


def test_registered_tool_is_available():
    from agent.core.tools import create_builtin_tools

    tool_names = {tool.name for tool in create_builtin_tools(local_mode=True)}

    assert "aws_sagemaker_jobs" in tool_names


@pytest.mark.asyncio
async def test_handler_runs_skeleton(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(),
    )

    output, ok = await aws_sagemaker_jobs_handler(
        {
            "operation": "run",
            "template": "sft",
            "dataset_name": "owner/dataset",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_model_id": "owner/aws-output",
            "max_run_seconds": 3600,
        }
    )

    assert ok is True
    assert "no job submitted" in output
