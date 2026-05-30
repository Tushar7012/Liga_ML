from types import SimpleNamespace

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
    staged_calls = []

    async def fake_stage(**kwargs):
        staged_calls.append(kwargs)
        return SimpleNamespace(
            s3_train_uri="s3://training-bucket/liga-ml/jobs/custom-job-name/input/train.jsonl",
            s3_prefix_uri="s3://training-bucket/liga-ml/jobs/custom-job-name/",
            s3_output_uri="s3://training-bucket/liga-ml/jobs/custom-job-name/output/",
            s3_checkpoint_uri="s3://training-bucket/liga-ml/jobs/custom-job-name/checkpoints/",
            row_count=3,
            bytes_uploaded=123,
            dataset_name=kwargs["dataset_name"],
            dataset_config=kwargs["dataset_config"],
            dataset_split=kwargs["dataset_split"],
        )

    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.stage_hf_dataset_to_s3", fake_stage
    )

    class ExplodingSageMakerClient:
        def create_training_job(self, **_kwargs):
            raise AssertionError("create_training_job must not be called in Phase 3")

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
    assert "AWS SageMaker dataset staging completed" in formatted
    assert "no SageMaker training job submitted" in formatted
    assert (
        "Dataset staged to S3. SageMaker job submission is not implemented until a later AWS phase; no training job was created."
        in formatted
    )
    assert "custom-job-name" in formatted
    assert "us-east-1" in formatted
    assert (
        "s3://training-bucket/liga-ml/jobs/custom-job-name/input/train.jsonl"
        in formatted
    )
    assert "s3://training-bucket/liga-ml/jobs/custom-job-name/output/" in formatted
    assert "s3://training-bucket/liga-ml/jobs/custom-job-name/checkpoints/" in formatted
    assert "3" in formatted
    assert "123" in formatted
    assert "ml.g5.xlarge" in formatted
    assert "1" in formatted
    assert "3600" in formatted
    assert "Role ARN configured: yes" in formatted
    assert "training-bucket" in formatted
    assert "liga-ml" in formatted
    assert "aws-private" in formatted
    assert "arn:aws:iam::123456789012:role/TestRole" not in formatted
    assert result["metadata"]["state"] == "staged"
    assert result["metadata"]["s3_train_uri"].endswith("/input/train.jsonl")
    assert staged_calls[0]["dataset_name"] == "owner/dataset"
    assert staged_calls[0]["dataset_config"] == "default"
    assert staged_calls[0]["dataset_split"] == "train"
    assert staged_calls[0]["s3_bucket"] == "training-bucket"
    assert staged_calls[0]["s3_prefix"] == "liga-ml"
    assert staged_calls[0]["job_name"] == "custom-job-name"


@pytest.mark.asyncio
async def test_run_missing_dataset_name_is_actionable(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(),
    )

    tool = AwsSageMakerJobsTool()
    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_model_id": "owner/aws-output",
        }
    )

    assert result["isError"] is True
    assert "dataset_name is required" in result["formatted"]


@pytest.mark.asyncio
async def test_run_dataset_load_failure_is_actionable(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(),
    )

    async def fake_stage(**_kwargs):
        raise RuntimeError(
            "Could not load dataset. If this is a private uploaded dataset, an HF token is required."
        )

    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.stage_hf_dataset_to_s3", fake_stage
    )

    tool = AwsSageMakerJobsTool()
    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "dataset_name": "owner/private-dataset",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_model_id": "owner/aws-output",
            "max_run_seconds": 3600,
        }
    )

    assert result["isError"] is True
    assert "Could not load dataset" in result["formatted"]
    assert "HF token is required" in result["formatted"]


@pytest.mark.asyncio
async def test_run_s3_upload_failure_is_actionable(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(),
    )

    async def fake_stage(**_kwargs):
        raise RuntimeError("Could not upload staged dataset to S3: access denied")

    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.stage_hf_dataset_to_s3", fake_stage
    )

    tool = AwsSageMakerJobsTool()
    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "dataset_name": "owner/dataset",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_model_id": "owner/aws-output",
            "max_run_seconds": 3600,
        }
    )

    assert result["isError"] is True
    assert "Could not upload staged dataset to S3" in result["formatted"]


@pytest.mark.asyncio
async def test_run_uses_readiness_defaults_and_explicit_overrides(monkeypatch):
    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.build_aws_sagemaker_readiness_snapshot",
        lambda: _ready_snapshot(
            s3_bucket="default-bucket",
            s3_prefix="default-prefix",
            default_instance_type="ml.g4dn.xlarge",
            default_instance_count=2,
            default_max_run_seconds=7200,
            output_policy="hf-hub",
        ),
    )
    staged_calls = []

    async def fake_stage(**kwargs):
        staged_calls.append(kwargs)
        return SimpleNamespace(
            s3_train_uri="s3://override-bucket/override-prefix/jobs/override-job/input/train.jsonl",
            s3_prefix_uri="s3://override-bucket/override-prefix/jobs/override-job/",
            s3_output_uri="s3://override-bucket/override-prefix/jobs/override-job/output/",
            s3_checkpoint_uri="s3://override-bucket/override-prefix/jobs/override-job/checkpoints/",
            row_count=1,
            bytes_uploaded=17,
            dataset_name=kwargs["dataset_name"],
            dataset_config=kwargs["dataset_config"],
            dataset_split=kwargs["dataset_split"],
        )

    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.stage_hf_dataset_to_s3", fake_stage
    )

    tool = AwsSageMakerJobsTool(session=SimpleNamespace(hf_token="session-token"))
    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "dataset_name": "owner/dataset",
            "dataset_split": "validation",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_model_id": "owner/aws-output",
            "s3_bucket": "override-bucket",
            "s3_prefix": "override-prefix",
            "job_name": "override-job",
            "output_policy": "aws-private",
        }
    )

    assert not result.get("isError")
    formatted = result["formatted"]
    assert "override-bucket" in formatted
    assert "override-prefix" in formatted
    assert "ml.g4dn.xlarge" in formatted
    assert "2" in formatted
    assert "7200" in formatted
    assert "aws-private" in formatted
    assert staged_calls[0]["s3_bucket"] == "override-bucket"
    assert staged_calls[0]["s3_prefix"] == "override-prefix"
    assert staged_calls[0]["dataset_split"] == "validation"
    assert staged_calls[0]["hf_token"] == "session-token"


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

    async def fake_stage(**kwargs):
        return SimpleNamespace(
            s3_train_uri="s3://training-bucket/liga-ml/jobs/generated/input/train.jsonl",
            s3_prefix_uri="s3://training-bucket/liga-ml/jobs/generated/",
            s3_output_uri="s3://training-bucket/liga-ml/jobs/generated/output/",
            s3_checkpoint_uri="s3://training-bucket/liga-ml/jobs/generated/checkpoints/",
            row_count=1,
            bytes_uploaded=17,
            dataset_name=kwargs["dataset_name"],
            dataset_config=kwargs["dataset_config"],
            dataset_split=kwargs["dataset_split"],
        )

    monkeypatch.setattr(
        "agent.tools.aws_sagemaker_jobs_tool.stage_hf_dataset_to_s3",
        fake_stage,
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
    assert "Dataset staged to S3" in output
    assert "no training job was created" in output
