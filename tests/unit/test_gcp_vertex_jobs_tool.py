import base64
import re
from types import SimpleNamespace

import pytest

from agent.tools.gcp_vertex_jobs_tool import GcpVertexJobsTool, gcp_vertex_jobs_handler


class FakeCustomJob:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.resource_name = (
            "projects/test-project/locations/us-central1/customJobs/123"
        )
        self.name = kwargs["display_name"]
        FakeCustomJob.instances.append(self)

    def run(self, **kwargs):
        self.run_kwargs = kwargs


class FakeSession:
    hf_token = "hf-session-token"
    sandbox = None

    def __init__(self):
        self.events = []

    async def send_event(self, event):
        self.events.append(event)


class FakeState:
    def __init__(self, name):
        self.name = name


class FakeJobServiceClient:
    get_calls = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_custom_job(self, name):
        FakeJobServiceClient.get_calls += 1
        return SimpleNamespace(
            name=name,
            display_name="finance-sft",
            state=FakeState("JOB_STATE_RUNNING"),
            create_time="created",
            update_time="updated",
        )


class FakeLoggingClient:
    list_calls = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def list_entries(self, **kwargs):
        FakeLoggingClient.list_calls += 1
        return []


@pytest.mark.asyncio
async def test_run_command_submits_vertex_custom_job(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    monkeypatch.setenv("VERTEX_AI_OUTPUT_DIR", "gs://liga-training/outputs")
    FakeCustomJob.instances = []

    session = FakeSession()
    tool = GcpVertexJobsTool(
        session=session,
        tool_call_id="call-1",
        custom_job_cls=FakeCustomJob,
    )

    result = await tool.execute(
        {
            "operation": "run",
            "command": ["python", "train.py"],
            "image": "python:3.12",
            "display_name": "gst-train",
            "machine_type": "n1-standard-8",
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": 1,
            "env": {"DATASET_ID": "ligaments/gst"},
        }
    )

    assert not result.get("isError")
    assert "Vertex AI job submitted" in result["formatted"]
    job = FakeCustomJob.instances[0]
    worker_pool = job.kwargs["worker_pool_specs"][0]
    assert worker_pool["machine_spec"] == {
        "machine_type": "n1-standard-8",
        "accelerator_type": "NVIDIA_TESLA_T4",
        "accelerator_count": 1,
    }
    assert worker_pool["container_spec"]["image_uri"] == "python:3.12"
    assert worker_pool["container_spec"]["command"] == ["python", "train.py"]
    env = {item["name"]: item["value"] for item in worker_pool["container_spec"]["env"]}
    assert env["DATASET_ID"] == "ligaments/gst"
    assert env["AIP_MODEL_DIR"] == "gs://liga-training/outputs/gst-train"
    assert env["HF_TOKEN"] == "hf-session-token"
    assert job.run_kwargs["sync"] is False
    assert session.events[0].data["tool"] == "gcp_vertex_jobs"
    assert session.events[0].data["state"] == "running"


@pytest.mark.asyncio
async def test_run_uses_modern_pytorch_vertex_image_by_default(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    monkeypatch.delenv("GCP_VERTEX_DEFAULT_IMAGE", raising=False)
    FakeCustomJob.instances = []

    tool = GcpVertexJobsTool(custom_job_cls=FakeCustomJob)

    result = await tool.execute(
        {
            "operation": "run",
            "command": ["python", "train.py"],
            "display_name": "medical-sft",
            "accelerator_type": "NVIDIA_TESLA_T4",
        }
    )

    assert not result.get("isError")
    worker_pool = FakeCustomJob.instances[0].kwargs["worker_pool_specs"][0]
    assert (
        worker_pool["container_spec"]["image_uri"]
        == "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310"
    )


@pytest.mark.asyncio
async def test_run_sft_template_generates_vertex_training_script(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    FakeCustomJob.instances = []

    session = FakeSession()
    tool = GcpVertexJobsTool(session=session, custom_job_cls=FakeCustomJob)

    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "display_name": "medical-sft",
            "dataset_name": "FreedomIntelligence/medical-o1-reasoning-SFT",
            "dataset_config": "en",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "hub_model_id": "ligaments-dev/medical-qwen2.5-0.5b-sft",
            "column_mapping": {
                "user": "Question",
                "assistant": ["Complex_CoT", "Response"],
            },
            "trackio_project": "medical-sft",
            "trackio_space_id": "ligaments-dev/ml-intern-trackio",
        }
    )

    assert not result.get("isError")
    assert (
        "**HF model target:** https://huggingface.co/ligaments-dev/medical-qwen2.5-0.5b-sft"
        in result["formatted"]
    )
    worker_pool = FakeCustomJob.instances[0].kwargs["worker_pool_specs"][0]
    encoded_runner = worker_pool["container_spec"]["command"][-1]
    encoded_script = re.search(r"b64decode\('([^']+)'\)", encoded_runner).group(1)
    decoded_script = base64.b64decode(encoded_script).decode("utf-8")
    assert "FreedomIntelligence/medical-o1-reasoning-SFT" in decoded_script
    assert "packing=False" in decoded_script
    env = {item["name"]: item["value"] for item in worker_pool["container_spec"]["env"]}
    assert env["TRACKIO_PROJECT"] == "medical-sft"
    assert env["TRACKIO_SPACE_ID"] == "ligaments-dev/ml-intern-trackio"
    assert env["HF_TOKEN"] == "hf-session-token"


@pytest.mark.asyncio
async def test_run_sft_template_requires_core_parameters(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")

    tool = GcpVertexJobsTool(custom_job_cls=FakeCustomJob)

    result = await tool.execute({"operation": "run", "template": "sft"})

    assert result["isError"] is True
    assert "dataset_name is required" in result["formatted"]


@pytest.mark.asyncio
async def test_run_sft_template_rejects_risky_options_before_submit(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    FakeCustomJob.instances = []

    tool = GcpVertexJobsTool(custom_job_cls=FakeCustomJob)

    result = await tool.execute(
        {
            "operation": "run",
            "template": "sft",
            "dataset_name": "trl-lib/Capybara",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "hub_model_id": "ligaments-dev/test-model",
            "packing": True,
        }
    )

    assert result["isError"] is True
    assert "packing=True is not allowed" in result["formatted"]
    assert FakeCustomJob.instances == []


@pytest.mark.asyncio
async def test_vertex_monitoring_is_throttled_per_session(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    FakeJobServiceClient.get_calls = 0
    FakeLoggingClient.list_calls = 0

    session = FakeSession()
    tool = GcpVertexJobsTool(
        session=session,
        job_service_client_cls=FakeJobServiceClient,
        logging_client_cls=FakeLoggingClient,
    )
    job_name = "projects/test-project/locations/us-central1/customJobs/123"

    first = await tool.execute({"operation": "inspect", "job_name": job_name})
    second = await tool.execute({"operation": "logs", "job_name": job_name})

    assert "JOB_STATE_RUNNING" in first["formatted"]
    assert "Vertex job monitoring is rate-limited" in second["formatted"]
    assert "wait" in second["formatted"].lower()
    assert not second.get("isError")
    assert FakeJobServiceClient.get_calls == 1
    assert FakeLoggingClient.list_calls == 0


@pytest.mark.asyncio
async def test_run_requires_project_region_and_bucket(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_REGION", raising=False)
    monkeypatch.delenv("GCS_BUCKET", raising=False)

    tool = GcpVertexJobsTool(custom_job_cls=FakeCustomJob)

    result = await tool.execute({"operation": "run", "command": ["python", "train.py"]})

    assert result["isError"] is True
    assert "GOOGLE_CLOUD_PROJECT" in result["formatted"]
    assert "GOOGLE_CLOUD_REGION" in result["formatted"]
    assert "GCS_BUCKET" in result["formatted"]


@pytest.mark.asyncio
async def test_handler_reads_script_from_active_sandbox(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    monkeypatch.setenv("GCS_BUCKET", "liga-training")
    FakeCustomJob.instances = []

    class Sandbox:
        pass

    session = FakeSession()
    session.sandbox = Sandbox()

    async def fake_resolve_sandbox_script(sandbox, script):
        assert script == "/app/train.py"
        return "print('train')", None

    monkeypatch.setattr(
        "agent.tools.sandbox_tool.resolve_sandbox_script",
        fake_resolve_sandbox_script,
    )
    monkeypatch.setattr(
        "agent.tools.gcp_vertex_jobs_tool._load_custom_job_cls",
        lambda: FakeCustomJob,
    )

    output, ok = await gcp_vertex_jobs_handler(
        {"operation": "run", "script": "/app/train.py"},
        session=session,
        tool_call_id="call-2",
    )

    assert ok is True
    assert "Vertex AI job submitted" in output
    command = FakeCustomJob.instances[0].kwargs["worker_pool_specs"][0][
        "container_spec"
    ]["command"]
    assert base64.b64encode(b"print('train')").decode("ascii") in command[-1]


def test_registered_tool_is_available():
    from agent.core.tools import create_builtin_tools

    tool_names = {tool.name for tool in create_builtin_tools(local_mode=True)}

    assert "gcp_vertex_jobs" in tool_names
