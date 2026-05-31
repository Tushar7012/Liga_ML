from agent.core.agent_loop import _needs_approval
from agent.core.tools import create_builtin_tools
from agent.config import Config


def test_training_planner_registered_as_builtin_tool():
    tool_names = {tool.name for tool in create_builtin_tools(local_mode=True)}

    assert "training_planner" in tool_names


def test_training_planner_does_not_require_approval():
    config = Config.model_validate(
        {
            "model_name": "moonshotai/Kimi-K2.6",
            "confirm_cpu_jobs": True,
            "auto_file_upload": False,
            "yolo_mode": False,
        }
    )

    assert not _needs_approval(
        "training_planner",
        {"operation": "recommend", "provider": "aws-sagemaker"},
        config,
    )
