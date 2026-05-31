from pathlib import Path


def test_prompt_mentions_training_planner_before_cloud_training():
    prompt = Path("agent/prompts/system_prompt_v3.yaml").read_text()

    assert "training_planner" in prompt
    assert "Before launching cloud training" in prompt
    assert "user approval still required" in prompt


def test_prompt_keeps_kaggle_as_future_work_not_download_source():
    prompt = Path("agent/prompts/system_prompt_v3.yaml").read_text()

    assert "Kaggle is future work" in prompt
    assert "Do not rely on Kaggle downloads" in prompt
