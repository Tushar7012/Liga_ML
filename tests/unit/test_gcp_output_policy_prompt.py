from pathlib import Path


def test_system_prompt_respects_cloud_private_output_policy():
    prompt = Path("agent/prompts/system_prompt_v3.yaml").read_text()

    assert "Google Cloud SFT final models must push to Hugging Face Hub" not in prompt
    assert "push final model artifacts to Hugging Face Hub only when" in prompt
    assert "`cloud-private` saves final artifacts to GCS only" in prompt
    assert "`LIGA_OUTPUT_POLICY`" in prompt


def test_system_prompt_keeps_smoke_tests_out_of_broad_research():
    prompt = Path("agent/prompts/system_prompt_v3.yaml").read_text()

    assert "For `training_goal=smoke-test`" in prompt
    assert "skip broad literature/research crawls" in prompt
    assert "approval-gated `gcp_vertex_jobs` plan directly" in prompt
