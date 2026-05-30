from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _env_example() -> str:
    return (ROOT / ".env.example").read_text(encoding="utf-8")


def test_env_example_includes_cloud_run_and_vertex_placeholders() -> None:
    text = _env_example()

    expected_lines = [
        "HF_TOKEN=",
        "GITHUB_TOKEN=",
        "OPENAI_API_KEY=",
        "ML_INTERN_DEFAULT_MODEL_ID=moonshotai/Kimi-K2.6",
        "ML_INTERN_KPIS_DISABLED=1",
        "GOOGLE_CLOUD_PROJECT=",
        "GOOGLE_CLOUD_REGION=us-central1",
        "GCS_BUCKET=",
        "VERTEX_AI_STAGING_BUCKET=",
        "VERTEX_AI_OUTPUT_DIR=",
        "VERTEX_AI_SERVICE_ACCOUNT=",
        "CORS_ALLOW_ORIGINS=",
        "ALLOWED_HOSTS=",
    ]
    for line in expected_lines:
        assert line in text


def test_env_example_includes_aws_sagemaker_placeholders() -> None:
    text = _env_example()

    expected_lines = [
        "# AWS / SageMaker",
        "AWS_REGION=us-east-1",
        "AWS_S3_BUCKET=",
        "AWS_S3_PREFIX=liga-ml",
        "AWS_SAGEMAKER_ROLE_ARN=",
        "AWS_DEFAULT_INSTANCE_TYPE=ml.g5.xlarge",
        "AWS_DEFAULT_INSTANCE_COUNT=1",
        "AWS_DEFAULT_MAX_RUN_SECONDS=3600",
        "AWS_OUTPUT_POLICY=aws-private",
        "AWS_SAGEMAKER_TRAINING_IMAGE_URI=",
        "AWS_ACCESS_KEY_ID=",
        "AWS_SECRET_ACCESS_KEY=",
        "AWS_SESSION_TOKEN=",
    ]
    for line in expected_lines:
        assert line in text


def test_env_example_does_not_contain_real_looking_secrets() -> None:
    text = _env_example()

    forbidden = [
        "hf_",
        "github_pat_",
        "ghp_",
        "sk-",
        "AKIA",
        "ASIA",
        "aws_secret_access_key=",
        "-----BEGIN PRIVATE KEY-----",
    ]
    assert all(marker not in text for marker in forbidden)
