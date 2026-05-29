from agent.training_templates.sft import SftTemplateConfig, build_sft_training_script


def test_sft_template_generates_safe_vertex_training_script():
    script = build_sft_training_script(
        SftTemplateConfig(
            dataset_name="FreedomIntelligence/medical-o1-reasoning-SFT",
            dataset_config="en",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            hub_model_id="ligaments-dev/medical-qwen2.5-0.5b-sft",
            task_type="sft",
            column_mapping={
                "user": "Question",
                "assistant": ["Complex_CoT", "Response"],
            },
            max_train_samples=40,
            num_train_epochs=1,
            trackio_project="medical-sft",
            trackio_space_id="ligaments-dev/ml-intern-trackio",
        )
    )

    assert 'DATASET_NAME = "FreedomIntelligence/medical-o1-reasoning-SFT"' in script
    assert 'MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"' in script
    assert 'HUB_MODEL_ID = "ligaments-dev/medical-qwen2.5-0.5b-sft"' in script
    assert 'OUTPUT_DIR = os.environ.get("AIP_MODEL_DIR"' in script
    assert "packing=False" in script
    assert "max_length=1024" in script
    assert "gradient_checkpointing=True" in script
    assert "disable_tqdm=True" in script
    assert "logging_first_step=True" in script
    assert "push_to_hub=True" in script
    assert "trainer.push_to_hub" in script
    assert "api.upload_folder" in script
    assert "upload_folder_to_gcs(final_dir, VERTEX_OUTPUT_DIR)" in script
    assert "Final Hugging Face model:" in script
    assert "Final GCS/Vertex output:" in script


def test_sft_template_uses_deterministic_dependency_install():
    script = build_sft_training_script(
        SftTemplateConfig(
            dataset_name="trl-lib/Capybara",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            hub_model_id="ligaments-dev/test-model",
        )
    )

    assert "REQUIRED_PACKAGES = [" in script
    assert '"--upgrade", "",' not in script
    assert "torch==2.4.0" in script
    assert "transformers" in script
    assert "trl==1.5.1" in script
    assert "google-cloud-storage" in script
    assert "subprocess.check_call" in script
    assert "for package in REQUIRED_PACKAGES:" not in script
    assert "*REQUIRED_PACKAGES" in script
