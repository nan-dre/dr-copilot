from configs.base import Config, ModelSettings

NAME = "recommend_gemma_12b_v4"

config = Config(
    seed=42,
    mlflow_url="http://localhost:5000",
    experiment_name="doctor-copilot-optimization",
    run_name=NAME,
    limit=100,
    model_settings=ModelSettings(
        model="hosted_vllm/google/gemma-3-12b-it",
        api_base="http://localhost:8000/v1",
        model_type="chat",
        api_key="o-parola",
        cache=False,
    ),
    scorer_model_uri=None,
    train_path=None,
    val_path=None,
    test_path=None,
    predict_path="sets/annotated/manual/v4/annotated-average-test.csv",
    output_path=f"sets/recommendations/{NAME}/results.json",
    checkpoint_path="checkpoints/selective_fewshot_gemma_12b_15_shots"
)