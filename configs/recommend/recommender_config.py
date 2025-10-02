from configs.base import Config, ModelSettings

NAME = "recommend"

config = Config(
    seed=42,
    mlflow_url="http://localhost:5000",
    experiment_name="doctor-copilot-optimization",
    run_name=NAME,
    limit=100,
    model_settings=ModelSettings(
        model="hosted_vllm/google/medgemma-27b-text-it",
        api_base="http://localhost:8000/v1",
        model_type="chat",
        api_key="o-parola",
        cache=True,
    ),
    scorer_model_uri="runs:/6410abc8ea8241c2aaffc31d6c01aef2/optimized_scorer",
    train_path=None,
    val_path=None,
    test_path=None,
    predict_path="sets/annotated/manual/average/annotated-average-test.csv",
    output_path=f"sets/recommendations/{NAME}/results.json",
)