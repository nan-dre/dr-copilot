from configs.base import Config, ModelSettings
from optimizers.bootstrap_fewshot_optimizer import BootstrapFewshotOptimizer
from functools import partial

NAME = "bootstrap_fewshot_gemma_27b"

config = Config(
    seed=42,
    mlflow_url="http://localhost:5000",
    experiment_name="doctor-copilot-optimization",
    run_name=NAME,
    limit=100,
    model_settings=ModelSettings(
        model="hosted_vllm/google/gemma-3-27b-it",
        api_base="http://localhost:8000/v1",
        model_type="chat",
        api_key="o-parola",
        cache=True,
    ),
    train_path="sets/annotated/manual/v4/annotated-average-train.csv",
    val_path="sets/annotated/manual/v4/annotated-average-test.csv",
    test_path=None,
    predict_path=None,
    output_path=f"sets/annotated/optimized/{NAME}/results.json",
    optimizer=partial(BootstrapFewshotOptimizer),
    checkpoint_path=f"checkpoints/{NAME}"
)
