from configs.base import Config, ModelSettings
from optimizers.simba_optimizer import SimbaOptimizer
from functools import partial

NAME = "simba_medgemma_27b_full_dataset"

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
    train_path="sets/annotated/manual/v4/annotated-average.csv",
    val_path="sets/annotated/manual/v4/annotated-average-test.csv",
    test_path=None,
    predict_path=None,
    output_path=f"sets/annotated/optimized/{NAME}/results.json",
    optimizer=partial(SimbaOptimizer, k=16),
    checkpoint_path=f"checkpoints/{NAME}"
)
