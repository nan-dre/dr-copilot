import dspy
from typing import Optional
import dspy.primitives
import dspy.primitives.program
from serde import serde
from functools import partial

@serde
class ModelSettings():
    model: str
    api_base: str
    model_type: str
    cache: bool
    api_key: Optional[str] = None

@serde
class Config():

    seed: int

    # Mlflow configs
    mlflow_url: str
    experiment_name: str
    run_name: str
    
    model_settings: ModelSettings

    # Loader settings
    train_path: Optional[str]
    val_path: Optional[str]
    test_path: Optional[str]
    predict_path: Optional[str]
    output_path: str
    limit: Optional[int]

    recommender_settings: ModelSettings | None = None
    optimizer: Optional[partial] = None
    
    scorer_model_uri: Optional[str] = None
    predict_module: type[dspy.primitives.program.Module] = dspy.Predict
    checkpoint_path: Optional[str] = None
