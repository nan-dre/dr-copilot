import dspy
import json
import argparse
import mlflow
import logging
from serde import to_dict
from importlib import import_module
from tqdm import tqdm
from pathlib import Path
from models.prompt_score_v4 import (
    DoctorResponseScorerModule,
    field_to_evaluator,
    metric_map,
)
from models.recommender_v2 import RecommenderModule
from models.reconciliator import ReconciliatorModule
from configs.base import Config
from dataloaders.recommendation_loader import RecommendationLoader

logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)


def path_to_module(path: str):
    """
    Converts a file path to a Python module path.
    """
    return path.rstrip(".py").replace("/", ".")


def main():
    parser = argparse.ArgumentParser(description="Optimize evaluator models")
    parser.add_argument("config_path", help="Path to the config file")
    args = parser.parse_args()

    cfg: Config = import_module(path_to_module(args.config_path)).config

    # Track experiment with MLflow
    mlflow.set_tracking_uri(cfg.mlflow_url)
    mlflow.set_experiment(cfg.experiment_name)
    mlflow.dspy.autolog()

    dspy.settings.configure(lm=dspy.LM(**to_dict(cfg.model_settings)))
    # scorer = DoctorResponseScorerModule(cfg)
    assert cfg.scorer_model_uri
    scorer = mlflow.pyfunc.load_model(cfg.scorer_model_uri)
    recommender = RecommenderModule(cfg)
    response_reconciliator = ReconciliatorModule(cfg)
    dataloader = RecommendationLoader(cfg)
    predict_loader = dataloader.predict_dataloader()

    results = []

    with mlflow.start_run(run_name=cfg.run_name):
        # Process each sample
        print(f"\n\n{'='*50}")
        print(f"Evaluating recommender")
        print(f"{'='*50}")
        for sample in tqdm(predict_loader):
            base_score = scorer.predict(
                {
                    "patient_question": sample.patient_question,
                    "doctor_response": sample.doctor_response,
                }
            )
            recommendations = recommender(
                field="all",
                score=base_score,
                patient_question=sample.patient_question,
                doctor_response=sample.doctor_response,
            )
            modifier_response = response_reconciliator(
                patient_question=sample.patient_question,
                doctor_response=sample.doctor_response,
                recommendations=recommendations,
            )
            modified_response_score = scorer.predict(
                {
                    "patient_question": sample.patient_question,
                    "doctor_response": modifier_response,
                }
            )
            results.append(
                {
                    "base_id": sample.base_id,
                    "patient_question": sample.patient_question,
                    "doctor_response": sample.doctor_response,
                    "recommendations": recommendations,
                    "modified_response": modifier_response,
                    "base_score": base_score,
                    "recommendation_score": modified_response_score,
                }
            )

        # Save results
        Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.output_path, "w") as f:
            json.dump(results, f)
        mlflow.log_artifact(cfg.output_path)


if __name__ == "__main__":
    main()
