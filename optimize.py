import dspy
import argparse
import mlflow
import logging
import json
from pathlib import Path
from serde import to_dict
from importlib import import_module
from models.prompt_score_v4 import (
    DoctorResponseScorerModule,
    field_to_evaluator,
    metric_map,
    type_map,
)
from mlflow.models import ModelSignature
from configs.base import Config
from dataloaders.prompt_score_v2_loader import PromptScoreV2Loader
from optimizers.base import BaseOptimizer

logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)


def path_to_module(path: str):
    return path.rstrip(".py").replace("/", ".")


def main():
    parser = argparse.ArgumentParser(description="Optimize evaluator models")
    parser.add_argument("config_path", help="Path to the config file")
    args = parser.parse_args()

    cfg: Config = import_module(path_to_module(args.config_path)).config

    dspy.settings.configure(lm=dspy.LM(**to_dict(cfg.model_settings)))
    dataloader = PromptScoreV2Loader(cfg=cfg)

    mlflow.set_tracking_uri(cfg.mlflow_url)
    mlflow.set_experiment(cfg.experiment_name)
    mlflow.dspy.autolog()  # type: ignore

    result_dict = {}

    with mlflow.start_run(run_name=cfg.run_name):
        optimized_models = {}

        for field, evaluator in field_to_evaluator.items():
            print(f"\n\n{'='*50}")
            print(f"Optimizing {field} evaluator")
            print(f"{'='*50}")

            predictor = cfg.predict_module(evaluator)

            # Create optimizer with the corresponding metric function
            metric_fn = metric_map.get(field, None)

            assert cfg.optimizer
            optimizer = cfg.optimizer(
                field=field, field_type=type_map[field], metric_fn=metric_fn
            )

            # Filter train and val examples that have labels for this field
            train_loader = dataloader.train_dataloader(field)
            val_loader = dataloader.val_dataloader(field)
            print(
                f"Using {len(train_loader)} training examples and {len(val_loader)} validation examples for {field}"
            )

            # Optimize the predictor
            (
                optimized_model,
                base_score,
                results,
                all_scores,
                optimized_score,
                optimized_results,
                optimized_all_scores,
            ) = optimizer.optimize(
                predictor, train_set=train_loader, val_set=val_loader
            )

            result_dict[field] = {
                "base_score": base_score,
                "optimized_score": optimized_score,
                "results": results,
                "all_scores": all_scores,
                "optimized_results": optimized_results,
                "optimized_all_scores": optimized_all_scores,
            }

            optimized_models[field] = optimized_model

            print(f"Logging {field} model to MLflow...")
            artifact_path = f"model_{field}"

            model_info = mlflow.dspy.log_model(  # type: ignore
                optimized_model,
                artifact_path=artifact_path,
                code_paths=["models/prompt_score_v3.py"],
            )
            print(model_info.model_uri)

            mlflow.log_param(f"model_uri_{field}", model_info.model_uri)
            mlflow.log_metric(f"{field}_base_score", base_score)
            mlflow.log_metric(f"{field}_optimized_score", optimized_score)
            mlflow.log_metric(f"{field}_improvement", optimized_score - base_score)

        # Save result_dict
        print("SAVED FIELDS:")
        print(optimized_models.keys())
        Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.output_path, "w") as f:
            json.dump(result_dict, f)
        # Create a combined scorer with all optimized models
        optimized_scorer = DoctorResponseScorerModule(cfg)
        optimized_scorer.scorers = optimized_models
        if cfg.checkpoint_path:
            optimized_scorer.save(cfg.checkpoint_path, save_program=True)

        # Log the complete optimized scorer model
        print("\nLogging the complete optimized scorer model to MLflow...")
        combined_model_info = mlflow.dspy.log_model(  # type: ignore
            optimized_scorer,
            artifact_path="optimized_scorer",
            code_paths=["models/prompt_score_v3.py"],
        )
        mlflow.log_artifact(args.config_path)

        # Log the combined model URI
        mlflow.log_param("combined_model_uri", combined_model_info.model_uri)

        print("\nOptimization complete!")


if __name__ == "__main__":
    main()
