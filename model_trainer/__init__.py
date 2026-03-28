"""Model training, evaluation, and registration."""

from model_trainer.evaluate import evaluate_regression, format_metrics_line, metrics_to_mlflow
from model_trainer.infer_signature import infer_signature_from_predictions
from model_trainer.pipelines import build_estimator, build_supervised_pipeline
from model_trainer.register import register_model_from_uri
from model_trainer.split import train_val_split
from model_trainer.train import run_training

__all__ = [
    "build_estimator",
    "build_supervised_pipeline",
    "evaluate_regression",
    "format_metrics_line",
    "infer_signature_from_predictions",
    "metrics_to_mlflow",
    "register_model_from_uri",
    "run_training",
    "train_val_split",
]
