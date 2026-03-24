"""
Training entrypoint: shared Phase 3 features, compare models, MLflow + local artifacts.

Selection rule: **lowest validation RMSE** (primary); MAE and R² are reported for context.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from data_engineer.feature_engineering import get_feature_metadata, transform_to_model_features
from data_engineer.preprocessing import load_training_dataframe
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from model_trainer import evaluate, infer_signature, register, reporting
from model_trainer.config import TrainingConfig, load_training_config
from model_trainer.pipelines import EstimatorName, TargetMode, build_supervised_pipeline
from model_trainer.split import train_val_split

logger = logging.getLogger(__name__)

ARTIFACT_PATH = "sklearn-model"


def select_best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Deterministic best run: minimize validation **RMSE**, then **MAE**, then maximize **R²**.

    Implemented as lexicographic minimization on ``(rmse, mae, -r2)`` so ties on RMSE
    resolve predictably (lower MAE wins; if still tied, higher R² wins).
    """
    if not rows:
        raise ValueError("no model rows to select from")
    return min(
        rows,
        key=lambda r: (float(r["rmse"]), float(r["mae"]), -float(r["r2"])),
    )


def _prepare_xy(
    merged: pd.DataFrame,
    *,
    reference_year: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Aligned feature matrix (Phase 3) and price target."""
    y = merged["price"].copy()
    X = transform_to_model_features(merged, reference_year=reference_year)
    if not X.index.equals(y.index):
        raise ValueError("Feature/target index mismatch after transform_to_model_features")
    return X, y


def _maybe_log1p_name(mode: TargetMode) -> str:
    return "log1p(price)" if mode == "log1p" else "price"


def run_training(
    *,
    config_path: Path | None = None,
    raw_data_dir: Path | None = None,
    output_dir: Path | None = None,
    use_mlflow: bool = True,
    models_filter: list[str] | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """
    Load data, train configured models, pick best by validation RMSE, save artifacts.

    Returns a summary dict including ``rows`` (metric table) and ``best_key``.
    """
    cfg: TrainingConfig = load_training_config(config_path)
    out = Path(output_dir or Path("model"))
    register.ensure_model_dir(out)

    meta = get_feature_metadata()
    (out / "feature_metadata.json").write_text(
        json.dumps(meta.to_dict(), indent=2),
        encoding="utf-8",
    )

    merged = load_training_dataframe(raw_data_dir)
    if max_rows is not None:
        merged = merged.iloc[:max_rows].copy()

    X, y = _prepare_xy(merged, reference_year=cfg.reference_year)
    X_train, X_val, y_train, y_val = train_val_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )

    run_list = cfg.models
    if models_filter:
        run_list = [m for m in run_list if m.name in models_filter]

    if not run_list:
        raise RuntimeError("No models to train (empty config or filter).")

    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline | TransformedTargetRegressor] = {}

    if use_mlflow:
        if cfg.mlflow_tracking_uri:
            mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.experiment_name)

    def _run_one(
        pipe: Pipeline | TransformedTargetRegressor,
        key: str,
        name: EstimatorName,
        target_mode: TargetMode,
    ) -> dict[str, Any]:
        cv_rmse_mean: float | None = None
        if cfg.cv_folds and cfg.cv_folds > 1:
            cv_scores = cross_val_score(
                pipe,
                X_train,
                y_train,
                cv=cfg.cv_folds,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            cv_rmse_mean = float(-np.mean(cv_scores))

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        metrics = evaluate.evaluate_regression(y_val, y_pred)
        fitted_models[key] = pipe

        row: dict[str, Any] = {
            "key": key,
            "model": name,
            "target_mode": target_mode,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "cv_rmse_mean": cv_rmse_mean,
            "run_id": None,
            "model_uri": None,
        }
        return row

    if use_mlflow:
        with mlflow.start_run(run_name="training_comparison"):
            mlflow.log_params(
                {
                    "reference_year": cfg.reference_year,
                    "test_size": cfg.test_size,
                    "random_state": cfg.random_state,
                    "n_rows": len(merged),
                    "n_features": X.shape[1],
                },
            )
            mlflow.log_param("feature_names_csv", ",".join(meta.final_feature_columns))
            mlflow.log_text(json.dumps(meta.to_dict(), indent=2), "feature_metadata.json")

            for model_cfg in run_list:
                name = model_cfg.name
                target_mode = model_cfg.target_mode
                key = f"{name}_{target_mode}"
                pipe = build_supervised_pipeline(
                    name,
                    metadata=meta,
                    target_mode=target_mode,
                    random_state=cfg.random_state,
                )

                with mlflow.start_run(run_name=key, nested=True) as child_run:
                    row = _run_one(pipe, key, name, target_mode)
                    row["run_id"] = child_run.info.run_id
                    if row["cv_rmse_mean"] is not None:
                        mlflow.log_metric("cv_rmse_mean", row["cv_rmse_mean"])
                    mlflow.log_metrics(
                        evaluate.metrics_to_mlflow(
                            {k: row[k] for k in ("mae", "rmse", "r2") if k in row},
                        ),
                    )
                    mlflow.log_params(
                        {
                            "estimator": name,
                            "target_mode": target_mode,
                            "target_transform": _maybe_log1p_name(target_mode),
                        },
                    )
                    sample = X_val.iloc[: min(5, len(X_val))]
                    pred_sample = pipe.predict(sample)
                    sig = infer_signature.infer_signature_from_predictions(sample, pred_sample)
                    ml_info = mlflow.sklearn.log_model(
                        pipe,
                        artifact_path=ARTIFACT_PATH,
                        signature=sig,
                    )
                    # Use MLflow-returned URI (correct for nested runs / MLflow 2.x model store).
                    row["model_uri"] = ml_info.model_uri
                    rows.append(row)

    else:
        for model_cfg in run_list:
            name = model_cfg.name
            target_mode = model_cfg.target_mode
            key = f"{name}_{target_mode}"
            pipe = build_supervised_pipeline(
                name,
                metadata=meta,
                target_mode=target_mode,
                random_state=cfg.random_state,
            )
            row = _run_one(pipe, key, name, target_mode)
            rows.append(row)

    best_row = select_best_row(rows)
    best_key = best_row["key"]
    selection_rule = (
        "Primary: **minimum validation RMSE** on the hold-out split. "
        "Tie-break (deterministic): lower **MAE**, then higher **R²** "
        "(implemented as lexicographic min on `(rmse, mae, -r2)` in `select_best_row`)."
    )
    reporting.write_comparison_report(
        out / "training_report.md",
        title="House price — model comparison",
        selection_rule=selection_rule,
        rows=rows,
        best_run_name=f"{best_row['model']} ({best_row['target_mode']})",
    )

    joblib.dump(fitted_models[best_key], out / "best_model.joblib")
    logger.info("Saved best model (%s) to %s", best_key, out / "best_model.joblib")

    if use_mlflow and best_row.get("model_uri"):
        register.register_model_from_uri(
            str(best_row["model_uri"]),
            cfg.registered_model_name,
            tracking_uri=cfg.mlflow_tracking_uri,
        )

    return {
        "output_dir": str(out),
        "rows": rows,
        "best": best_row,
        "best_key": best_key,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train house price regressors (Phase 4)")
    parser.add_argument("--config", type=Path, default=None, help="YAML config path")
    parser.add_argument("--raw-dir", type=Path, default=None, help="data/raw directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("model"),
        dest="output_dir",
        help="Local artifact directory",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows (debug/smoke)")
    args = parser.parse_args(argv)

    models_filter = [m.strip() for m in args.models.split(",")] if args.models else None
    cfg_path = args.config
    if cfg_path is None:
        default_cfg = Path("config/model_config.yaml")
        cfg_path = default_cfg if default_cfg.exists() else None

    run_training(
        config_path=cfg_path,
        raw_data_dir=args.raw_dir,
        output_dir=args.output_dir,
        use_mlflow=not args.no_mlflow,
        models_filter=models_filter,
        max_rows=args.max_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
