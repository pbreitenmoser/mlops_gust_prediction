"""Training pipeline: Feature_View -> time split -> fit -> evaluate -> save -> register."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.common import config, hopsworks_io, report
from src.common.config import EVENT_TIME, FEATURE_COLUMNS, LABEL_COLUMN, PRIMARY_KEY
from src.common.features import validate_feature_columns
from src.common.logging_utils import run_with_timeout, step
from src.common.metrics import mae, persistence_baseline_rmse, rmse, time_split
from src.common.settings import Settings

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_FILE_NAME = "gust_model.joblib"
FEATURE_COLUMNS_FILE_NAME = "feature_columns.json"

HYPERPARAMETERS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_samples_leaf": 50,
    "early_stopping": False,
    "random_state": 42,
}


class TrainingError(Exception):
    """A training pipeline step failed."""


def build_model() -> HistGradientBoostingRegressor:
    """Return the untrained model with the documented hyperparameters."""
    return HistGradientBoostingRegressor(**HYPERPARAMETERS)


def drop_unlabeled(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows with a null label. Returns (labeled frame, number of rows dropped)."""
    labeled = df.dropna(subset=[LABEL_COLUMN]).reset_index(drop=True)
    return labeled, len(df) - len(labeled)


def feature_columns_of(df: pd.DataFrame) -> list[str]:
    """Columns of the frame that are neither key, event time nor label (order preserved)."""
    return [c for c in df.columns if c not in (PRIMARY_KEY, EVENT_TIME, LABEL_COLUMN)]


def round2(value: float) -> float:
    return round(float(value), 2)


def compute_metrics(model: Any, test_df: pd.DataFrame) -> dict[str, float]:
    """RMSE, MAE (full test set) and persistence baseline RMSE, rounded to 2 decimals."""
    pred = model.predict(test_df[FEATURE_COLUMNS])
    y_true = test_df[LABEL_COLUMN]
    baseline, used = persistence_baseline_rmse(test_df)
    logger.info("baseline evaluated on %d of %d test rows (gust_now not null)", used, len(test_df))
    return {
        "rmse": round2(rmse(y_true, pred)),
        "mae": round2(mae(y_true, pred)),
        "baseline_rmse": round2(baseline),
    }


def save_model_atomically(model: Any, path: str | Path) -> Path:
    """Dump the model to `<path>.tmp`, then os.replace it onto `path`.

    If dumping fails the previous file at `path` stays untouched and the tmp file is removed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        joblib.dump(model, tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def save_feature_columns(path: str | Path, columns: list[str] = FEATURE_COLUMNS) -> Path:
    """Write the ordered feature column list as JSON (atomically)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(list(columns), indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _log_split(name: str, df: pd.DataFrame) -> None:
    logger.info(
        "%s set: rows=%d min_reference_timestamp=%s max_reference_timestamp=%s",
        name,
        len(df),
        df[EVENT_TIME].min(),
        df[EVENT_TIME].max(),
    )


def _report_data(frame: pd.DataFrame, labeled: pd.DataFrame, dropped: int) -> None:
    report.banner("TRAINING PIPELINE")
    report.section("Data")
    report.kv("Feature view", f"{config.FEATURE_VIEW_NAME} v{config.FEATURE_VIEW_VERSION}")
    report.kv("Feature group", f"{config.FEATURE_GROUP_NAME} v{config.FEATURE_GROUP_VERSION}")
    report.kv("Label", LABEL_COLUMN)
    report.kv("Features used", f"{len(FEATURE_COLUMNS)}")
    report.line("  aggregated: " + ", ".join(config.AGGREGATED_FEATURES))
    report.line("  realtime:   " + ", ".join(config.REALTIME_FEATURES))
    report.kv("Rows read", f"{len(frame):,}")
    report.kv("  labeled", f"{len(labeled):,}")
    report.kv("  unlabeled (dropped)", f"{dropped:,}")


def _report_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    total = len(train_df) + len(test_df)
    report.section("Split (time based, no shuffling)")
    report.table(
        ["set", "rows", "share", "from (UTC)", "to (UTC)"],
        [
            [
                name,
                f"{len(df):,}",
                f"{100 * len(df) / total:.1f}%",
                report.fmt_ts(df[EVENT_TIME].min()),
                report.fmt_ts(df[EVENT_TIME].max()),
            ]
            for name, df in (("train", train_df), ("test", test_df))
        ],
        first_width=6,
    )


def _report_label_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    report.section(f"Label distribution ({LABEL_COLUMN}, km/h)")
    rows = []
    for name, df in (("train", train_df), ("test", test_df)):
        y = df[LABEL_COLUMN]
        rows.append(
            [
                name,
                report.fmt_num(y.mean()),
                report.fmt_num(y.std()),
                report.fmt_num(y.median()),
                report.fmt_num(y.quantile(0.95)),
                report.fmt_num(y.max()),
            ]
        )
    report.table(["set", "mean", "std", "median", "p95", "max"], rows, first_width=6)


def _report_metrics(metrics: dict[str, float], n_test: int) -> None:
    report.section("Metrics (test set, km/h)")
    report.table(
        ["metric", "value"],
        [
            ["RMSE (model)", f"{metrics['rmse']:.2f}"],
            ["MAE (model)", f"{metrics['mae']:.2f}"],
            ["RMSE (persistence)", f"{metrics['baseline_rmse']:.2f}"],
        ],
    )
    base = metrics["baseline_rmse"]
    if base > 0:
        gain = 100 * (base - metrics["rmse"]) / base
        verdict = "yes" if metrics["rmse"] < base else "no"
        report.line()
        report.kv("Improvement vs baseline", f"{gain:+.1f}% RMSE")
        report.kv("Model beats baseline", verdict)
    report.kv("Test rows", f"{n_test:,}")


def run_training(settings: Settings, artifacts_dir: str | Path = ARTIFACTS_DIR) -> dict[str, float]:
    """Run the whole training pipeline. Raises on any failure. Returns the metrics."""
    artifacts_dir = Path(artifacts_dir)
    started = time.monotonic()

    with step("login"):
        project = hopsworks_io.login(settings)

    with step("feature_view"):
        fg = hopsworks_io.get_feature_group(project)
        fv = hopsworks_io.get_feature_view(project, fg)

    with step("read_training_data") as info:
        frame = hopsworks_io.read_training_frame(fv)
        info.records = len(frame)

    with step("drop_unlabeled") as info:
        labeled, dropped = drop_unlabeled(frame)
        info.records = len(labeled)
        logger.info("excluded %d unlabeled rows; %d labeled rows remain", dropped, len(labeled))
        _report_data(frame, labeled, dropped)

    with step("validate_and_split") as info:
        validate_feature_columns(feature_columns_of(labeled))
        try:
            train_df, test_df = time_split(labeled)
        except ValueError as exc:
            raise TrainingError(
                f"Too few labeled rows to train: found {len(labeled)} labeled rows ({exc})"
            ) from None
        info.records = len(labeled)
        _log_split("train", train_df)
        _log_split("test", test_df)
        _report_split(train_df, test_df)
        _report_label_distribution(train_df, test_df)

    with step("training", records=len(train_df)):
        model = build_model()
        run_with_timeout(
            lambda: model.fit(train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN]),
            config.TRAINING_TIMEOUT_S,
        )

    with step("evaluation", records=len(test_df)):
        metrics = compute_metrics(model, test_df)
        logger.info(
            "Model RMSE=%.2f MAE=%.2f baseline_rmse=%.2f (km/h)",
            metrics["rmse"],
            metrics["mae"],
            metrics["baseline_rmse"],
        )
        _report_metrics(metrics, len(test_df))

    with step("save_model"):
        save_model_atomically(model, artifacts_dir / MODEL_FILE_NAME)
        save_feature_columns(artifacts_dir / FEATURE_COLUMNS_FILE_NAME)

    with step("register_model"):
        if not (config.MODEL_NAME or "").strip():
            logger.error("Model name is empty (config.MODEL_NAME); skipping model upload")
            raise TrainingError("register_model: model name is empty")
        input_example = train_df[FEATURE_COLUMNS].head(1)
        run_with_timeout(
            lambda: hopsworks_io.register_model(
                project, fv, artifacts_dir, metrics, input_example
            ),
            config.MODEL_UPLOAD_TIMEOUT_S,
        )

    report.banner("TRAINING PIPELINE: SUCCESS")
    report.kv("Model", config.MODEL_NAME)
    report.kv("Total runtime", report.fmt_duration(time.monotonic() - started))
    return metrics
