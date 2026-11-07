"""All Hopsworks calls live here (signatures verified against hopsworks 5.2.3)."""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import (
    EVENT_TIME,
    FEATURE_COLUMNS,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    LABEL_COLUMN,
    MODEL_NAME,
    PRIMARY_KEY,
)
from src.common.selection import select_latest_version
from src.common.settings import Settings

logger = logging.getLogger(__name__)


class HopsworksError(Exception):
    """A Hopsworks interaction failed. Carries the step name and the cause."""

    def __init__(self, step: str, cause: BaseException | str):
        self.step = step
        self.cause = cause
        super().__init__(f"Hopsworks step '{step}' failed: {cause}")


def _wrap(step: str, exc: BaseException) -> HopsworksError:
    if isinstance(exc, HopsworksError):
        return exc
    return HopsworksError(step, f"{type(exc).__name__}: {exc}")


def _to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Treat `reference_timestamp` as UTC (tz-naive values are assumed UTC)."""
    if EVENT_TIME in df.columns:
        df = df.copy()
        df[EVENT_TIME] = pd.to_datetime(df[EVENT_TIME], utc=True)
    return df


def _strip_key_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Hopsworks prefixes key/event-time columns that are not selected as features
    (e.g. `<fg>_<version>_station`). Rename them back to their plain names."""
    rename = {}
    for col in df.columns:
        for plain in (PRIMARY_KEY, EVENT_TIME):
            if col != plain and col.endswith("_" + plain) and plain not in df.columns:
                rename[col] = plain
    return df.rename(columns=rename) if rename else df


def login(settings: Settings) -> Any:
    """Log in to the configured project. Returns the Project."""
    import hopsworks

    try:
        return hopsworks.login(
            project=settings.project_name,
            api_key_value=settings.api_key,
        )
    except Exception as exc:  # noqa: BLE001
        raise _wrap("login", exc) from None


def get_feature_group(project: Any) -> Any:
    """Get or create the offline-only feature group (PK station, event time reference_timestamp)."""
    try:
        fs = project.get_feature_store()
        return fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            primary_key=[PRIMARY_KEY],
            event_time=EVENT_TIME,
            online_enabled=False,
            description="Wind gust features for MeteoSwiss station BER (10-minute grid, UTC). "
            "Source: MeteoSwiss.",
        )
    except Exception as exc:  # noqa: BLE001
        raise _wrap("get_feature_group", exc) from None


def insert_rows(fg: Any, df: pd.DataFrame) -> None:
    """Upsert rows on (station, reference_timestamp) and wait for materialization."""
    try:
        df = _to_utc(df)
        # operation="upsert" is the default; wait=True blocks until the job finished.
        fg.insert(df, operation="upsert", wait=True)
    except Exception as exc:  # noqa: BLE001
        raise _wrap("insert_rows", exc) from None


def get_feature_view(project: Any, fg: Any) -> Any:
    """Get or create the feature view: 16 features + label, no label filter.

    `project` may be a Project or a FeatureStore (anything with get_or_create_feature_view).
    """
    try:
        fs = project if hasattr(project, "get_or_create_feature_view") else project.get_feature_store()
        query = fg.select(FEATURE_COLUMNS + [LABEL_COLUMN])
        return fs.get_or_create_feature_view(
            name=FEATURE_VIEW_NAME,
            version=FEATURE_VIEW_VERSION,
            query=query,
            labels=[LABEL_COLUMN],
            description="Features and label gust_max_next_1h for station BER.",
        )
    except Exception as exc:  # noqa: BLE001
        raise _wrap("get_feature_view", exc) from None


def read_training_frame(fv: Any) -> pd.DataFrame:
    """Return one frame with station, reference_timestamp, features and label (unlabeled rows included)."""
    try:
        x, y = fv.training_data(primary_key=True, event_time=True)
        x = pd.DataFrame(x)
        if y is None:
            df = x
        else:
            y = pd.DataFrame(y)
            if LABEL_COLUMN in x.columns:
                df = x
            else:
                if y.shape[1] == 1 and y.columns[0] != LABEL_COLUMN:
                    y = y.set_axis([LABEL_COLUMN], axis=1)
                df = pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        return _to_utc(_strip_key_prefix(df))
    except Exception as exc:  # noqa: BLE001
        raise _wrap("read_training_frame", exc) from None


def read_latest_rows(fv: Any, since: datetime | pd.Timestamp) -> pd.DataFrame:
    """Batch-read rows with event time >= `since` from the offline store (label column included)."""
    try:
        start = pd.Timestamp(since)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        df = fv.get_batch_data(
            start_time=start.to_pydatetime(),
            primary_key=True,
            event_time=True,
        )
        return _to_utc(_strip_key_prefix(pd.DataFrame(df)))
    except Exception as exc:  # noqa: BLE001
        raise _wrap("read_latest_rows", exc) from None


def register_model(
    project: Any,
    fv: Any,
    local_model_path: str | Path,
    metrics: dict[str, float],
    input_example: pd.DataFrame | None = None,
) -> Any:
    """Register the model directory with metrics (rmse, mae, baseline_rmse) and feature view reference.

    The registry assigns the version (previous max + 1).
    """
    try:
        if not MODEL_NAME:
            raise ValueError("MODEL_NAME is empty")
        mr = project.get_model_registry()
        model = mr.sklearn.create_model(
            name=MODEL_NAME,
            metrics={k: float(v) for k, v in metrics.items()},
            description=(
                f"HistGradientBoostingRegressor for gust_max_next_1h; "
                f"feature view {FEATURE_VIEW_NAME} v{FEATURE_VIEW_VERSION}."
            ),
            input_example=input_example,
            feature_view=fv,
        )
        return model.save(str(local_model_path))
    except Exception as exc:  # noqa: BLE001
        raise _wrap("register_model", exc) from None


def download_latest_model(project: Any, target_dir: str | Path | None = None) -> tuple[Any, Path]:
    """Download the model with the highest explicit version. Returns (model, local dir)."""
    try:
        mr = project.get_model_registry()
        models = mr.get_models(MODEL_NAME)
        model = select_latest_version(models)  # ValueError("model not found") if empty
        local = str(target_dir) if target_dir is not None else tempfile.mkdtemp(prefix="gust_model_")
        path = model.download(local_path=local)
        return model, Path(path)
    except Exception as exc:  # noqa: BLE001
        raise _wrap("download_latest_model", exc) from None
