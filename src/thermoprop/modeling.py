"""Grouped validation for research-only thermal-to-reference regression.

Every row must represent an independent participant/session/ROI summary, not a
video frame. Grouped cross-validation prevents observations from the same
participant leaking into both train and validation folds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelEvaluation:
    model_name: str
    target_column: str
    group_column: str
    feature_columns: list[str]
    metrics: dict[str, float]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    warnings: list[str]
    fitted_model: Any


def feature_table_template() -> bytes:
    example = pd.DataFrame(columns=[
        "participant_id", "session_id", "roi_name", "reference_perfusion",
        "mean_temperature", "temperature_sd", "mean_event_delta",
        "max_warming_rate", "max_cooling_rate", "rewarming_tau_s",
        "event_temperature_auc", "warming_coverage", "cooling_coverage",
        "ambient_temperature_c", "relative_humidity_percent", "acclimatization_min",
    ])
    return example.to_csv(index=False).encode("utf-8")


def _build_estimator(model_name: str, random_state: int = 42):
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import ElasticNet

    if model_name == "ElasticNet":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("model", ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=10000))])
    if model_name == "Random Forest":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                            max_features=0.8, random_state=random_state, n_jobs=-1))])
    if model_name == "Extra Trees":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("model", ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2,
                            max_features=0.8, random_state=random_state, n_jobs=-1))])
    if model_name == "HistGradientBoosting":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("model", HistGradientBoostingRegressor(max_iter=250, l2_regularization=1.0,
                            random_state=random_state))])
    if model_name == "XGBoost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("XGBoost is optional. Install with: pip install -e '.[ml-xgboost]'") from exc
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("model", XGBRegressor(n_estimators=350, max_depth=3, learning_rate=0.035,
                            subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror",
                            reg_lambda=3.0, random_state=random_state, n_jobs=-1))])
    raise ValueError(f"Unsupported model: {model_name}")


def train_grouped_reference_model(
    table: pd.DataFrame,
    *,
    target_column: str,
    group_column: str,
    model_name: str = "Random Forest",
    feature_columns: list[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> ModelEvaluation:
    """Train with grouped OOF validation. Requires at least 6 distinct groups.

    Input/schema and minimum-sample checks intentionally run before importing
    scikit-learn. This keeps validation useful in the core installation where
    the optional ``ml`` extra has not been installed. A sufficiently large
    training request reports an actionable optional-dependency message.
    """
    if target_column not in table or group_column not in table:
        raise ValueError("The table must include the selected target and participant/group columns")
    frame = table.copy()
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame[group_column] = frame[group_column].astype(str).str.strip()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame[frame[target_column].notna() & frame[group_column].ne("")]
    if feature_columns is None:
        excluded = {target_column, group_column, "session_id", "roi_name", "notes"}
        feature_columns = [c for c in frame.select_dtypes(include=[np.number]).columns if c not in excluded]
    feature_columns = [c for c in feature_columns if c in frame.columns and c not in {target_column, group_column}]
    if len(feature_columns) < 2:
        raise ValueError("Select at least two numeric thermal/acquisition feature columns")
    groups = frame[group_column].to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 6:
        raise ValueError(f"At least 6 distinct participant groups are required for grouped validation; found {len(unique_groups)}")
    if len(frame) < 12:
        raise ValueError("At least 12 participant/session/ROI rows are required for an exploratory model")
    X = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = frame[target_column].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Target contains non-finite values after cleaning")
    if np.nanstd(y) <= 1e-12:
        raise ValueError("Target column is constant; regression cannot be trained")

    # scikit-learn is deliberately an optional extra. Delay its import until
    # the request passes all data checks so small/invalid datasets receive the
    # useful validation error instead of an unrelated ModuleNotFoundError.
    try:
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.inspection import permutation_importance
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for reference-model training but is not installed. "
            "Install it with: pip install -e '.[ml]'"
        ) from exc

    splits = min(int(n_splits), len(unique_groups))
    splitter = GroupKFold(n_splits=splits)
    oof = np.full(len(frame), np.nan, dtype=float)
    fold_ids = np.full(len(frame), -1, dtype=int)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        model = _build_estimator(model_name, random_state + fold)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = model.predict(X.iloc[test_idx])
        fold_ids[test_idx] = fold
    valid = np.isfinite(oof)
    rmse = float(np.sqrt(mean_squared_error(y[valid], oof[valid])))
    mae = float(mean_absolute_error(y[valid], oof[valid]))
    r2 = float(r2_score(y[valid], oof[valid])) if valid.sum() > 1 else float("nan")
    corr = float(np.corrcoef(y[valid], oof[valid])[0, 1]) if valid.sum() > 2 and np.std(oof[valid]) > 1e-12 else float("nan")

    final_model = _build_estimator(model_name, random_state)
    final_model.fit(X, y)
    # Permutation importance is evaluated on the same full data only as a
    # descriptive ranking; it is not an unbiased estimate of generalization.
    try:
        importance = permutation_importance(final_model, X, y, n_repeats=8,
                                            random_state=random_state, scoring="neg_mean_absolute_error")
        imp_df = pd.DataFrame({"feature": feature_columns,
                               "importance_mae_increase": importance.importances_mean,
                               "importance_std": importance.importances_std}).sort_values(
                                   "importance_mae_increase", ascending=False)
    except Exception:
        imp_df = pd.DataFrame({"feature": feature_columns, "importance_mae_increase": np.nan})
    warnings = [
        "Research-only regression. Output predicts the uploaded reference target, not a directly measured thermal blood-flow quantity.",
        "Metrics are grouped out-of-fold estimates, but external validation on an independent cohort and site is still required.",
        "Permutation feature importance is calculated on the full training table and is descriptive, not a generalization estimate.",
        "Rows should be participant/session/ROI summaries; never upload every video frame as an independent sample.",
    ]
    prediction_frame = pd.DataFrame({
        "participant_group": groups,
        "observed_reference": y,
        "oof_prediction": oof,
        "fold": fold_ids,
        "residual": y - oof,
    })
    metrics = {"rows": float(len(frame)), "participants": float(len(unique_groups)),
               "folds": float(splits), "mae": mae, "rmse": rmse, "r2": r2, "pearson_r_oof": corr}
    return ModelEvaluation(model_name, target_column, group_column, feature_columns,
                           metrics, prediction_frame, imp_df, warnings, final_model)


def explain_tree_model(fitted_model: Any, X: pd.DataFrame, *, max_samples: int = 256,
                       random_state: int = 42) -> pd.DataFrame:
    """Return global SHAP importance for a fitted sklearn tree-regression pipeline.

    This explains the fitted model on the supplied cohort. It is not causal
    evidence, and the explanation sample is not an independent validation set.
    SHAP is an optional dependency installed with the ``explainability`` extra.
    """
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("Provide a non-empty feature DataFrame for explanation.")
    if int(max_samples) < 5 or int(max_samples) > 5000:
        raise ValueError("max_samples must be between 5 and 5000.")
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is optional. Install it with: pip install -e '.[explainability]'") from exc
    if not hasattr(fitted_model, "named_steps") or "model" not in fitted_model.named_steps:
        raise ValueError("Expected a fitted sklearn pipeline with a 'model' step.")
    estimator = fitted_model.named_steps["model"]
    # Keep this function deliberately limited to tree explainers with stable
    # regression output shapes. Other estimators need a suitable explainer.
    supported = ("RandomForestRegressor", "ExtraTreesRegressor", "XGBRegressor")
    if estimator.__class__.__name__ not in supported:
        raise ValueError(f"SHAP tree explanation is not enabled for {estimator.__class__.__name__}.")
    imputer = fitted_model.named_steps.get("imputer")
    transformed = imputer.transform(X) if imputer is not None else X.to_numpy(dtype=float)
    if len(transformed) > int(max_samples):
        indices = np.linspace(0, len(transformed) - 1, int(max_samples), dtype=int)
        transformed = transformed[indices]
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or values.shape[1] != len(X.columns):
        raise ValueError(f"Unexpected SHAP output shape {values.shape} for {len(X.columns)} features.")
    return pd.DataFrame({
        "feature": list(X.columns),
        "mean_absolute_shap": np.mean(np.abs(values), axis=0),
        "median_absolute_shap": np.median(np.abs(values), axis=0),
    }).sort_values("mean_absolute_shap", ascending=False, ignore_index=True)
