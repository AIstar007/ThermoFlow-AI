"""Optional local MLflow tracking of a reproducible thermal analysis run."""
from __future__ import annotations


def log_analysis_run(summary: dict, settings: dict, metrics: dict | None = None,
                     experiment_name: str = "thermal-pattern-propagation") -> str:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow is optional. Install with: pip install -e '.[tracking]'") from exc
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="thermal-sequence-analysis") as run:
        mlflow.log_params({k: str(v)[:500] for k, v in settings.items()
                           if isinstance(v, (str, int, float, bool)) or v is None})
        flat = metrics or {}
        for key, value in flat.items():
            if isinstance(value, (int, float)) and value == value:
                mlflow.log_metric(key, float(value))
        mlflow.set_tag("application", "Thermal Pattern Propagation Lab")
        mlflow.set_tag("scientific_scope", "surface-temperature patterns; not direct perfusion")
        mlflow.log_dict(summary, "analysis_summary.json")
        return run.info.run_id
