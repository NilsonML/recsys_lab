"""MLflow tracking helpers."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import mlflow
from mlflow.entities import Run
from omegaconf import DictConfig, OmegaConf


def flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping into dot-separated keys: {"a": {"b": 1}} -> {"a.b": 1}."""
    out: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


@contextmanager
def start_run(cfg: DictConfig) -> Iterator[Run]:
    """Open an MLflow run and log the fully resolved config as params and an artifact."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        mlflow.log_params(flatten({str(k): v for k, v in container.items() if k != "mlflow"}))
        mlflow.log_text(OmegaConf.to_yaml(cfg, resolve=True), "config.yaml")
        yield run
