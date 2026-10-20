"""Train a model and evaluate it, logging everything to MLflow.

Usage:
    poetry run python -m recsys.pipelines.train_eval model=ease split=temporal_global
"""

from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig

from recsys.utils.seed import seed_everything
from recsys.utils.tracking import start_run

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


def run(cfg: DictConfig) -> str:
    """Run the pipeline for a composed config and return the MLflow run id."""
    rng = seed_everything(cfg.seed)
    with start_run(cfg) as active:
        # Data loading, training and evaluation are not implemented yet; this metric
        # only checks that seeding works (same seed -> same value across runs).
        mlflow.log_metric("sanity.seeded_uniform", float(rng.random()))
        return str(active.info.run_id)


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
