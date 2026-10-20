from pathlib import Path

import mlflow
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from recsys.pipelines.train_eval import CONFIG_DIR, run
from recsys.utils.seed import seed_everything
from recsys.utils.tracking import flatten


def _compose(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def test_seed_everything_is_reproducible() -> None:
    a = seed_everything(123).random(5)
    b = seed_everything(123).random(5)
    np.testing.assert_array_equal(a, b)


def test_flatten() -> None:
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}


@pytest.mark.parametrize("model", ["popularity", "ease", "bpr_mf"])
def test_config_composes(model: str) -> None:
    cfg = _compose([f"model={model}"])
    assert cfg.model.name == model
    assert cfg.seed == 42


def test_run_logs_to_mlflow(tmp_path: Path) -> None:
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    cfg = _compose([f"mlflow.tracking_uri={uri}", "model=ease"])

    run_id = run(cfg)

    logged = mlflow.MlflowClient(tracking_uri=uri).get_run(run_id)
    assert logged.data.params["model.name"] == "ease"
    assert logged.data.params["seed"] == "42"
    assert "sanity.seeded_uniform" in logged.data.metrics
