# recsys-lab

Top-K recommendation on MovieLens-1M: baselines → matrix factorization → neural models.
See [docs/design_note.md](docs/design_note.md).

## Setup

```powershell
poetry install
poetry run pre-commit install
```

## Run

```powershell
poetry run python -m recsys.pipelines.train_eval                     # defaults from configs/config.yaml
poetry run python -m recsys.pipelines.train_eval model=ease seed=7   # override
poetry run python -m recsys.pipelines.train_eval -m model=popularity,ease,bpr_mf  # sweep
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db         # http://127.0.0.1:5000
```

## Checks

```powershell
poetry run ruff check . ; poetry run ruff format --check . ; poetry run mypy ; poetry run pytest
```
