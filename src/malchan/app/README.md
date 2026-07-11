# malchan FastAPI application

`malchan.app` provides an opt-in FastAPI layer for training and serving
single-output and multi-output `malchan` model pipelines through HTTP.

## Install and run

```bash
pip install -e ".[web]"
uvicorn "malchan.app:create_app" --factory --reload
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.
The API prefix defaults to `/api` and can be changed with
`MALCHAN_API_PREFIX`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Process health and package version |
| `POST` | `/api/models` | Train and register a single-output or multi-output model |
| `GET` | `/api/models` | List registered models |
| `GET` | `/api/models/{model_id}` | Read model metadata |
| `POST` | `/api/models/{model_id}/predict` | Run prediction or class-probability inference |
| `DELETE` | `/api/models/{model_id}` | Remove a registered model |

## Train a single-output regression model

The existing `target_col` and `task` request fields remain supported.

```bash
curl -X POST http://127.0.0.1:8000/api/models \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"x1": 0.1, "x2": 1.0, "y": 10.0},
      {"x1": 0.2, "x2": 0.9, "y": 12.0},
      {"x1": 0.3, "x2": 0.7, "y": 13.5},
      {"x1": 0.4, "x2": 0.4, "y": 16.0}
    ],
    "target_col": "y",
    "task": "regression",
    "num_cols": ["x1", "x2"],
    "cat_cols": [],
    "model_names": ["ランダムフォレスト回帰"]
  }'
```

## Train a multi-objective model

Use `target_cols` and `tasks` to train multiple outputs from the same input
features. The number and order of `tasks` must match `target_cols`.

`model_names` is applied to every target by default. Use
`model_names_by_target` when each objective should use a different estimator.
`model_params_by_target` and `base_model_params_by_target` provide the same
per-target override mechanism for predictor parameters.

```bash
curl -X POST http://127.0.0.1:8000/api/models \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"temperature": 700, "pressure": 1.0, "strength": 410, "cost": 120},
      {"temperature": 750, "pressure": 1.2, "strength": 445, "cost": 135},
      {"temperature": 800, "pressure": 1.4, "strength": 470, "cost": 151}
    ],
    "target_cols": ["strength", "cost"],
    "tasks": ["regression", "regression"],
    "num_cols": ["temperature", "pressure"],
    "cat_cols": [],
    "model_names": ["ランダムフォレスト回帰"],
    "model_names_by_target": {
      "cost": ["線形回帰"]
    },
    "model_params_by_target": {
      "strength": {"n_estimators": 200}
    }
  }'
```

The registered model metadata contains `target_cols`, `tasks`, and
`model_names_by_target`. For a single-output request, the legacy `target_col`,
`task`, and `model_names` response fields are also populated.

## Predict all objectives

The training response contains a `model_id`. Prediction returns one record per
input row containing every objective.

```bash
curl -X POST http://127.0.0.1:8000/api/models/<model_id>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"temperature": 775, "pressure": 1.3},
      {"temperature": 825, "pressure": 1.5}
    ]
  }'
```

Example response:

```json
{
  "model_id": "<model_id>",
  "predictions": [
    {"strength": 458.2, "cost": 143.1},
    {"strength": 481.7, "cost": 158.4}
  ]
}
```

Regression and classification targets can be mixed. When `"proba": true` is
specified, classification targets return class-probability columns while
regression targets continue to return their normal predictions.

## Current lifecycle behavior

The initial implementation stores fitted models in memory. Models are removed
when the process restarts and are not shared between multiple Uvicorn workers.
This is suitable for local analysis, prototypes, and a single-process internal
service. A production deployment should replace `InMemoryModelService` with a
persistent model registry and add authentication, request-size limits, and a
background job system for expensive training workloads.
