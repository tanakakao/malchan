# malchan FastAPI application

`malchan.app` provides an opt-in FastAPI layer for training, serving, and
inverse-analyzing single-output and multi-output `malchan` models through HTTP.

## Install and run

Prediction-only usage requires the `web` extra. Inverse analysis also requires
Optuna from the `inverse` extra.

```bash
pip install -e ".[web,inverse]"
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
| `POST` | `/api/models/{model_id}/inverse-analysis` | Search input candidates with Optuna |
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

## Run inverse analysis

Inverse analysis searches the registered model's input space with Optuna. Each
objective can be minimized, maximized, or optimized toward a requested numeric
value or classification label.

```bash
curl -X POST \
  http://127.0.0.1:8000/api/models/<model_id>/inverse-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "objectives": [
      {"target": "strength", "direction": "max"},
      {"target": "cost", "direction": "min"}
    ],
    "sampler_type": "NSGAII",
    "bounds": {
      "temperature": {
        "min": 680,
        "max": 850,
        "dtype": "int",
        "step": 10
      },
      "pressure": {
        "min": 0.8,
        "max": 1.6,
        "dtype": "float",
        "step": 0.05
      }
    },
    "trials": 500,
    "n_candidates": 20
  }'
```

Example response:

```json
{
  "model_id": "<model_id>",
  "objectives": [
    {"target": "strength", "direction": "max", "target_value": null},
    {"target": "cost", "direction": "min", "target_value": null}
  ],
  "candidates": [
    {
      "temperature": 810,
      "pressure": 1.25,
      "pred_strength": 482.4,
      "pred_cost": 146.8
    }
  ],
  "n_trials": 500,
  "n_completed_trials": 500,
  "pareto_size": 17
}
```

### Search controls

- `bounds` overrides the observed training-data range for numeric features.
- `categories` sets allowed values for categorical, SMILES, or composition
  inputs. When omitted, observed training values are used.
- `fixed_values` removes selected features from the search and fixes them to a
  specified value.
- `sum_constraint` requires selected numeric columns to sum to one value. It is
  useful for composition ratios.
- `sampler_type` supports `TPE`, `MOTPE`, `CmaEs`, `GP`, `QMS`, `NSGAII`, and
  `NSGAIII`.

For a regression objective, specify either `direction` or `target_value`:

```json
{"target": "strength", "target_value": 450.0}
```

A classification objective must use `target_value` with a fitted class label:

```json
{"target": "quality", "target_value": "OK"}
```

A composition-style equality constraint can be specified as follows:

```json
{
  "sum_constraint": {
    "columns": ["component_a", "component_b", "component_c"],
    "value": 1.0
  },
  "fixed_values": {
    "component_c": 0.2
  }
}
```

Inverse analysis currently runs synchronously in the API request. Large trial
counts can therefore occupy one server worker for a long time. A production
service should move expensive searches to a persistent background-job system.

## Current lifecycle behavior

The initial implementation stores fitted models in memory. Models are removed
when the process restarts and are not shared between multiple Uvicorn workers.
This is suitable for local analysis, prototypes, and a single-process internal
service. A production deployment should replace `InMemoryModelService` with a
persistent model registry and add authentication, request-size limits, and a
background job system for expensive training and inverse-analysis workloads.
