# malchan FastAPI application

`malchan.app` provides an opt-in FastAPI layer for training, serving,
comparing, tuning, and inverse-analyzing single-output and multi-output
`malchan` models through HTTP.

## Install and run

Model training and comparison require the `models` extra. Inverse analysis also
requires Optuna from the `inverse` extra.

```bash
pip install -e ".[web,models,inverse]"
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
| `POST` | `/api/models/{model_id}/compare` | Compare model families and optionally tune the best |
| `GET` | `/api/models/{model_id}/comparison` | Read the latest comparison and tuning state |
| `POST` | `/api/models/{model_id}/comparison/tune-best` | Tune selected best models after comparison |
| `POST` | `/api/models/{model_id}/inverse-analysis` | Search input candidates with Optuna |
| `DELETE` | `/api/models/{model_id}` | Remove a registered model and its comparison state |

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

Regression and classification targets can be mixed. When `"proba": true` is
specified, classification targets return class-probability columns while
regression targets continue to return normal predictions.

## Compare candidate models

Comparison reuses the registered model's training data, feature definitions,
material featurizers, and preprocessing settings. Regression candidates are
ranked by test RMSE by default and classification candidates by test F1.

```bash
curl -X POST http://127.0.0.1:8000/api/models/<model_id>/compare \
  -H "Content-Type: application/json" \
  -d '{
    "model_names": [
      "線形回帰",
      "Ridge",
      "ランダムフォレスト回帰",
      "LightGBM"
    ],
    "method": "kfold",
    "n_splits": 5
  }'
```

Example response:

```json
{
  "model_id": "<model_id>",
  "targets": {
    "y": {
      "target": "y",
      "metric": "RMSE",
      "higher_is_better": false,
      "ranking": [
        {
          "rank": 1,
          "model_name": "LightGBM",
          "target": "y",
          "task": "regression",
          "test_RMSE": 1.23
        }
      ],
      "failures": {},
      "best_model_name": "LightGBM",
      "best_params": null,
      "best_is_tuned": false,
      "best_cv_scores": {
        "train": [{"RMSE": 0.52, "MAE": 0.41, "MAPE": 0.03, "R2": 0.97}],
        "test": [{"RMSE": 1.23, "MAE": 0.95, "MAPE": 0.07, "R2": 0.84}]
      }
    }
  }
}
```

Use `metric` to change the ranking metric and `model_params` to supply
candidate-specific parameters.

### Compare and tune only the best model

`tune_best` runs the recommended two-stage workflow: compare untuned model
families fairly, select the best family, tune only that family, and evaluate the
tuned model with the same CV settings.

```bash
curl -X POST http://127.0.0.1:8000/api/models/<model_id>/compare \
  -H "Content-Type: application/json" \
  -d '{
    "model_names": ["Ridge", "ランダムフォレスト回帰", "LightGBM"],
    "n_splits": 5,
    "tune_best": true,
    "tuning_trials": 100,
    "tuning_verbose": 0
  }'
```

The original `ranking` remains the untuned comparison record. The response's
`best_model_name`, `best_params`, `best_is_tuned`, and `best_cv_scores` describe
the selected and tuned model.

`tuning: true` instead tunes every candidate before comparison. Because this is
more expensive, `tuning` and `tune_best` cannot both be true.

### Tune the best model later

Run comparison first, inspect the result, and then tune the selected candidate:

```bash
curl -X POST \
  http://127.0.0.1:8000/api/models/<model_id>/comparison/tune-best \
  -H "Content-Type: application/json" \
  -d '{
    "n_trials": 100,
    "verbose": 0,
    "evaluate": true
  }'
```

Set `evaluate` to `false` to skip the post-tuning CV. Calling this endpoint
before `/compare` returns HTTP `409`.

The latest state can be fetched without rerunning comparison:

```bash
curl http://127.0.0.1:8000/api/models/<model_id>/comparison
```

### Multi-output comparison and tuning

Candidate lists, ranking metrics, and trial counts can be specified by target.

```bash
curl -X POST http://127.0.0.1:8000/api/models/<model_id>/compare \
  -H "Content-Type: application/json" \
  -d '{
    "model_names": {
      "strength": ["Ridge", "ランダムフォレスト回帰"],
      "cost": ["線形回帰", "LightGBM"]
    },
    "metric": {
      "strength": "R2",
      "cost": "RMSE"
    },
    "tune_best": true,
    "tuning_trials": {
      "strength": 100,
      "cost": 50
    }
  }'
```

A deferred request can tune only selected outputs:

```bash
curl -X POST \
  http://127.0.0.1:8000/api/models/<model_id>/comparison/tune-best \
  -H "Content-Type: application/json" \
  -d '{
    "targets": ["strength"],
    "n_trials": {"strength": 150},
    "evaluate": true
  }'
```

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

### Search controls

- `bounds` overrides the observed training-data range for numeric features.
- `categories` sets allowed values for categorical, SMILES, or composition
  inputs. When omitted, observed training values are used.
- `fixed_values` fixes selected features to specified values.
- `sum_constraint` requires selected numeric columns to sum to one value.
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

## Current lifecycle behavior

Fitted models, comparison results, tuned models, and inverse-analysis state are
stored in memory. They are removed when the process restarts and are not shared
between multiple Uvicorn workers.

Training, comparison, tuning, and inverse analysis currently run synchronously.
Large candidate sets, high Optuna trial counts, or expensive CV settings can
occupy one server worker for a long time. A production deployment should use a
persistent model registry, authentication, request-size limits, and a
background-job system for expensive workloads.
