# malchan FastAPI application

`malchan.app` provides an opt-in FastAPI layer for training and serving a
single-output `SingleOutputMLModelPipeline` through HTTP.

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
| `POST` | `/api/models` | Train and register a single-output model |
| `GET` | `/api/models` | List registered models |
| `GET` | `/api/models/{model_id}` | Read model metadata |
| `POST` | `/api/models/{model_id}/predict` | Run prediction or class-probability inference |
| `DELETE` | `/api/models/{model_id}` | Remove a registered model |

## Train a regression model

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

The response contains a `model_id`. Use it for prediction:

```bash
curl -X POST http://127.0.0.1:8000/api/models/<model_id>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"x1": 0.25, "x2": 0.8},
      {"x1": 0.45, "x2": 0.3}
    ]
  }'
```

For classification probabilities, set `"proba": true` in the prediction
request.

## Current lifecycle behavior

The initial implementation stores fitted models in memory. Models are removed
when the process restarts and are not shared between multiple Uvicorn workers.
This is suitable for local analysis, prototypes, and a single-process internal
service. A production deployment should replace `InMemoryModelService` with a
persistent model registry and add authentication, request-size limits, and a
background job system for expensive training workloads.
