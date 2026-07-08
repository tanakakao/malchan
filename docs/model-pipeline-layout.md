# Model pipeline module layout

`malchan.pipeline` is the intended public home for high-level model pipeline classes.

Planned responsibilities:

- `malchan.pipeline.single_output`: single-target model pipeline entrypoint.
- `malchan.pipeline.multi_output`: multi-target model pipeline entrypoint.
- `malchan.pipeline.base`: shared protocols, base classes, and validation helpers for pipeline objects.

The implementation is intentionally moved in small steps.  The first step adds the new public package and routes top-level exports through it.  Follow-up PRs should move the actual implementation out of `malchan.models.models` after remaining old absolute imports are replaced.

Long-term target:

```text
src/malchan/
├── pipeline/
│   ├── base.py
│   ├── single_output.py
│   └── multi_output.py
├── models/
│   ├── registry.py
│   ├── defaults.py
│   ├── training.py
│   ├── tuning.py
│   └── evaluation.py
├── preprocessing/
└── features/
```
