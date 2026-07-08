# Preprocessing module layout

`malchan.preprocessing` is the intended public home for preprocessing builders.

The current implementation still lives in `malchan.models.pipelines`, but it should be split by responsibility in follow-up PRs.

Target responsibilities:

- `malchan.preprocessing.compose`: compose preprocessing and estimator pipelines.
- `malchan.preprocessing.numeric`: numeric preprocessing such as imputation, scaling, and polynomial features.
- `malchan.preprocessing.categorical`: categorical preprocessing such as encoding and categorical imputation.
- `malchan.preprocessing.missing`: missing-value handling policies.
- `malchan.preprocessing.scaling`: scaling helpers and scaling registry.

Longer term, chemistry and materials feature generation should move out of preprocessing into `malchan.features` so heavy optional dependencies remain isolated.
