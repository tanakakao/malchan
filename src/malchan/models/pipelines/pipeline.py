from sklearn.pipeline import Pipeline

from .preprocess_pipeline import make_preprocess
from .predictor_pipeline import make_predictor


def make_pipeline(
    model_names,
    task,
    num_cols,
    cat_cols,
    smiles_cols,
    comp_cols,
    num_impute_type,
    num_scale_type,
    cat_impute,
    fingerprints,
    comp_method,
    comp_feats,
    poly,
    poly_degree,
    poly_interaction_only,
    decomposition,
    decomposition_method,
    n_components,
    ensemble,
    ens_type,
    base_model,
    model_params,
    base_model_params
):
    """
    データ前処理とモデルを組み合わせたパイプラインを作成する関数。
    """
    preprocess = make_preprocess(
        model_name=model_names[0] if len(model_names) > 0 else base_model,
        num_cols=num_cols,
        cat_cols=cat_cols,
        smiles_cols=smiles_cols,
        comp_cols=comp_cols,
        num_impute_type=num_impute_type,
        num_scale_type=num_scale_type,
        cat_impute=cat_impute,
        fingerprints=fingerprints,
        comp_method=comp_method,
        comp_feats=comp_feats,
        poly=poly,
        poly_degree=poly_degree,
        poly_interaction_only=poly_interaction_only,
        decomposition = decomposition,
        decomposition_method = decomposition_method,
        n_components=n_components,
        ensemble=ensemble
    )

    predictor = make_predictor(
        model_names=model_names,
        ensemble=ensemble,
        ens_type=ens_type,
        base_model=base_model,
        model_params=model_params,
        base_model_params=base_model_params,
        task=task
    )

    model = Pipeline(
        steps=[
            ('preprocess', preprocess),
            ('predictor', predictor)
        ]
    )
    return model, preprocess, predictor
