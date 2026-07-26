"""数値・カテゴリ・外部特徴量を組み合わせる前処理Pipeline。"""

from __future__ import annotations

from typing import Any

from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import FastICA, KernelPCA, NMF, PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
)


def _identity(value: Any) -> Any:
    return value


def make_numeric_preprocess(
    impute_type: str | None = None,
    scale_type: str | None = None,
) -> Pipeline:
    """数値列の欠損補完・スケーリングPipelineを作成する。"""
    imputer = {
        "Multiple": IterativeImputer(),
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "most_frequent": SimpleImputer(strategy="most_frequent"),
        "knn": KNNImputer(),
    }.get(impute_type)
    scaler = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "centering": StandardScaler(with_std=False),
        "MaxAbsScaler": MaxAbsScaler(),
    }.get(scale_type)

    steps: list[tuple[str, Any]] = []
    if imputer is not None:
        steps.append(("imputer", imputer))
    if scaler is not None:
        steps.append(("scaler", scaler))
    else:
        steps.append(
            (
                "identity",
                FunctionTransformer(
                    _identity,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            )
        )
    return Pipeline(steps=steps)


def make_categorical_preprocess(
    model_name: str,
    impute: bool = False,
    poly: bool = False,
    pca: bool = False,
    ensemble: bool = False,
) -> Pipeline:
    """カテゴリ列の補完・エンコードPipelineを作成する。"""
    steps: list[tuple[str, Any]] = []
    if impute:
        steps.append(("imputer", SimpleImputer(strategy="most_frequent")))

    if model_name == "LightGBM" and not poly and not pca and not ensemble:
        steps.append(("ordinal", OrdinalEncoder()))
    elif model_name == "CatBoost" and not poly and not pca and not ensemble:
        steps.append(
            (
                "identity",
                FunctionTransformer(
                    _identity,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            )
        )
    else:
        steps.append(
            (
                "one-hot",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
            )
        )
    return Pipeline(steps=steps)


def make_numcat_common_preprocess(
    poly: bool = False,
    degree: int = 1,
    interaction_only: bool = True,
) -> PolynomialFeatures | None:
    """数値・カテゴリ結合後の多項式特徴量変換を作成する。"""
    if not poly:
        return None
    return PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
    )


def make_common_preprocess(
    decomposition: bool = False,
    decomposition_method: str = "PCA",
    n_components: int = 2,
) -> Any | None:
    """全特徴量結合後の次元削減器を作成する。"""
    if not decomposition:
        return None

    if decomposition_method == "PCA":
        return PCA(n_components=n_components)
    if decomposition_method in {"KernelPCA", "KernalPCA"}:
        return KernelPCA(n_components=n_components, kernel="rbf")
    if decomposition_method == "NMF":
        return NMF(n_components=n_components)
    if decomposition_method == "ICA":
        return FastICA(n_components=n_components)
    raise ValueError(
        "不正なdecomposition_methodです。PCA、KernelPCA、NMF、ICAから指定してください。"
    )


def make_preprocess_pipeline(
    num_process: Pipeline,
    cat_process: Pipeline,
    smiles_process: Pipeline | None = None,
    comp_process: Pipeline | None = None,
    numcat_common_preprocess: Any | None = None,
    common_process: Any | None = None,
    num_cols: list[str] | tuple[str, ...] = (),
    cat_cols: list[str] | tuple[str, ...] = (),
    smiles_cols: list[str] | tuple[str, ...] = (),
    comp_cols: list[str] | tuple[str, ...] = (),
) -> ImbalancedPipeline:
    """列種別の前処理をColumnTransformerへ統合する。"""
    transforms: list[tuple[str, Any, list[str] | tuple[str, ...]]] = []
    numcat_transforms: list[tuple[str, Any, list[str] | tuple[str, ...]]] = []

    if num_cols:
        numcat_transforms.append(("num", num_process, num_cols))
    if cat_cols:
        numcat_transforms.append(("cat", cat_process, cat_cols))
    if numcat_transforms:
        numcat_steps: list[tuple[str, Any]] = [
            (
                "num_cat_prerprocess",
                ColumnTransformer(transformers=numcat_transforms),
            )
        ]
        if numcat_common_preprocess is not None:
            numcat_steps.append(("num_cat_common", numcat_common_preprocess))
        transforms.append(
            (
                "num_cat",
                Pipeline(numcat_steps),
                [*num_cols, *cat_cols],
            )
        )

    if smiles_cols:
        if smiles_process is None:
            raise ValueError("smiles_colsを指定する場合はfingerprintsを指定してください。")
        transforms.append(("smiles", smiles_process, smiles_cols))

    if comp_cols:
        if comp_process is None:
            raise ValueError("comp_colsを指定する場合はcomp_methodを指定してください。")
        transforms.append(("comp", comp_process, comp_cols))

    steps: list[tuple[str, Any]] = [
        ("column_preprocess", ColumnTransformer(transformers=transforms))
    ]
    if common_process is not None:
        steps.append(("common_preprocess", common_process))
    return ImbalancedPipeline(steps=steps)


def make_preprocess(
    model_name: str,
    num_cols: list[str] | tuple[str, ...] = (),
    cat_cols: list[str] | tuple[str, ...] = (),
    num_impute_type: str | None = None,
    num_scale_type: str | None = None,
    cat_impute: bool = False,
    smiles_cols: list[str] | tuple[str, ...] = (),
    fingerprints: list[str] | tuple[str, ...] = (),
    comp_cols: list[str] | tuple[str, ...] = (),
    comp_method: str | None = None,
    comp_feats: list[str] | tuple[str, ...] = (),
    poly: bool = False,
    poly_degree: int = 1,
    poly_interaction_only: bool = True,
    decomposition: bool = False,
    decomposition_method: str = "PCA",
    n_components: int = 2,
    ensemble: bool = False,
) -> ImbalancedPipeline:
    """モデル設定から前処理Pipeline全体を作成する。"""
    num_process = make_numeric_preprocess(
        impute_type=num_impute_type,
        scale_type=num_scale_type,
    )
    cat_process = make_categorical_preprocess(
        model_name=model_name,
        impute=cat_impute,
        poly=poly,
        pca=decomposition,
        ensemble=ensemble,
    )

    smiles_process = None
    if smiles_cols:
        from malchan.features.chemistry.pipeline import make_smiles_preprocess

        smiles_process = make_smiles_preprocess(fingerprints=fingerprints)

    comp_process = None
    if comp_cols:
        from malchan.features.materials.pipeline import make_comp_preprocess

        comp_process = make_comp_preprocess(
            method=comp_method,
            feats=comp_feats,
        )

    return make_preprocess_pipeline(
        num_process=num_process,
        cat_process=cat_process,
        smiles_process=smiles_process,
        comp_process=comp_process,
        numcat_common_preprocess=make_numcat_common_preprocess(
            poly=poly,
            degree=poly_degree,
            interaction_only=poly_interaction_only,
        ),
        common_process=make_common_preprocess(
            decomposition=decomposition,
            decomposition_method=decomposition_method,
            n_components=n_components,
        ),
        num_cols=num_cols,
        cat_cols=cat_cols,
        smiles_cols=smiles_cols,
        comp_cols=comp_cols,
    )


__all__ = [
    "make_categorical_preprocess",
    "make_common_preprocess",
    "make_numcat_common_preprocess",
    "make_numeric_preprocess",
    "make_preprocess",
    "make_preprocess_pipeline",
]
