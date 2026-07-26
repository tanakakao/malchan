"""数値・カテゴリ・分子特徴量を組み合わせる前処理Pipeline。"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import FastICA, KernelPCA, NMF, PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline, make_union
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
)

from skfp.fingerprints import (
    AtomPairFingerprint,
    AutocorrFingerprint,
    AvalonFingerprint,
    E3FPFingerprint,
    ECFPFingerprint,
    MACCSFingerprint,
    MORSEFingerprint,
    PhysiochemicalPropertiesFingerprint,
    PubChemFingerprint,
    RDFFingerprint,
    RDKit2DDescriptorsFingerprint,
)
from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer


FINGERPRINTS = {
    "ECFP": ECFPFingerprint(count=True),
    "MACCS": MACCSFingerprint(),
    "RDKit": RDKit2DDescriptorsFingerprint(),
    "PubChem": PubChemFingerprint(),
    "AtomPair": AtomPairFingerprint(),
    "Avalon": AvalonFingerprint(),
    "PhysChem": PhysiochemicalPropertiesFingerprint(),
    "Autocorr": AutocorrFingerprint(use_3D=True),
    "E3FP": E3FPFingerprint(),
    "MORSE": MORSEFingerprint(),
    "RDF": RDFFingerprint(),
}


def _identity(value: Any) -> Any:
    return value


def make_numeric_preprocess(
    impute_type: Optional[str] = None,
    scale_type: Optional[str] = None,
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


class SmilesToMol(BaseEstimator, TransformerMixin):
    """SMILESをRDKit Molへ変換し、必要に応じて配座を生成する。"""

    def __init__(
        self,
        *,
        generate_conformers: bool = False,
        error_on_invalid: bool = False,
        sanitize: bool = True,
        mol_n_jobs: int | None = 1,
        conf_num_conformers: int = 1,
        conf_errors: str = "ignore",
        conf_n_jobs: int | None = 1,
        random_state: int | None = 0,
        use_cache: bool = True,
        cache_max: int = 50000,
    ):
        self.generate_conformers = generate_conformers
        self.error_on_invalid = error_on_invalid
        self.sanitize = sanitize
        self.mol_n_jobs = mol_n_jobs
        self.conf_num_conformers = conf_num_conformers
        self.conf_errors = conf_errors
        self.conf_n_jobs = conf_n_jobs
        self.random_state = random_state
        self.use_cache = use_cache
        self.cache_max = cache_max

    def fit(self, X: Any, y: Any = None) -> "SmilesToMol":
        self.mol_transformer_ = MolFromSmilesTransformer(
            sanitize=self.sanitize,
            valid_only=False,
            n_jobs=self.mol_n_jobs,
        )
        self.conf_transformer_ = ConformerGenerator(
            num_conformers=self.conf_num_conformers,
            errors=self.conf_errors,
            n_jobs=self.conf_n_jobs,
            random_state=self.random_state,
        )
        self._mol_cache = {} if self.use_cache else None
        self._conf_cache = {} if self.use_cache else None
        return self

    def transform(self, X: Any) -> list[Any]:
        smiles = np.asarray(X).ravel().tolist()
        unique_smiles = list(dict.fromkeys(smiles))

        if self.use_cache:
            missing = [value for value in unique_smiles if value not in self._mol_cache]
            if missing:
                generated = self.mol_transformer_.transform(missing)
                self._mol_cache.update(dict(zip(missing, generated)))
            mol_by_smiles = self._mol_cache
        else:
            generated = self.mol_transformer_.transform(unique_smiles)
            mol_by_smiles = dict(zip(unique_smiles, generated))

        molecules = [mol_by_smiles[value] for value in smiles]
        if self.error_on_invalid and any(molecule is None for molecule in molecules):
            invalid_rows = [
                index
                for index, molecule in enumerate(molecules)
                if molecule is None
            ][:10]
            raise ValueError(f"Invalid SMILES at rows: {invalid_rows}")

        if not self.generate_conformers:
            return molecules

        if self.use_cache:
            missing_conf = [
                value
                for value in unique_smiles
                if value not in self._conf_cache
            ]
            valid_pairs = [
                (value, mol_by_smiles[value])
                for value in missing_conf
                if mol_by_smiles[value] is not None
            ]
            if valid_pairs:
                values, valid_molecules = zip(*valid_pairs)
                generated_conf = self.conf_transformer_.transform(list(valid_molecules))
                self._conf_cache.update(dict(zip(values, generated_conf)))
            for value in missing_conf:
                self._conf_cache.setdefault(value, None)

            if self.cache_max and len(self._mol_cache) > self.cache_max:
                self._mol_cache.clear()
                self._conf_cache.clear()
            return [self._conf_cache[value] for value in smiles]

        valid_pairs = [
            (value, mol_by_smiles[value])
            for value in unique_smiles
            if mol_by_smiles[value] is not None
        ]
        conformers: dict[Any, Any] = {value: None for value in unique_smiles}
        if valid_pairs:
            values, valid_molecules = zip(*valid_pairs)
            generated_conf = self.conf_transformer_.transform(list(valid_molecules))
            conformers.update(dict(zip(values, generated_conf)))
        return [conformers[value] for value in smiles]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(["mol"], dtype=object)


class PassthroughNames(BaseEstimator, TransformerMixin):
    """匿名配列へ安定した特徴量名を付与する。"""

    def fit(self, X: Any, y: Any = None) -> "PassthroughNames":
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: Any) -> Any:
        return X

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            return np.array(
                [f"feat_{index}" for index in range(self.n_features_in_)],
                dtype=object,
            )
        return np.asarray(input_features, dtype=object)


def make_smiles_preprocess(fingerprints: list[str] = []) -> Pipeline | None:
    """SMILES列向けのfingerprint生成Pipelineを作成する。"""
    if not fingerprints:
        return None

    unknown = [name for name in fingerprints if name not in FINGERPRINTS]
    if unknown:
        raise ValueError(
            f"未対応のfingerprintです: {unknown}. 利用可能: {sorted(FINGERPRINTS)}"
        )

    selected = [FINGERPRINTS[name] for name in fingerprints]
    generate_conformers = any(
        name in {"Autocorr", "E3FP", "MORSE", "RDF"}
        for name in fingerprints
    )
    return Pipeline(
        [
            ("to_mol", SmilesToMol(generate_conformers=generate_conformers)),
            ("fp", make_union(*selected)),
            ("sc", StandardScaler(with_mean=False)),
            ("names", PassthroughNames()),
        ]
    )


def make_preprocess_pipeline(
    num_process: Pipeline,
    cat_process: Pipeline,
    smiles_process: Pipeline | None = None,
    comp_process: Pipeline | None = None,
    numcat_common_preprocess: Any | None = None,
    common_process: Any | None = None,
    num_cols: list[str] = [],
    cat_cols: list[str] = [],
    smiles_cols: list[str] = [],
    comp_cols: list[str] = [],
) -> ImbalancedPipeline:
    """列種別の前処理をColumnTransformerへ統合する。"""
    transforms: list[tuple[str, Any, list[str]]] = []
    numcat_transforms: list[tuple[str, Any, list[str]]] = []

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
    num_cols: list[str] = [],
    cat_cols: list[str] = [],
    num_impute_type: Optional[str] = None,
    num_scale_type: Optional[str] = None,
    cat_impute: bool = False,
    smiles_cols: list[str] = [],
    fingerprints: list[str] = [],
    comp_cols: list[str] = [],
    comp_method: str | None = None,
    comp_feats: list[str] = [],
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
    smiles_process = (
        make_smiles_preprocess(fingerprints=fingerprints)
        if smiles_cols
        else None
    )

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
    "FINGERPRINTS",
    "PassthroughNames",
    "SmilesToMol",
    "make_categorical_preprocess",
    "make_common_preprocess",
    "make_numcat_common_preprocess",
    "make_numeric_preprocess",
    "make_preprocess",
    "make_preprocess_pipeline",
    "make_smiles_preprocess",
]
