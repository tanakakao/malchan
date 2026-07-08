import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, MaxAbsScaler, PolynomialFeatures,
    FunctionTransformer, OrdinalEncoder, OneHotEncoder, LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_union

from rdkit import Chem
from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer
from skfp.fingerprints import ECFPFingerprint, MACCSFingerprint, RDKit2DDescriptorsFingerprint, MordredFingerprint, PubChemFingerprint, AtomPairFingerprint
from types import SimpleNamespace
from pathlib import Path
import shutil
from xenonpy._conf import __cfg_root__
from xenonpy.datatools import preset
from xenonpy.descriptor import Compositions
from pymatgen.core import Composition as PMGComposition
from matminer.featurizers.composition import (
    ElementProperty, OxidationStates, ValenceOrbital, AtomicOrbitals, YangSolidSolution,
    IonProperty, TMetalFraction, Stoichiometry, ElementFraction,
    Meredig, BandCenter, Miedema, CohesiveEnergy, WenAlloys, Stoichiometry, ElectronAffinity
)
from matminer.featurizers.base import MultipleFeaturizer
from mendeleev.fetch import fetch_table

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.impute import KNNImputer

from sklearn.decomposition import PCA, KernelPCA, NMF, FastICA

from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_sample_weight

from collections import Counter
import copy
from typing import List, Optional, Union, Dict, Tuple, Callable, Any, Type, Hashable
import warnings

from ..utils import FP_dcit, INORG_dict

warnings.simplefilter('ignore')
def _func(x):
    return x

def make_numeric_preprocess(
    impute_type: Optional[str] = None,
    scale_type: Optional[str] = None
) -> Pipeline:
    """
    数値データの前処理パイプラインを作成する関数。

    Args:
        impute_type (Optional[str]): 欠損値の補完方法を指定する文字列。
            "Multiple", "mean", "median", "most_frequent", "knn" から選択可能。
        scale_type (Optional[str]): スケーリング方法を指定する文字列。
            "StandardScaler", "MinMaxScaler", "centering" から選択可能。

    Returns:
        Pipeline: 指定された補完方法およびスケーリング方法を含む前処理パイプライン。
    """
    # 補完方法の辞書を定義し、指定されたタイプの補完方法を取得
    imputer = {
        "Multiple": IterativeImputer(),
        "mean": SimpleImputer(strategy='mean'),
        "median": SimpleImputer(strategy='median'),
        "most_frequent": SimpleImputer(strategy='most_frequent'),
        "knn": KNNImputer()
    }.get(impute_type)

    # スケーリング方法の辞書を定義し、指定されたタイプのスケーラーを取得
    num_scaler = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "centering": StandardScaler(with_std=False),
        "MaxAbsScaler": MaxAbsScaler()
    }.get(scale_type)

    # パイプラインのステップリストを初期化
    num_steps = []

    # 補完方法が指定されていればステップに追加
    if imputer:
        num_steps.append(('imputer', imputer))

    # スケーラーが指定されていればステップに追加、なければ元のデータをそのまま返す関数を追加
    if num_scaler:
        num_steps.append(('scaler', num_scaler))
    else:
        num_steps.append(('identity', FunctionTransformer(_func, validate=False, feature_names_out='one-to-one')))

    # パイプラインを作成して返す
    return Pipeline(steps=num_steps)

def _fun_trans(x):
    return x

def make_categorical_preprocess(
    model_name: str,
    impute: bool = False,
    poly: bool = False,
    pca: bool = False,
    ensemble: bool = False
) -> Pipeline:
    """
    カテゴリカルデータの前処理パイプラインを作成する関数。

    Args:
        model_name (str): 使用するモデルの名前。'LightGBM' または 'CatBoost' など。
        impute (bool): 欠損値の補完を行うかどうかを指定するフラグ。デフォルトは False。
        poly (bool): 多項式特徴量を使用するかどうかを指定するフラグ。デフォルトは False。
        pca (bool): 主成分分析（PCA）を行うかどうかを指定するフラグ。デフォルトは False。
        ensemble (bool): アンサンブル学習を行うかどうかを指定するフラグ。デフォルトは False。

    Returns:
        Pipeline: 指定された前処理方法を含むカテゴリカルデータの前処理パイプライン。
    """
    # パイプラインのステップリストを初期化
    cat_steps = []

    # 欠損値の補完を行う場合、SimpleImputerを追加
    if impute:
        cat_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))

    # モデルの名前とその他のフラグに基づいて前処理を選択
    if model_name == 'LightGBM' and not poly and not pca and not ensemble:
        cat_steps.append(('ordinal', OrdinalEncoder()))
    elif model_name == 'CatBoost' and not poly and not pca and not ensemble:
        cat_steps.append(('identity', FunctionTransformer(_fun_trans, validate=False, feature_names_out='one-to-one')))
    else:
        cat_steps.append(('one-hot', OneHotEncoder(drop='first', handle_unknown='ignore')))

    # パイプラインを作成して返す
    return Pipeline(steps=cat_steps)

def make_numcat_common_preprocess(
    poly: bool = False,
    degree: int = 1,
    interaction_only: bool = True,
) -> Optional[Pipeline]:
    """
    共通の前処理パイプラインを作成する関数。

    Args:
        poly (bool): 多項式特徴量を作成するかどうかを指定するフラグ。デフォルトは False。
        degree (int): 多項式特徴量の次数。デフォルトは 1。
        interaction_only (bool): 相互作用項のみを含めるかどうかを指定するフラグ。デフォルトは True。
        pca (bool): 主成分分析（PCA）を行うかどうかを指定するフラグ。デフォルトは False。
        n_components (int): PCAで使用する主成分の数。デフォルトは 2。

    Returns:
        Optional[Pipeline]: 指定された前処理方法を含む共通の前処理パイプライン。
                            何も指定されなければ None を返す。
    """
    # パイプラインのステップリストを初期化
    common_transforms = []

    # 多項式特徴量の作成を行う場合、PolynomialFeaturesを追加
    if poly:
        # common_transforms.append(('poly', PolynomialFeatures(degree=degree, interaction_only=interaction_only)))
        common_transforms = PolynomialFeatures(degree=degree, interaction_only=interaction_only)

    # パイプラインを作成して返す。ステップがない場合は None を返す。
    return common_transforms if common_transforms else None

def make_common_preprocess(
    decomposition: bool = False,
    decomposition_method: str = "PCA",
    n_components: int = 2
) -> Optional[Pipeline]:
    """
    共通の前処理パイプラインを作成する関数。

    Args:
        poly (bool): 多項式特徴量を作成するかどうかを指定するフラグ。デフォルトは False。
        degree (int): 多項式特徴量の次数。デフォルトは 1。
        interaction_only (bool): 相互作用項のみを含めるかどうかを指定するフラグ。デフォルトは True。
        pca (bool): 主成分分析（PCA）を行うかどうかを指定するフラグ。デフォルトは False。
        n_components (int): PCAで使用する主成分の数。デフォルトは 2。

    Returns:
        Optional[Pipeline]: 指定された前処理方法を含む共通の前処理パイプライン。
                            何も指定されなければ None を返す。
    """
    # パイプラインのステップリストを初期化
    common_transforms = []

    # PCAを行う場合、PCAを追加
    if decomposition:
        # common_transforms.append(('pca', PCA(n_components=n_components)))
        if decomposition_method=="PCA":
            common_transforms = PCA(n_components=n_components)
        elif decomposition_method=="KernalPCA":
            common_transforms = PCA(n_components=n_components, kernel="rbf")
        elif decomposition_method=="NMF":
            common_transforms = NMF(n_components=n_components)
        elif decomposition_method=="ICA":
            common_transforms = FastICA(n_components=n_components)
        else:
            raise ValueError("不正なdecomposition_methodです。PCA,KernelPCA,NMF,ICAから指定してください。")

    # パイプラインを作成して返す。ステップがない場合は None を返す。
    return common_transforms if common_transforms else None

# class SmilesToMol(BaseEstimator, TransformerMixin):
#     def __init__(self, drop_invalid=False): self.drop_invalid = drop_invalid
#     def fit(self, X, y=None): return self
#     def transform(self, X):
#         vals = np.ravel(X)
#         mols = [Chem.MolFromSmiles(str(s)) if s is not None else None for s in vals]
#         # mols = [MolFromSmilesTransformer().transform(str(s)) if s is not None else None for s in vals]
#         if self.drop_invalid and any(m is None for m in mols):
#             bad = [i for i,m in enumerate(mols) if m is None][:5]
#             raise ValueError(f"Invalid SMILES at rows: {bad}")
#         return np.array(mols, dtype=object)
#     def get_feature_names_out(self, input_features=None):
#         # 1列をそのまま次工程へ渡す想定
#         return np.array(["mol"], dtype=object)

# class SmilesToMol(BaseEstimator, TransformerMixin):
#     def __init__(self, drop_invalid=False):
#         self.drop_invalid = drop_invalid
#         # skfpのMolFromSmilesTransformerのインスタンスを初期化時に作成し、保持します。
#         # このトランスフォーマーは引数を必要としないことが多いです。
#         self.mol_transformer = MolFromSmilesTransformer()
#         self.conf_transformer = ConformerGenerator()

#     def fit(self, X, y=None):
#         # MolFromSmilesTransformerはデータから学習するパラメータがないため、
#         # fitメソッドは特に何もする必要がありません。
#         # self.mol_transformer.fit(X, y) を呼び出しても動作しますが、内部でpassなので冗長です。
#         return self

#     def transform(self, X):
#         # 入力Xは通常、Nx1の配列（例: Pandas DataFrameの1列）として渡されます。
#         # np.ravel(X)を使って1次元のSMILES文字列の配列に変換します。
#         smiles_strings = np.ravel(X)

#         # skfp.preprocessing.MolFromSmilesTransformerのtransformメソッドは、
#         # SMILES文字列の配列を受け取り、RDKitのMolオブジェクトの配列を返します。
#         # 無効なSMILESやNoneが入力された場合、対応する位置にNoneを格納したMolオブジェクトの配列を返します。
#         mols = self.mol_transformer.transform(smiles_strings)
#         mols = self.conf_transformer.transform(mols)

#         # 無効なSMILES（MolFromSmilesTransformerがNoneを返したもの）があるかチェックします。
#         if self.drop_invalid and any(m is None for m in mols):
#             bad = [i for i, m in enumerate(mols) if m is None][:5]
#             raise ValueError(f"Invalid SMILES at rows: {bad}")

#         # MolFromSmilesTransformerは既にnumpy.ndarray(dtype=object)を返しているため、
#         # 再度np.array()でラップする必要はありません。
#         return mols

#     def get_feature_names_out(self, input_features=None):
#         # 1列をそのまま次工程へ渡す想定
#         return np.array(["mol"], dtype=object)

class SmilesToMol(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        generate_conformers: bool = False,
        error_on_invalid: bool = False,
        # Mol生成
        sanitize: bool = True,
        mol_n_jobs: int | None = 1,
        # Conformer生成（必要時のみ）
        conf_num_conformers: int = 1,
        conf_errors: str = "ignore",     # "raise"|"ignore"|"filter"（"filter"は長さが変わるので注意）:contentReference[oaicite:4]{index=4}
        conf_n_jobs: int | None = 1,
        random_state: int | None = 0,
        # キャッシュ
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

        self.mol_transformer = MolFromSmilesTransformer(
            sanitize=self.sanitize,
            valid_only=False,            # 無効は None で返す :contentReference[oaicite:5]{index=5}
            n_jobs=self.mol_n_jobs,
        )
        self.conf_transformer = ConformerGenerator(
            num_conformers=self.conf_num_conformers,
            errors=self.conf_errors,
            n_jobs=self.conf_n_jobs,
            random_state=self.random_state,
        )  # conf_id を Mol に保存する :contentReference[oaicite:6]{index=6}

    def fit(self, X, y=None):
        self._mol_cache = {} if self.use_cache else None
        self._conf_cache = {} if self.use_cache else None
        return self

    def transform(self, X):
        smiles = np.asarray(X).ravel().tolist()

        # --- (1) Mol 生成（ユニークSMILESだけ） ---
        if self.use_cache:
            need = [s for s in dict.fromkeys(smiles) if s not in self._mol_cache]
        else:
            need = list(dict.fromkeys(smiles))

        if need:
            mols_new = self.mol_transformer.transform(need)  # list[Mol|None] :contentReference[oaicite:7]{index=7}
            if self.use_cache:
                for s, m in zip(need, mols_new):
                    self._mol_cache[s] = m

        mols = [self._mol_cache[s] if self.use_cache else
                self.mol_transformer.transform([s])[0] for s in smiles]

        # invalidチェック（必要なら）
        if self.error_on_invalid and any(m is None for m in mols):
            bad = [i for i, m in enumerate(mols) if m is None][:10]
            raise ValueError(f"Invalid SMILES at rows: {bad}")

        # --- (2) Conformer 生成（必要時のみ、Noneはスキップ） ---
        if not self.generate_conformers:
            return mols

        # 既に conf 付きならキャッシュから復元
        if self.use_cache:
            need_conf = [s for s in dict.fromkeys(smiles) if s not in self._conf_cache]
        else:
            need_conf = list(dict.fromkeys(smiles))

        if need_conf:
            # None を落として conformer 化
            valid_pairs = [(s, self._mol_cache[s] if self.use_cache else None) for s in need_conf]
            valid_pairs = [(s, m) for s, m in valid_pairs if m is not None]

            if valid_pairs:
                ss, mm = zip(*valid_pairs)
                mm_conf = self.conf_transformer.transform(list(mm))  # list[Mol] :contentReference[oaicite:8]{index=8}

                if self.use_cache:
                    for s, m in zip(ss, mm_conf):
                        self._conf_cache[s] = m

            if self.use_cache:
                # None のSMILESはそのまま None を入れておく
                for s in need_conf:
                    if s not in self._conf_cache:
                        self._conf_cache[s] = None

            # キャッシュ肥大化対策（雑）
            if self.use_cache and self.cache_max and len(self._mol_cache) > self.cache_max:
                self._mol_cache.clear()
                self._conf_cache.clear()

        mols_conf = [self._conf_cache[s] if self.use_cache else m for s, m in zip(smiles, mols)]
        return mols_conf

    def get_feature_names_out(self, input_features=None):
        return np.array(["mol"], dtype=object)

class PassthroughNames(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self
    def transform(self, X):
        return X
    def get_feature_names_out(self, input_features=None):
        # 入力が匿名（NumPy配列）でも列数だけで名前を作る
        n = getattr(self, "n_features_in_", None) or (input_features.shape[1] if hasattr(input_features, "shape") else 0)
        if input_features is None or isinstance(input_features, np.ndarray):
            return np.array([f"feat_{i}" for i in range(n)], dtype=object)
        return np.asarray(input_features, dtype=object)

def make_smiles_preprocess(
    fingerprints: List[str] = None,
) -> Pipeline:
    """
    数値データの前処理パイプラインを作成する関数。

    Args:
        impute_type (Optional[str]): 欠損値の補完方法を指定する文字列。
            "Multiple", "mean", "median", "most_frequent", "knn" から選択可能。
        scale_type (Optional[str]): スケーリング方法を指定する文字列。
            "StandardScaler", "MinMaxScaler", "centering" から選択可能。

    Returns:
        Pipeline: 指定された補完方法およびスケーリング方法を含む前処理パイプライン。
    """
    if fingerprints is not None:
        fingerprints_list = [FP_dcit[f] for f in fingerprints]

        generate_conformers = any([f in ["Autocorr", "E3FP", "MORSE","RDF"] for f in fingerprints])
        smiles_pipe = Pipeline(
            [
                ("to_mol", SmilesToMol(generate_conformers=generate_conformers)),
                ("fp", make_union(*fingerprints_list)),
                ("sc", StandardScaler(with_mean=False)),
                ("names", PassthroughNames())   
                 ]
        )
        return smiles_pipe
    else:
        return None

def xenonpy_prest():
    # elements_completed.pd.xz をロード
    elements_completed = pd.read_pickle("elements_completed.pd.xz")
    
    # 1) 配布したファイルの場所（ここを自由に指定）
    src = Path(r"elements_completed.pd.xz")
    
    # 2) XenonPy のデータ置き場へコピー
    
    dst_dir = Path(__cfg_root__) / "dataset"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / "elements_completed.pd.xz")
    
    # 3) インデックスを更新（これが肝）
    preset._make_index(prefix=['dataset'])  # protectedですが実用上OK

class FormulaToFractionDict(BaseEstimator, TransformerMixin):
    """
    'LiFePO4' などの組成式 -> 比率dict（{'Li':0.xx, ...}）の1列に変換
    ColumnTransformerから単一列DataFrame/ndarrayが来てもOK
    """
    def __init__(self, invalid="empty"):  # 'empty'|'error'
        self.invalid = invalid
    def fit(self, X, y=None): return self
    def _to_series(self, X):
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0]
        else:
            s = pd.Series(np.ravel(X))
        return s
    def _formula_to_dict(self, formula):
        if formula is None or (isinstance(formula, float) and np.isnan(formula)):
            return {}
        try:
            comp = PMGComposition(str(formula))
            d = comp.get_el_amt_dict()
            tot = sum(d.values())
            return {str(k): v/tot for k, v in d.items()}
        except Exception:
            if self.invalid == "error":
                raise
            return {}
    def transform(self, X):
        s = self._to_series(X)
        return s.apply(self._formula_to_dict).to_frame(name="comp_dict")
    def get_feature_names_out(self, input_features=None):
        return np.array(["comp_dict"], dtype=object)

# class XenoCompositionsTransformer(BaseEstimator, TransformerMixin):
#     """
#     XenonPy Compositions を sklearn で使う薄いラッパ
#     入力: 1列DataFrame（列名は何でも可）で中身は dict
#     出力: 数値特徴の DataFrame（列名付き）
#     """
#     def __init__(self, prefix="comp__", sample_in_fit=None):
#         self.prefix = prefix
#         self.sample_in_fit = sample_in_fit  # fit時の列名推定用サンプル数（大規模なら間引き）
#     def fit(self, X, y=None):
#         self._comp = Compositions()
#         s = X.iloc[:, 0]
#         if self.sample_in_fit:
#             s = s.iloc[:self.sample_in_fit]
#         desc = self._comp.transform(s)  # DataFrame
#         self._cols_in_ = list(desc.columns)
#         self._cols_out_ = [f"{self.prefix}{c}" for c in self._cols_in_]
#         return self
#     def transform(self, X):
#         s = X.iloc[:, 0]
#         desc = self._comp.transform(s).copy()
#         desc.columns = self._cols_out_
#         return desc
#     def get_feature_names_out(self, input_features=None):
#         return np.array(self._cols_out_, dtype=object)
def _comp_key(comp: Any, *, ndigits: int = 12) -> Hashable:
    """
    dict / pymatgen.Composition / None を安定キーに変換。
    - dict: (("Al", 1.0), ("O", 1.5), ...) のような sorted tuple
    - None/空: 特殊キー
    """
    if comp is None:
        return ("__NONE__",)

    # pymatgen.Composition を想定
    if hasattr(comp, "as_dict"):
        try:
            comp = comp.as_dict()
        except Exception:
            return ("__BAD__", str(comp))

    if isinstance(comp, dict):
        if len(comp) == 0:
            return ("__EMPTY__",)
        items = []
        for k, v in comp.items():
            try:
                items.append((str(k), round(float(v), ndigits)))
            except Exception:
                items.append((str(k), v))
        return tuple(sorted(items))

    # 想定外
    return ("__BAD__", str(comp))


class XenoCompositionsTransformer(BaseEstimator, TransformerMixin):
    """
    XenonPy Compositions を sklearn Transformer として使いつつ、
    - 同一組成の重複計算を避ける（dedupe + cache）
    - joblib の overhead 回避用に n_jobs を調整可能
    """

    def __init__(
        self,
        prefix: str = "comp__",
        sample_in_fit: int = 1,
        *,
        n_jobs: int = 1,                  # ★まずは 1 推奨（必要なら -1）
        featurizers: Union[str, List[str]] = "classic",
        on_errors: str = "nan",
        cache: bool = True,
        key_round_digits: int = 12,
    ):
        self.prefix = prefix
        self.sample_in_fit = sample_in_fit
        self.n_jobs = n_jobs
        self.featurizers = featurizers
        self.on_errors = on_errors
        self.cache = cache
        self.key_round_digits = key_round_digits

    def fit(self, X, y=None):
        self._comp = Compositions(
            n_jobs=self.n_jobs,
            featurizers=self.featurizers,
            on_errors=self.on_errors,
        )

        # 列名確定：少数サンプルで transform（重い sample は不要）
        s = X.iloc[:, 0]
        if self.sample_in_fit and len(s) > self.sample_in_fit:
            s = s.iloc[: self.sample_in_fit]

        # “空”だけだと例外経由で nan 列になる可能性があるので、ダミーを混ぜる
        # （Si は preset にほぼ確実に存在）
        probe = list(s.values)
        probe.append({"Si": 1.0})

        tmp = self._comp.transform(probe)  # DataFrame
        self._cols_in_ = list(tmp.columns)
        self._cols_out_ = [f"{self.prefix}{c}" for c in self._cols_in_]

        self._cache: Dict[Hashable, np.ndarray] = {}
        return self

    def transform(self, X):
        s = X.iloc[:, 0]
        keys = [_comp_key(v, ndigits=self.key_round_digits) for v in s.values]

        # ユニーク組成を抽出（順序保持）
        uniq: Dict[Hashable, Any] = {}
        for k, v in zip(keys, s.values):
            if k not in uniq:
                uniq[k] = v

        # 未計算分だけ計算
        if self.cache:
            need = [(k, v) for k, v in uniq.items() if k not in self._cache]
        else:
            need = list(uniq.items())
            self._cache = {}

        if need:
            comps_to_compute = [v for _, v in need]
            df_new = self._comp.transform(comps_to_compute)  # DataFrame（内部で各行 featurize）:contentReference[oaicite:3]{index=3}

            # 念のため列順固定
            for c in self._cols_in_:
                if c not in df_new.columns:
                    df_new[c] = np.nan
            df_new = df_new[self._cols_in_]

            arr_new = df_new.to_numpy(dtype=float, copy=False)
            for (k, _), row in zip(need, arr_new):
                self._cache[k] = row

        # 元の順番で復元
        arr = np.vstack([self._cache[k] for k in keys])
        out = pd.DataFrame(arr, index=s.index, columns=self._cols_out_)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self._cols_out_, dtype=object)


# ========== 1) 変換器: formula(str) -> pymatgen.Composition ==========
class FormulaToComposition(BaseEstimator, TransformerMixin):
    """ 'LiFePO4' などの組成式 -> pymatgen.Composition（1列） """
    def __init__(self, invalid="empty"):  # 'empty' or 'error'
        self.invalid = invalid
    def fit(self, X, y=None): return self
    def _to_series(self, X):
        if isinstance(X, pd.DataFrame): return X.iloc[:, 0]
        return pd.Series(np.ravel(X))
    def _parse(self, s):
        if s is None or (isinstance(s, float) and np.isnan(s)): return None
        try:
            return PMGComposition(str(s))
        except Exception:
            if self.invalid == "error": raise
            return None
    def transform(self, X):
        s = self._to_series(X).apply(self._parse)
        return pd.DataFrame({"composition": s})
    def get_feature_names_out(self, input_features=None):
        return np.array(["composition"], dtype=object)

# ========== 2) 変換器: matminer で特徴量生成 ==========
# class MatminerCompositionFeaturizer(BaseEstimator, TransformerMixin):
#     """
#     matminer の featurizer 群を sklearn で使えるように。
#     入力: 1列 DataFrame（column名は何でも可）で中身が pymatgen.Composition / None
#     """
#     def __init__(self, featurizers=None, input_col="composition", prefix="mm__", sample_in_fit=256):
#         self.featurizers = featurizers or []
#         self.input_col = input_col
#         self.prefix = prefix
#         self.sample_in_fit = sample_in_fit

#     def _featurize(self, s: pd.Series) -> pd.DataFrame:
#         df_tmp = pd.DataFrame({self.input_col: s})
#         for fz in self.featurizers:
#             # ignore_errors=True で欠損や未知元素でも止まらない
#             df_tmp = fz.featurize_dataframe(df_tmp, col_id=self.input_col,
#                                             ignore_errors=True, inplace=False)
#         # 入力列は学習には不要なので落とす
#         if self.input_col in df_tmp.columns:
#             df_tmp = df_tmp.drop(columns=[self.input_col])
#         return df_tmp

#     def fit(self, X, y=None):
#         s = X.iloc[:, 0]
#         if self.sample_in_fit and len(s) > self.sample_in_fit:
#             s = s.iloc[:self.sample_in_fit]
#         out = self._featurize(s)
#         # 列名を固定 & 接頭辞
#         self.cols_in_ = list(out.columns)
#         self.cols_out_ = [f"{self.prefix}{c}" for c in self.cols_in_]
#         return self

#     def transform(self, X):
#         s = X.iloc[:, 0]
#         out = self._featurize(s)
#         # fit時の列順に合わせ、不足列はNaNで補う（安全）
#         for c in self.cols_in_:
#             if c not in out.columns:
#                 out[c] = np.nan
#         out = out[self.cols_in_].copy()
#         out.columns = self.cols_out_
#         return out

#     def get_feature_names_out(self, input_features=None):
#         return np.array(self.cols_out_, dtype=object)
class MatminerCompositionFeaturizer(BaseEstimator, TransformerMixin):
    """
    入力: 1列 DataFrame（中身が pymatgen.Composition / None）
    出力: 特徴量 DataFrame
    """
    def __init__(
        self,
        featurizers=[],
        input_col="composition",
        prefix="mm__",
        n_jobs=1,
        use_cache=True,
        cache_max=20000,
    ):
        self.featurizers = featurizers
        self.input_col = input_col
        self.prefix = prefix
        self.n_jobs = n_jobs
        self.use_cache = use_cache
        self.cache_max = cache_max

    def _key(self, comp):
        # まずは安定＆軽いキー。必要なら fractional_composition の dict などに変更可
        try:
            return comp.reduced_formula
        except Exception:
            return None

    def fit(self, X, y=None):
        self.mm_ = MultipleFeaturizer(self.featurizers)

        # matminer 側が対応していれば並列化
        try:
            self.mm_.set_n_jobs(self.n_jobs)
        except Exception:
            pass

        self.cols_in_ = list(self.mm_.feature_labels())
        self.cols_out_ = [f"{self.prefix}{c}" for c in self.cols_in_]

        self._cache_ = {} if self.use_cache else None
        return self

    def transform(self, X):
        comps = X.iloc[:, 0].tolist()
        n = len(comps)
        d = len(self.cols_in_)
        nan_row = [np.nan] * d

        # キャッシュなし：素直に一括 featurize
        if not self.use_cache:
            rows = self.mm_.featurize_many(comps, ignore_errors=True)
            rows = [nan_row if r is None else r for r in rows]
            out = pd.DataFrame(rows, columns=self.cols_in_)
            out.columns = self.cols_out_
            return out

        # キャッシュあり：重複組成は再計算しない
        rows = [None] * n
        uncached = []  # (i, key, comp)

        for i, c in enumerate(comps):
            if c is None:
                rows[i] = nan_row
                continue
            k = self._key(c)
            if k is not None and k in self._cache_:
                rows[i] = self._cache_[k]
            else:
                uncached.append((i, k, c))

        if uncached:
            # 同じ key は1回だけ計算
            uniq_keys = []
            uniq_comps = []
            for _, k, c in uncached:
                if k is None:
                    continue
                if k not in self._cache_:
                    uniq_keys.append(k)
                    uniq_comps.append(c)

            if uniq_comps:
                vals = self.mm_.featurize_many(uniq_comps, ignore_errors=True, pbar=False)
                for k, v in zip(uniq_keys, vals):
                    self._cache_[k] = nan_row if v is None else v

                # 雑な上限管理（必要なら LRU に置換）
                if self.cache_max and len(self._cache_) > self.cache_max:
                    self._cache_.clear()

            # 反映
            for i, k, c in uncached:
                if c is None:
                    rows[i] = nan_row
                elif k is None:
                    v = self.mm_.featurize(c, ignore_errors=True)
                    rows[i] = nan_row if v is None else v
                else:
                    rows[i] = self._cache_.get(k, nan_row)

        out = pd.DataFrame(rows, columns=self.cols_in_)
        out.columns = self.cols_out_
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.cols_out_, dtype=object)

class MendeleevCompositionFeaturizer(BaseEstimator, TransformerMixin):
    """
    入力: 1列 DataFrame（中身が pymatgen.Composition / None）
    出力: mendeleev元素プロパティを組成分率で集約した特徴量
    """
    def __init__(
        self,
        props=None,
        stats=("mean", "std", "min", "max", "range"),
        prefix="md__",
        use_cache=True,
        cache_max=20000,
    ):
        self.props = props or [
            "atomic_number",
            "atomic_weight",
            "atomic_radius",
            "covalent_radius_cordero",
            "electron_affinity",
            "boiling_point",
            "density",
        ]
        self.stats = tuple(stats)
        self.prefix = prefix
        self.use_cache = use_cache
        self.cache_max = cache_max

    @staticmethod
    def _weighted_stats(v, w):
        """v,w: 1D np.array。v の NaN は除外。"""
        mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if not np.any(mask):
            return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan, range=np.nan)

        v = v[mask]
        w = w[mask]
        w = w / w.sum()

        mean = np.sum(w * v)
        var = np.sum(w * (v - mean) ** 2)
        std = np.sqrt(var)
        vmin = np.min(v)
        vmax = np.max(v)
        return dict(mean=mean, std=std, min=vmin, max=vmax, range=vmax - vmin)

    def fit(self, X, y=None):
        ptable = fetch_table("elements").set_index("symbol", drop=False)
    
        self._prop_map_ = {}
        block_map = {"s": 0.0, "p": 1.0, "d": 2.0, "f": 3.0}
    
        for p in self.props:
            if p not in ptable.columns:
                raise ValueError(f"Unknown mendeleev property: {p}")
    
            if p == "block":
                # 文字(s/p/d/f) -> 数値
                series = ptable[p].map(block_map).astype(float)
                self._prop_map_[p] = series.to_dict()
            else:
                self._prop_map_[p] = ptable[p].to_dict()
    
        self.cols_out_ = [f"{self.prefix}{p}__{st}" for p in self.props for st in self.stats]
        self._cache_ = {} if self.use_cache else None
        return self

    def transform(self, X):
        comps = X.iloc[:, 0].tolist()
        rows = []

        for comp in comps:
            if comp is None:
                rows.append([np.nan] * len(self.cols_out_))
                continue

            # キャッシュキー（組成が同じなら再計算しない）
            key = None
            if self.use_cache:
                try:
                    key = comp.reduced_formula
                except Exception:
                    key = None
                if key is not None and key in self._cache_:
                    rows.append(self._cache_[key])
                    continue

            # 分率（fractional_composition が安全）
            try:
                frac = comp.fractional_composition.get_el_amt_dict()  # {"Fe":0.5,"O":0.5} 等
            except Exception:
                rows.append([np.nan] * len(self.cols_out_))
                continue

            syms = list(frac.keys())
            w = np.array([frac[s] for s in syms], dtype=float)

            feats = []
            for p in self.props:
                mp = self._prop_map_[p]
                v = np.array([mp.get(s, np.nan) for s in syms], dtype=float)
                st = self._weighted_stats(v, w)
                for name in self.stats:
                    feats.append(st.get(name, np.nan))

            rows.append(feats)

            if self.use_cache and key is not None:
                self._cache_[key] = feats
                if self.cache_max and len(self._cache_) > self.cache_max:
                    self._cache_.clear()

        return pd.DataFrame(rows, columns=self.cols_out_)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.cols_out_, dtype=object)

def make_comp_preprocess(
    method="xenonpy", # "matminer"
    feats=None
) -> Pipeline:
    """
    数値データの前処理パイプラインを作成する関数。

    Args:
        impute_type (Optional[str]): 欠損値の補完方法を指定する文字列。
            "Multiple", "mean", "median", "most_frequent", "knn" から選択可能。
        scale_type (Optional[str]): スケーリング方法を指定する文字列。
            "StandardScaler", "MinMaxScaler", "centering" から選択可能。

    Returns:
        Pipeline: 指定された補完方法およびスケーリング方法を含む前処理パイプライン。
    """
    xenonpy_prest()
    if method=="xenonpy":
        return Pipeline([
            ("formula2dict", FormulaToFractionDict(invalid="empty")),
            ("xeno",         XenoCompositionsTransformer(prefix="comp__", sample_in_fit=512)),
            ("imp",          SimpleImputer(strategy="median")),
            ("sc",           StandardScaler(with_mean=False)),
        ])
    elif method=="matminer":
        feats_list = [INORG_dict[f] for f in feats]
        return Pipeline([
            ("f2c",  FormulaToComposition(invalid="empty")),
            ("mm",   MatminerCompositionFeaturizer(
                featurizers=feats_list,
                prefix="comp__",
                # sample_in_fit=512
            )),
            ("imp",  SimpleImputer(strategy="median")),
            ("sc",   StandardScaler(with_mean=False)),
        ])
    elif method=="mendeleev":
        return Pipeline([
            ("f2c",  FormulaToComposition(invalid="empty")),
            ("md",   MendeleevCompositionFeaturizer(props=feats, prefix="comp__")),
            ("imp",          SimpleImputer(strategy="median")),
            ("sc",           StandardScaler(with_mean=False)),
        ])

    else:
        return None

def make_preprocess_pipeline(
    num_process: Pipeline,
    cat_process: Pipeline,
    smiles_process: Optional[Pipeline] = None, 
    comp_process: Optional[Pipeline] = None, 
    numcat_common_preprocess: Optional[Pipeline] = None,
    common_process: Optional[Pipeline] = None,
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    smiles_cols: Optional[List[str]] = None,
    comp_cols: Optional[List[str]] = None
) -> Pipeline:
    """
    数値データとカテゴリカルデータの前処理パイプラインを作成する関数。

    Args:
        num_process (Pipeline): 数値データの前処理パイプライン。
        cat_process (Pipeline): カテゴリカルデータの前処理パイプライン。
        common_process (Optional[Pipeline]): 共通の前処理パイプライン。デフォルトは None。
        num_cols (Optional[List[str]]): 数値データのカラム名のリスト。デフォルトは None。
        cat_cols (Optional[List[str]]): カテゴリカルデータのカラム名のリスト。デフォルトは None。

    Returns:
        Pipeline: 数値データ、カテゴリカルデータ、および共通の前処理を含むパイプライン。
    """
    # カラムごとの前処理を定義するリストを初期化
    transforms = []
    numcat_transforms = []

    # 数値データのカラムが指定されている場合、数値データの前処理を追加
    if num_cols:
        numcat_transforms.append(("num", num_process, num_cols))

    # カテゴリカルデータのカラムが指定されている場合、カテゴリカルデータの前処理を追加
    if cat_cols:
        numcat_transforms.append(("cat", cat_process, cat_cols))

    if num_cols+cat_cols:
        numcat_preprocess = [("num_cat_prerprocess", ColumnTransformer(transformers=numcat_transforms))]
        if numcat_common_preprocess:
            numcat_preprocess.append(("num_cat_common", numcat_common_preprocess))
        transforms.append(("num_cat", Pipeline(numcat_preprocess), num_cols+cat_cols))
    
    # SMILESデータのカラムが指定されている場合、SMILESデータの前処理を追加
    if smiles_cols:
        transforms.append(("smiles", smiles_process, smiles_cols))

    if comp_cols:
        transforms.append(("comp", comp_process, comp_cols))

    preprocess = []
    # 全体の前処理パイプラインを初期化
    preprocess.append(("column_preprocess", ColumnTransformer(transformers=transforms)))

    # 共通の前処理が指定されている場合、共通の前処理を追加
    if common_process:
        preprocess.append(("common_preprocess", common_process))

    # パイプラインを作成して返す
    return imbPipeline(steps=preprocess)

def make_preprocess(
    model_name: str,
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    num_impute_type: Optional[str] = None,
    num_scale_type: Optional[str] = None,
    cat_impute: bool = False,
    smiles_cols= None,
    fingerprints=None,
    comp_cols= None,
    comp_method=None,
    comp_feats=None,
    poly: bool = False,
    poly_degree: int = 1,
    poly_interaction_only: bool = True,
    decomposition: bool = False,
    decomposition_method: str = "PCA",
    n_components: int = 2,
    ensemble: bool = False
) -> Pipeline:
    """
    前処理パイプラインを作成する関数。

    Args:
        model_name (str): 使用するモデルの名前。'LightGBM' または 'CatBoost' など。
        num_cols (Optional[List[str]]): 数値データのカラム名のリスト。デフォルトは None。
        cat_cols (Optional[List[str]]): カテゴリカルデータのカラム名のリスト。デフォルトは None。
        num_impute_type (Optional[str]): 数値データの欠損値の補完方法。デフォルトは None。
        num_scale_type (Optional[str]): 数値データのスケーリング方法。デフォルトは None。
        cat_impute (bool): カテゴリカルデータの欠損値の補完を行うかどうか。デフォルトは False。
        poly (bool): 多項式特徴量を作成するかどうか。デフォルトは False。
        poly_degree (int): 多項式特徴量の次数。デフォルトは 1。
        poly_interaction_only (bool): 相互作用項のみを含めるかどうか。デフォルトは True。
        pca (bool): 主成分分析（PCA）を行うかどうか。デフォルトは False。
        pca_n_components (int): PCAで使用する主成分の数。デフォルトは 2。
        ensemble (bool): アンサンブル学習を行うかどうか。デフォルトは False。

    Returns:
        Pipeline: 前処理パイプライン。
    """    
    # 数値データの前処理パイプラインを作成
    num_process = make_numeric_preprocess(
        impute_type=num_impute_type,
        scale_type=num_scale_type
    )

    # カテゴリカルデータの前処理パイプラインを作成
    cat_process = make_categorical_preprocess(
        model_name=model_name,
        impute=cat_impute,
        poly=poly,
        pca=decomposition,
        ensemble=ensemble
    )

    # カテゴリカルデータの前処理パイプラインを作成
    smiles_process = make_smiles_preprocess(
        fingerprints=fingerprints,
    )

    comp_process = make_comp_preprocess(
        method=comp_method,
        feats=comp_feats
    )
    

    numcat_common_preprocess = make_numcat_common_preprocess(
        poly=poly,
        degree=poly_degree,
        interaction_only=poly_interaction_only,
    )
    
    # 共通の前処理パイプラインを作成
    common_process = make_common_preprocess(
        decomposition=decomposition,
        decomposition_method=decomposition_method,
        n_components=n_components
    )

    # 全体の前処理パイプラインを作成
    preprocess_pipeline = make_preprocess_pipeline(
        num_process=num_process,
        cat_process=cat_process,
        smiles_process=smiles_process,
        comp_process=comp_process,
        numcat_common_preprocess=numcat_common_preprocess,
        common_process=common_process,
        num_cols=num_cols or [],
        cat_cols=cat_cols or [],
        smiles_cols=smiles_cols or [],
        comp_cols=comp_cols or []
    )

    return preprocess_pipeline