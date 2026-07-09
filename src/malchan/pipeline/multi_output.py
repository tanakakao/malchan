import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, LeaveOneOut

from typing import List, Optional, Union, Dict, Tuple, Callable, Any
import warnings

warnings.simplefilter('ignore')

from .single_output import PipelineSharedContext, SingleOutputMLModelPipeline

class MLModelPipeline:
    def __init__(
        self,
    ):
        """
        機械学習モデルパイプラインの初期化。

        Args:
            X (pd.DataFrame): 特徴量データ。
            target_cols (List[str]): 目的変数のカラム名のリスト。
            num_cols (List[str]): 数値特徴量のカラム名のリスト。
            cat_cols (List[str]): カテゴリカル特徴量のカラム名のリスト。
            model_names (List[str]): 使用するモデルの名前のリスト。
            tuning (bool): モデルチューニングを実施するかどうかのフラグ。デフォルトは False。
            ens_type (Optional[str]): アンサンブルの種類。'アンサンブル', 'スタッキング', 'バギング', 'ブースティング' から選択。デフォルトは None。
            base_model (Optional[str]): ベースモデルの名前。スタッキング、バギング、ブースティングのときに使用。デフォルトは None。
            model_params (Optional[List[Dict[str, Any]]]): 各モデルのパラメータを指定するリスト。デフォルトは None。
            base_model_params (Optional[Dict[str, Any]]): ベースモデルのパラメータを指定する辞書。デフォルトは None。
            num_impute_type (Optional[str]): 数値データの補完方法。デフォルトは None。
            num_scale_type (Optional[str]): 数値データのスケーリング方法。デフォルトは None。
            cat_impute (bool): カテゴリカルデータの補完を実施するかどうかのフラグ。デフォルトは False。
            poly (bool): 多項式特徴量を追加するかどうかのフラグ。デフォルトは False。
            poly_degree (int): 多項式の次数。デフォルトは 1。
            poly_interaction_only (bool): 交互作用のみの多項式特徴量を追加するかどうかのフラグ。デフォルトは True。
            pca (bool): 主成分分析 (PCA) を実施するかどうかのフラグ。デフォルトは False。
            pca_n_components (int): PCA のコンポーネント数。デフォルトは 2。
        """
        self.X = None
        self.y = None
        self.context = None

        self.num_cols = None
        self.cat_cols = None
        self.all_cols = None
        self.target_cols = None
        self.model_names = None
        self.smiles_cols = None
        self.comp_cols = None
        self.models = {}
        self.feature_names = []

        self.unique_cols = None

        self.tasks = []
        self.tunings = []
        self.ensembles = []
        self.ens_types = []
        self.base_models = []
        self.model_params = []
        self.base_model_params = []

        self.ad_model_name = None
        self.ad_model_param = None
        self.ad = False

        self.num_impute_type = None
        self.num_scale_type = None
        self.cat_impute = False
        self.poly = False
        self.poly_degree = 2
        self.poly_interaction_only = False
        self.pca = False
        self.pca_n_components = 2
        self.decomposition = False
        self.decomposition_method = "PCA"
        self.dec_n_components = 2
        self.sampling_method = None

        self.target_items = {}

        self.fingerprints = None
        self.comp_method = None
        self.comp_feats = None

    def fit(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        tasks: List[str],
        num_cols: List[str],
        cat_cols: List[str],
        model_names: List[str],
        smiles_cols: List[str] = None,
        fingerprints: List[str] = None,
        comp_cols: List[str] = None,
        comp_method: str = None,
        comp_feats: List[str] = None,
        ad_model_name: str = None,
        ad_model_param: dict = None,
        impute: bool =False,
        tunings: bool = False,
        ensembles: bool = False,
        ens_types: Optional[str] = None,
        base_models: Optional[str] = None,
        model_params: Optional[List[Dict[str, Any]]] = None,
        base_model_params: Optional[Dict[str, Any]] = None,
        num_impute_type: Optional[str] = None,
        num_scale_type: Optional[str] = None,
        cat_impute: bool = False,
        poly: bool = False,
        poly_degree: int = 1,
        poly_interaction_only: bool = True,
        decomposition: bool= False,
        decomposition_method: str = "PCA",
        dec_n_components: int = 2,
        sampling_method=None
    ) -> None:
        """
        モデルパイプラインをデータに適合させる関数。チューニングが必要な場合はチューニングを実施。
        """
        from ..models.utils import get_cat_unique_values

        num_cols = [] if num_cols is None else num_cols
        cat_cols = [] if cat_cols is None else cat_cols
        smiles_cols = [] if (smiles_cols is None or smiles_cols==[None]) else smiles_cols
        comp_cols = [] if (comp_cols is None or comp_cols == [None]) else comp_cols

        # 入力データ、目的変数、数値・カテゴリカル特徴量などの設定
        feature_cols = num_cols + cat_cols + smiles_cols + comp_cols
        target_cols = [] if target_cols is None else target_cols
        if impute:
            if any(t == "regression" for t in ([tasks] if isinstance(tasks, str) else tasks)):
                X = df[feature_cols]
                Y = df[target_cols]
            else:
                imputer = SimpleImputer(strategy="most_frequent")
                X = df[feature_cols]
                Y = pd.DataFrame(imputer.fit_transform(df[target_cols]), columns=target_cols, index=df.index)
        else:
            _df = df[feature_cols + target_cols].dropna()
            X = _df[feature_cols]
            Y = _df[target_cols]
        # self.y = df[target_cols]
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.all_cols = num_cols + cat_cols + smiles_cols + comp_cols
        self.target_cols = target_cols
        self.model_names = model_names
        self.smiles_cols = smiles_cols
        self.comp_cols = comp_cols
        self.models = {}
        self.feature_names = {}

        # カテゴリカル変数のユニーク値を取得
        self.unique_cols = get_cat_unique_values(X, self.cat_cols+self.smiles_cols+self.comp_cols)
        self.context = PipelineSharedContext(
            X=X,
            Y=Y,
            num_cols=self.num_cols,
            cat_cols=self.cat_cols,
            smiles_cols=self.smiles_cols,
            comp_cols=self.comp_cols,
            all_cols=self.all_cols,
            unique_cols=self.unique_cols,
        )
        self.X = None
        self.y = None

        self.tasks = [tasks]*len(self.target_cols) if type(tasks)==str else tasks
        self.tunings = [tunings]*len(self.target_cols) if type(tunings)==bool else tunings
        self.ensembles = [ensembles]*len(self.target_cols) if type(ensembles)==bool else ensembles
        self.ens_types = [ens_types]*len(self.target_cols) if (type(ens_types)==str)|(ens_types is None) else ens_types
        self.base_models = [base_models]*len(self.target_cols) if (type(base_models)==str)|(base_models is None) else base_models
        self.model_params = [None]*len(self.target_cols) if model_params is None else model_params
        self.base_model_params = [None]*len(self.target_cols) if base_model_params is None else base_model_params

        self.ad_model_name = ad_model_name
        self.ad_model_param = ad_model_param
        self.ad = ad_model_name is not None
        if self.ad:
            self.tasks += ["AD"]

        self.num_impute_type = num_impute_type
        self.num_scale_type = num_scale_type
        self.cat_impute = cat_impute
        self.poly = poly
        self.poly_degree = poly_degree
        self.poly_interaction_only = poly_interaction_only
        self.decomposition = decomposition
        self.decomposition_method = decomposition_method
        self.dec_n_components = dec_n_components
        self.sampling_method = sampling_method

        self.target_items = {}

        self.fingerprints = fingerprints
        self.comp_method = comp_method
        self.comp_feats = comp_feats

        target_cols = self.target_cols if not self.ad else self.target_cols+["AD"]
        for i, target in enumerate(target_cols):
            self.models[target] = SingleOutputMLModelPipeline()
            if target != "AD":
                self.models[target].fit_from_context(
                    context=self.context,
                    target_col=self.target_cols[i],
                    task=self.tasks[i],
                    model_names=self.model_names[i],
                    fingerprints=self.fingerprints,
                    comp_method=self.comp_method,
                    comp_feats=self.comp_feats,
                    ad=False,
                    tuning=self.tunings[i],
                    ensemble=self.ensembles[i],
                    ens_type=self.ens_types[i],
                    base_model=self.base_models[i],
                    model_params=self.model_params[i],
                    base_model_param = self.base_model_params[i],
                    num_impute_type=self.num_impute_type,
                    num_scale_type=self.num_scale_type,
                    cat_impute=self.cat_impute,
                    poly=self.poly,
                    poly_degree=self.poly_degree,
                    poly_interaction_only=self.poly_interaction_only,
                    decomposition = self.decomposition,
                    decomposition_method = self.decomposition_method,
                    dec_n_components = self.dec_n_components,
                    sampling_method = self.sampling_method
                )
            else:
                self.models[target].fit_from_context(
                    context=self.context,
                    target_col=None,
                    task="AD",
                    model_names=[self.ad_model_name],
                    fingerprints=self.fingerprints,
                    comp_method=self.comp_method,
                    comp_feats=self.comp_feats,
                    ad=True,
                    tuning=False,
                    ensemble=False,
                    ens_type=None,
                    base_model=None,
                    model_params=[self.ad_model_param],
                    base_model_param = None,
                    num_impute_type=self.num_impute_type,
                    num_scale_type=self.num_scale_type,
                    cat_impute=self.cat_impute,
                    poly=self.poly,
                    poly_degree=self.poly_degree,
                    poly_interaction_only=self.poly_interaction_only,
                    decomposition = self.decomposition,
                    decomposition_method = self.decomposition_method,
                    dec_n_components = self.dec_n_components,
                )
            self.feature_names[target] = self.models[target].feature_names
            if self.tasks[i]=="classification":
                self.target_items[target] = self.models[target].target_items

    def predict(
        self,
        X: Optional[pd.DataFrame] = None,
        models: Optional[Dict[str, Pipeline]] = None,
        proba: bool = False,
        idx2item=False
    ) -> pd.DataFrame:
        """
        モデルを用いて予測を行う関数。

        Args:
            X (Optional[pd.DataFrame]): 予測に使用する特徴量データ。デフォルトは None。
            models (Optional[Dict[str, Pipeline]]): 使用するモデルの辞書。デフォルトは None。

        Returns:
            pd.DataFrame: 予測結果。
        """
        # 各ターゲット変数に対して予測を実施
        target_cols = self.target_cols if not self.ad else self.target_cols+["AD"]

        models = self.models if models is None else models
        X_data = self.context.X if X is None and self.context is not None else (models[target_cols[0]]._get_X() if X is None else X)

        predictions = pd.concat(
            [
                models[target].predict(X_data[self.all_cols], proba=(models[target].task=="classification")&proba, idx2item=idx2item) for target in target_cols
            ],
            axis=1
        )
        return predictions

    def score(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        models: Optional[Dict[str, Pipeline]] = None
    ) -> pd.DataFrame:
        """
        モデルのスコアを計算する関数。

        Args:
            X (Optional[pd.DataFrame]): 予測に使用する特徴量データ。デフォルトは None。
            y (Optional[Union[np.ndarray, pd.Series]]): 予測値と比較するターゲットデータ。デフォルトは None。
            models (Optional[Dict[str, Pipeline]]): 使用するモデルの辞書。デフォルトは None。

        Returns:
            pd.DataFrame: RMSE、MAE、R2 スコアを含むデータフレーム。
        """
        # 入力データが指定されていない場合は学習データを使用
        scores = {}

        # 目的変数ごとにスコアを計算
        for target in self.target_cols:
            scores[target] = self.models[target].score()
        return scores

    def cv_score(
        self,
        method: Optional[str] = 'kfold',
        n_splits: Optional[int] = 5,
        X: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        クロスバリデーションを用いてモデルのスコアを計算する関数。

        Args:
            method (Optional[str]): クロスバリデーションの手法。'kfold' または 'loo' を指定。
            n_splits (Optional[int]): KFoldの分割数。デフォルトは5。
            X (Optional[pd.DataFrame]): 予測に使用する特徴量データ。デフォルトは None。
            y (Optional[Union[np.ndarray, pd.Series]]): 予測値と比較するターゲットデータ。デフォルトは None。

        Returns:
            Dict[str, pd.DataFrame]: トレーニングとテストのスコアを含む辞書。
        """
        for target in self.target_cols:
            self.models[target].cv_score()

    def shap(
        self,
        X=None
    ) -> None:
        """
        SHAP値を計算する関数。各ターゲット変数に対してSHAP値とエクスプレイナーを取得。
        """
        target_cols = self.target_cols if not self.ad else self.target_cols+["AD"]
        for target in target_cols:
            self.models[target].shap()

    def get_xai(self):
        target_cols = self.target_cols if not self.ad else self.target_cols+["AD"]
        for target in target_cols:
            self.models[target].get_xai()


    def get_instance_vars(
        self
    ) -> Dict[str, Any]:
        """
        インスタンス変数を取得する関数。

        Returns:
            Dict[str, Any]: インスタンスの変数を含む辞書。
        """
        return vars(self)

    def predict_objective(
        self,
        X: Optional[pd.DataFrame] = None,
        obj_values: Optional[List[Optional[float]]] = None,
        models: Optional[Dict[str, Pipeline]] = None,
    ) -> pd.DataFrame:
        """
        特定の目的変数を考慮して予測を行う関数。

        Args:
            X (Optional[pd.DataFrame]): 予測に使用する特徴量データ。デフォルトは None。
            obj_values (Optional[List[Optional[float]]]): 目標値のリスト。デフォルトは None。
            models (Optional[Dict[str, Pipeline]]): 使用するモデルの辞書。デフォルトは None。

        Returns:
            pd.DataFrame: 調整された予測結果。
        """
        _tcols = self.target_cols
        _ovals = obj_values if obj_values is not None else [None]*len(_tcols)
        if self.ad:
            _tcols = _tcols+["AD"]
            _ovals = _ovals+[None]
        y_pred = pd.concat(
            [
                self.models[target].predict_objective(X, obj_value) for target, obj_value in zip(_tcols, _ovals)
            ],
            axis=1
        )
        return y_pred

    def to_checkpoint(
        self,
        include_data: bool = False,
        include_preprocessed: bool = False,
        include_cv: bool = False,
        include_importances: bool = False,
        include_shap_arrays: bool = False,
    ) -> Dict[str, Any]:
        """
        MLModelPipeline のチェックポイントを辞書形式で生成する。
        各ターゲットごとの SingleOutputMLModelPipeline のチェックポイントと、
        MLModelPipeline 自身の状態をまとめる。
        """
        state: Dict[str, Any] = {}

        # MLModelPipeline 自身のインスタンス変数を保存
        # (SingleOutputMLModelPipeline には含まれない共通設定)
        keys_self = [
            "num_cols", "cat_cols", "all_cols", "target_cols",
            "model_names", "smiles_cols", "comp_cols", "unique_cols",
            "tasks", "tunings", "ensembles", "ens_types", "base_models",
            "model_params", "base_model_params",
            "ad_model_name", "ad_model_param", "ad",
            "num_impute_type", "num_scale_type", "cat_impute",
            "poly", "poly_degree", "poly_interaction_only",
            "pca", "pca_n_components", "sampling_method",
            "target_items", "fingerprints", "comp_method", "comp_feats",
            "feature_names", # 各ターゲットごとのfeature_names（MLModelPipelineが保持する辞書）
        ]
        for k in keys_self:
            if hasattr(self, k):
                state[k] = getattr(self, k)
        if hasattr(self, "context") and self.context is not None:
            state["context"] = PipelineSharedContext(
                X=self.context.X if include_data else None,
                Y=self.context.Y if include_data else None,
                num_cols=self.context.num_cols,
                cat_cols=self.context.cat_cols,
                smiles_cols=self.context.smiles_cols,
                comp_cols=self.context.comp_cols,
                all_cols=self.context.all_cols,
                unique_cols=self.context.unique_cols,
            )

        # 各 SingleOutputMLModelPipeline のチェックポイントを保存
        models_checkpoints = {}
        if hasattr(self, "models") and self.models:
            for target_name, single_model_pipeline in self.models.items():
                models_checkpoints[target_name] = single_model_pipeline.to_checkpoint(
                    include_data=include_data,
                    include_preprocessed=include_preprocessed,
                    include_cv=include_cv,
                    include_importances=include_importances,
                    include_shap_arrays=include_shap_arrays,
                )
        state["models"] = models_checkpoints

        return state

    @classmethod
    def load_checkpoint(cls, state: Dict[str, Any]) -> "MLModelPipeline":
        """
        チェックポイントから MLModelPipeline インスタンスを復元する。
        """
        state = state.copy()
        obj = cls.__new__(cls)  # __init__ を通さずにインスタンスを作成

        # 'models' キーは後で特別に処理するため、先にpopで削除しておく
        models_checkpoints = state.pop("models", {})

        # 残りのインスタンス変数を復元
        obj.__dict__.update(state)
        context = getattr(obj, "context", None)

        # 各 SingleOutputMLModelPipeline インスタンスを復元し、obj.models に格納
        restored_models = {} # 新しい辞書を準備
        for target_name, ckpt_data in models_checkpoints.items():
            restored = SingleOutputMLModelPipeline.load_checkpoint(ckpt_data)
            if getattr(restored, "shared_mode", False):
                restored.context = context
            restored_models[target_name] = restored
        obj.models = restored_models # 復元されたモデルをobj.modelsに代入

        return obj
