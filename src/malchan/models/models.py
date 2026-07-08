import numpy as np
import pandas as pd
import optuna

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, LeaveOneOut

from .training import fit_model, tune_model, cv_fit
from .utils import get_cat_unique_values, feature_names_from_pipeline, label_encode
from .explainability import get_shap_values, get_importances, get_pfi_values, get_shap_scatter, get_pd_and_ice, get_pd_and_ice_2d
from .pipelines import make_pipeline

from typing import List, Optional, Union, Dict, Tuple, Callable, Any
import warnings

optuna.logging.disable_default_handler()
warnings.simplefilter('ignore')

class SingleOutputMLModelPipeline:
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
        self.num_cols = None
        self.cat_cols = None
        self.all_cols = None
        self.target_col = None
        self.smiles_cols = None
        self.comp_cols = None
        self.unique_cols = None

        self.task = None
        self.model_names = None
        self.ensemble = False
        self.ens_type = None
        self.base_model = None
        self.model_params = None
        self.base_model_param = None
        self.tuning = False

        self.num_impute_type = None
        self.num_scale_type = None
        self.cat_impute = False
        self.poly = False
        self.poly_degree = 2
        self.poly_interaction_only = False
        self.decomposition = False
        self.decomposition_method = "PCA"
        self.dec_n_components = 2

        self.sampling_method = None

        self.fingerprints = None
        self.comp_method = None
        self.comp_feats = None

        self.ad = False
        
        self.le = None
        self.target_items = None
        # # アンサンブルや特徴量変換を考慮したカテゴリカル列のインデックス作成
        self.cat_index = None
        self.cat_index_fit = None
        self.item2idx = None
        self.idx2item = None
        self.X_sample = None

        self.feature_names = None
        self.df_prerpocessed = None

        self.cv_scores = None
        self.cv_preds = None

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        task: str,
        num_cols: List[str],
        cat_cols: List[str],
        model_names: List[str],
        smiles_cols: List[str] = None,
        fingerprints: List[str] = None,
        comp_cols: List[str] = None,
        comp_method: str = None,
        comp_feats: List[str] = None,
        ad: bool = False,
        impute=False,
        tuning: bool = False,
        ensemble: bool = False,
        ens_type: Optional[str] = None,
        base_model: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        base_model_param: Optional[Dict[str, Any]] = None,
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
        # 入力データ、目的変数、数値・カテゴリカル特徴量などの設定
        num_cols = [] if num_cols is None else num_cols
        cat_cols = [] if cat_cols is None else cat_cols
        smiles_cols = [] if (smiles_cols is None or smiles_cols==[None]) else smiles_cols
        comp_cols = [] if (comp_cols is None or comp_cols == [None]) else comp_cols

        all_cols = num_cols+cat_cols+smiles_cols+comp_cols
        if target_col is not None:
            all_cols = all_cols + [target_col]
        
        if impute:
            if task=="AD":
                _df = df[all_cols].dropna()
                self.X = _df[num_cols + cat_cols + smiles_cols+ comp_cols]
                self.y = _df[[target_col]] if target_col is not None else None
            elif task=="regression":
                iterative_imputer = IterativeImputer(
                    max_iter=10,
                    random_state=0
                )
                _df = pd.DataFrame(
                    iterative_imputer.fit_transform(df[num_cols + [target_col]]),
                    columns=num_cols + [target_col]
                )
                self.X = df[num_cols + cat_cols + smiles_cols + comp_cols]
                self.y = _df[[target_col]] if target_col is not None else None
            else:
                imputer = SimpleImputer(strategy='most_frequent')
                self.y = pd.DataFrame(
                    imputer.fit_transform(df[[target_col]]),
                    columns=[target_col]
                )
                self.X = df[num_cols + cat_cols + smiles_cols+ comp_cols]
        else:
            _df = df[all_cols].dropna()
            self.X = _df[num_cols + cat_cols + smiles_cols+ comp_cols]
            self.y = _df[[target_col]] if target_col is not None else None

        # self.X = df[num_cols + cat_cols]
        # self.y = df[[target_col]]
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.all_cols = num_cols + cat_cols + smiles_cols + comp_cols
        self.target_col = target_col if not ad else "AD"
        self.smiles_cols = smiles_cols
        self.comp_cols = comp_cols

        # カテゴリカル変数のユニーク値を取得
        self.unique_cols = get_cat_unique_values(self.X, self.cat_cols+self.smiles_cols+self.comp_cols)

        self.task = task if not ad else "AD"
        self.model_names = model_names
        self.ensemble = ensemble if not ad else False
        self.ens_type = ens_type
        self.base_model = base_model
        self.model_params = model_params
        self.base_model_param = base_model_param
        self.tuning = tuning if not ad else False

        self.num_impute_type = num_impute_type
        self.num_scale_type = num_scale_type
        self.cat_impute = cat_impute
        self.poly = poly
        self.poly_degree = poly_degree
        self.poly_interaction_only = poly_interaction_only
        self.decomposition= decomposition
        self.decomposition_method = decomposition_method
        self.dec_n_components = dec_n_components

        self.sampling_method = sampling_method if task=="classification" else None

        self.fingerprints = fingerprints
        self.comp_method = comp_method
        self.comp_feats = comp_feats

        self.ad = ad
        
        # アンサンブルや特徴量変換を考慮したカテゴリカル列のインデックス作成
        self.cat_index = [self.all_cols.index(c) for c in cat_cols + smiles_cols + comp_cols]
        self.cat_index_fit = [] if decomposition or poly or self.ensemble else [self.all_cols.index(c) for c in cat_cols]
    
        self.model = self._make_pipeline()
        # 目的変数ごとにモデルを適合
        if self.task == "classification":
            self.target_items = np.unique(self.y.values)
            self.y, self.le = label_encode(self.y)
            self.idx2item = {k: v for k, v in zip(self.y.unique(), self.target_items)}
            self.item2idx = {k: v for k, v in zip(self.target_items, self.y.unique())}
        else:
            self.sampling_method = None
        if self.tuning:
            # モデルチューニングを実行[target]
            self.model, best_params, best_base_param = tune_model(
                X=self.X,
                y=self.y,
                model_pipeline=self.model,
                model_names=self.model_names,
                base_model=self.base_model,
                ens_type=self.ens_type,
                sampling_method = self.sampling_method,
                cat_index=self.cat_index,
                cat_index_fit=self.cat_index_fit,
                task=self.task,
                n_trials=30,
                verbose=2
            )
            self.model_params = best_params
            self.base_model_param = best_base_param
        else:
            # モデルを適合させる
            self.model = fit_model(
                X=self.X,
                y=self.y,
                model_pipeline=self.model,
                model_names=self.model_names,
                ensemble=self.ensemble,
                sampling_method = self.sampling_method,
                cat_index=self.cat_index,
                cat_index_fit=self.cat_index_fit,
            )
        self.feature_names = feature_names_from_pipeline(self.model)
        # 前処理を実施
        self.df_prerpocessed = pd.DataFrame(
            self.model['preprocess'].transform(self.X),
            columns=self.feature_names
        )
       
    def _make_pipeline(
        self
    ) -> None:
        """
        前処理とモデルパイプラインを作成する内部関数。
        """        
        model_items = make_pipeline(
            model_names=self.model_names,
            task=self.task,
            num_cols=self.num_cols,
            cat_cols=self.cat_cols,
            smiles_cols=self.smiles_cols,
            comp_cols=self.comp_cols,
            num_impute_type=self.num_impute_type,
            num_scale_type=self.num_scale_type,
            cat_impute=self.cat_impute,
            fingerprints=self.fingerprints,
            comp_method=self.comp_method,
            comp_feats=self.comp_feats,
            poly=self.poly,
            poly_degree=self.poly_degree,
            poly_interaction_only=self.poly_interaction_only,
            decomposition=self.decomposition,
            decomposition_method=self.decomposition_method,
            n_components=self.dec_n_components,
            ensemble=self.ensemble,
            ens_type=self.ens_type,
            base_model=self.base_model,
            model_params=self.model_params,
            base_model_params=self.base_model_param
        )
        # 各モデルと前処理の設定を保存
        return model_items[0]
        # self.model = model_items[0]
        
    def predict(
        self,
        X: Optional[pd.DataFrame] = None,
        model: Optional[Dict[str, Pipeline]] = None,
        proba=False,
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
        models_pred = self.model if model is None else model
        X_data = self.X if X is None else X
        # 各ターゲット変数に対して予測を実施
        if proba:
            predictions = models_pred.predict_proba(X_data[self.all_cols])
        elif self.task == "AD":
            predictions = models_pred.decision_function(X_data[self.all_cols])
        else:
            predictions = models_pred.predict(X_data[self.all_cols]).reshape(-1,1)

        if self.task == "regression":
            predictions = pd.DataFrame(
                predictions,
                columns=[self.target_col]
            )    
        else:
            if proba:
                predictions = pd.DataFrame(
                    predictions,
                    columns=[self.target_col+"_"+c for c in self.le.inverse_transform(np.arange(predictions.shape[-1])).astype(str)]
                )
            else:
                predictions = pd.DataFrame(
                    predictions,
                    columns=[self.target_col]
                )
                if (idx2item)&(self.idx2item is not None):
                    predictions[self.target_col]=predictions[self.target_col].map(self.idx2item)
    
        return predictions

    def score(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        model: Optional[Dict[str, Pipeline]] = None
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
        if X is None:
            X_data = self.X
            y_data = self.y
        else:
            X_data = X
            y_data = y

        pred = self.predict(X=X_data, model=model)
        
        # 各ターゲット変数に対するスコアを計算
        if self.task == "regression":
            score_df = pd.DataFrame(
                {
                    'RMSE': [root_mean_squared_error(y_data, pred)],
                    'MAE': [mean_absolute_error(y_data, pred)],
                    'MAPE': [mean_absolute_percentage_error(y_data, pred)],
                    'R2': [r2_score(y_data, pred)]
                }
            )
        else:
            average = "binary" if len(np.unique(self.target_items))<=2 else "macro"
            score_df = pd.DataFrame(
                {
                    'ACCURACY': [accuracy_score(y_data, pred)],
                    'PRECISION': [precision_score(y_data, pred, average=average)],
                    'RECALL': [recall_score(y_data, pred, average=average)],
                    'F1': [f1_score(y_data, pred, average=average)]
                }
            )

        return score_df

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
        # 入力データが指定されていない場合は学習データを使用
        if X is None:
            x = self.X
            Y = self.y
        else:
            x = X
            Y = y

        # クロスバリデーションの設定
        if method == 'kfold':
            CV = KFold(n_splits=n_splits, random_state=0, shuffle=True)
        else:
            CV = LeaveOneOut()

        score_train_cv = []
        score_test_cv = []
        predicts_train_cv = []
        predicts_test_cv = []

        # クロスバリデーションを実行
        for train_index, test_index in CV.split(x):
            _X_train, _X_test = x.iloc[train_index], x.iloc[test_index]
            _y_train, _y_test = Y.iloc[train_index], Y.iloc[test_index]
            cv_model = self._make_pipeline()
            cv_model = cv_fit(
                _X_train,
                _y_train,
                cv_model,
                self.cat_index,
                self.cat_index_fit,
                self.model_names,
                self.task,
                self.sampling_method,
                ensemble=self.ensemble,
            )

            # トレーニングとテストスコアを計算
            cv_train = self.score(_X_train, _y_train, cv_model)
            train_predicts = self.predict(_X_train, cv_model)
            score_train_cv.append(cv_train)
            train_predicts["index"] = train_index
            predicts_train_cv.append(train_predicts)
            
            cv_test = self.score(_X_test, _y_test, cv_model)
            test_predicts = self.predict(_X_test, cv_model)
            score_test_cv.append(cv_test)
            test_predicts["index"] = test_index
            predicts_test_cv.append(test_predicts)
            
        # 各 fold のスコアを結合・平均し、ターゲット別に統計量をまとめる
        score_train_cv = pd.concat(score_train_cv).groupby(level=0).mean()
        score_test_cv = pd.concat(score_test_cv).groupby(level=0).mean()

        predicts_train_cv = pd.concat(predicts_train_cv).reset_index(drop=True).set_index("index").sort_index().groupby('index').mean()
        predicts_test_cv = pd.concat(predicts_test_cv).reset_index(drop=True).set_index("index").sort_index()
        
        self.cv_scores = {
            'train': score_train_cv,
            'test': score_test_cv
        }
        self.cv_preds = {
            'train': predicts_train_cv,
            'test': predicts_test_cv
        }

    def shap(
        self,
        X=None
    ) -> None:
        """
        SHAP値を計算する関数。各ターゲット変数に対してSHAP値とエクスプレイナーを取得。
        """
        self.shap_values, self.base_values, self.explainer = {}, {}, {}
        if X is not None:
            df_prerpocessed = pd.DataFrame(
                self.model['preprocess'].transform(X[self.feature_names]),
                columns=self.feature_names
            )
        else:
            df_prerpocessed = self.df_prerpocessed
       
        shap_values, base_values, explainer, X_sample = get_shap_values(self.model['predictor'], df_prerpocessed)

        self.shap_values = shap_values
        self.base_values = base_values
        self.explainer = explainer
        self.X_sample = X_sample
    
    def model_importance(self):
        if  self.task == "AD":
            return None
        return get_importances(
            self.model["predictor"]
        )
        
    def pfi_importance(self):
        if  self.task == "AD":
            return None
        return get_pfi_values(
            self.model["predictor"],
            self.df_prerpocessed,
            self.y
        )

    def shap_importance(self):
        if self.shap_values is None:
            return None
        elif len(self.shap_values.shape)==2:
            return np.sqrt((self.shap_values**2).sum(axis=0))
        else:
            return np.sqrt((self.shap_values**2).sum(axis=2).sum(axis=0))        

    def _combine_cat_importance(self, imp):
        if imp is None:
            return None
        if self.decomposition:
            return imp

        feature_names = self.df_prerpocessed.columns
        num_cols = [c in self.num_cols for c in feature_names]
        cat_cols = [c in self.cat_cols for c in feature_names]

        mat_imp = [imp[num_cols]]

        for col in self.cat_cols:
            mat_imp.append(imp[[name.startswith(f"{col}_") for name in feature_names]].sum(keepdims=True))
            
        material_cols = feature_names.str.startswith(("smiles__", "comp__"))
        smiles_cols = feature_names.str.startswith("smiles__")
        comp_cols = feature_names.str.startswith("comp__")
        non_material_cols = ~feature_names.str.startswith(("smiles__", "comp__"))

        if sum(material_cols) > 0 and len(imp) > 0:
            
            if sum(smiles_cols)>0:
                mat_imp.append(imp[smiles_cols].sum(keepdims=True))
            if sum(comp_cols)>0:
                mat_imp.append(imp[comp_cols].sum(keepdims=True))
        imp = np.concatenate(mat_imp)
        return imp
        
    def _combine_shap(self):
        if self.shap_values is None:
            return None
        if self.decomposition:
            return self.shap_values

        shap_values = self.shap_values
        feature_names = self.df_prerpocessed.columns
        num_cols = [i for i, c in enumerate(feature_names) if c in self.num_cols]
        cat_cols = [i for i, c in enumerate(feature_names) if c in self.cat_cols]
        mat_shap = [shap_values[:, num_cols]]
    
        for col in self.cat_cols:
            _idx = [i for i, name in enumerate(feature_names) if name.startswith(f"{col}_")]
            mat_shap.append(shap_values[:,_idx].sum(axis=1, keepdims=True))
        
        ma_idx = [i for i, name in enumerate(feature_names) if name.startswith(("smiles__", "comp__"))]
        sm_idx = [i for i, name in enumerate(feature_names) if name.startswith("smiles__")]
        co_idx = [i for i, name in enumerate(feature_names) if name.startswith("comp__")]
    
        if sum(ma_idx) > 0 and len(shap_values) > 0:
            if sum(sm_idx)>0:
                mat_shap.append(shap_values[:,sm_idx].sum(axis=1, keepdims=True))
            if sum(co_idx)>0:
                mat_shap.append(shap_values[:,co_idx].sum(axis=1, keepdims=True))
    
        shap = np.concatenate(mat_shap, axis=1)
        return shap
    
    def get_shap_scatter_data(self, target_col):
        if self.shap_values is not None:
            df_prerpocessed = self.X_sample
            return get_shap_scatter(
                X=df_prerpocessed,
                shap_values=self.shap_values,
                target_col=target_col,
                modelname=self.model_names[0],
                unique_dict=self.unique_cols,
                smiles_cols=self.smiles_cols,
                comp_cols=self.comp_cols,
                le=self.le
            )
        else:
            return None

    def get_pd_and_ice(self, target_col):
        return get_pd_and_ice(
            X=self.X,
            _model=self,
            target=target_col,
            unique_dict=self.unique_cols,
        )

    def get_pd_2d(self, target_cols):
        return get_pd_and_ice_2d(
            X=self.X,
            _model=self,
            targets=target_cols,
            unique_dict=self.unique_cols,
            bounds=None
        )

    def get_xai(self):
        self.importances = {
            "model":self.model_importance(),
            "pfi":self.pfi_importance(),
            "shap":self.shap_importance(),
            "shap_pd": {feature: self.get_shap_scatter_data(feature) for feature in self.feature_names},
            "pd": {feature: self.get_pd_and_ice(feature) for feature in self.all_cols}
        }
        self.importances["model_combine"] = self._combine_cat_importance(self.importances["model"])
        self.importances["pfi_combine"] = self._combine_cat_importance(self.importances["pfi"])
        self.importances["shap_combine"] = self._combine_cat_importance(self.importances["shap"])
        
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
        obj_value = None,
        model: Optional[Dict[str, Pipeline]] = None
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
        y_obj = self.predict(X, model, self.task == "classification")

        if self.task == "classification":
            if obj_value is not None:
                y_obj_result = 1-y_obj[self.target_col+"_"+obj_value].values
            else:
                y_obj_result = 1-y_obj.iloc[:,[-1]].values
            y_obj_result = pd.DataFrame(y_obj_result, columns=[self.target_col])
        else:
            # 目的変数ごとに調整
            if obj_value is not None:
                y_obj_result = np.sqrt((y_obj - obj_value) ** 2)
            else:
                y_obj_result = y_obj

        return y_obj_result

    def to_checkpoint(
        self,
        include_data: bool = False,
        include_preprocessed: bool = False,
        include_cv: bool = False,
        include_importances: bool = False,
        include_shap_arrays: bool = False,
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {}

        keys_core = [
            "num_cols", "cat_cols", "all_cols", "target_col",
            "smiles_cols", "comp_cols", "unique_cols",
            "task", "model_names", "ensemble", "ens_type", "base_model",
            "model_params", "base_model_param", "tuning",
            "num_impute_type", "num_scale_type", "cat_impute",
            "poly", "poly_degree", "poly_interaction_only",
            "pca", "pca_n_components",
            "sampling_method",
            "fingerprints", "comp_method", "comp_feats",
            "ad", "le", "target_items", "item2idx", "idx2item",
            "feature_names", "model", "cat_index", "cat_index_fit",
        ]
        for k in keys_core:
            if hasattr(self, k):
                state[k] = getattr(self, k)

        if include_data:
            for k in ["X", "y"]:
                if hasattr(self, k):
                    state[k] = getattr(self, k)

        if include_preprocessed:
            for k in ["df_prerpocessed"]:
                if hasattr(self, k):
                    state[k] = getattr(self, k)

        if include_cv:
            for k in ["cv_scores", "cv_preds"]:
                if hasattr(self, k):
                    state[k] = getattr(self, k)

        if include_importances and hasattr(self, "importances"):
            state["importances"] = getattr(self, "importances")

        if include_shap_arrays:
            for k in ["shap_values", "base_values", "X_sample"]:
                if hasattr(self, k):
                    state[k] = getattr(self, k)
        
        # explainer は pickle 互換性の問題があるため保存しない
        state.pop("explainer", None)

        return state

    @classmethod
    def load_checkpoint(cls, state: Dict[str, Any]) -> "SingleOutputMLModelPipeline":
        state = state.copy()
        obj = cls.__new__(cls)
        # explainer は保存していないため、復元時にも削除しておく
        state.pop("explainer", None)
        obj.__dict__.update(state)
        return obj
    
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
        num_cols = [] if num_cols is None else num_cols
        cat_cols = [] if cat_cols is None else cat_cols
        smiles_cols = [] if (smiles_cols is None or smiles_cols==[None]) else smiles_cols
        comp_cols = [] if (comp_cols is None or comp_cols == [None]) else comp_cols
        
        # 入力データ、目的変数、数値・カテゴリカル特徴量などの設定
        X = df[num_cols + cat_cols + smiles_cols + comp_cols]
        # self.y = df[target_cols]
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.all_cols = num_cols + cat_cols + smiles_cols + comp_cols
        self.target_cols = target_cols if target_cols is not None else []
        self.model_names = model_names
        self.smiles_cols = smiles_cols
        self.comp_cols = comp_cols
        self.models = {}
        self.feature_names = {}

        # カテゴリカル変数のユニーク値を取得
        self.unique_cols = get_cat_unique_values(X, self.cat_cols+self.smiles_cols+self.comp_cols)
        
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
                self.models[target].fit(
                    df,
                    target_col=self.target_cols[i],
                    task=self.tasks[i],
                    num_cols=self.num_cols,
                    cat_cols=self.cat_cols,
                    model_names=self.model_names[i],
                    smiles_cols=self.smiles_cols,
                    fingerprints=self.fingerprints,
                    comp_cols=self.comp_cols,
                    comp_method=self.comp_method,
                    comp_feats=self.comp_feats,
                    ad=False,
                    impute=impute,
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
                self.models[target].fit(
                    df,
                    target_col=None,
                    task="AD",
                    num_cols=self.num_cols,
                    cat_cols=self.cat_cols,
                    model_names=[self.ad_model_name],
                    smiles_cols=self.smiles_cols,
                    fingerprints=self.fingerprints,
                    comp_cols=self.comp_cols,
                    comp_method=self.comp_method,
                    comp_feats=self.comp_feats,
                    ad=True,
                    impute=impute,
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
        X_data = models[target_cols[0]].X if X is None else X
        
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

        # 各 SingleOutputMLModelPipeline インスタンスを復元し、obj.models に格納
        restored_models = {} # 新しい辞書を準備
        for target_name, ckpt_data in models_checkpoints.items():
            restored_models[target_name] = SingleOutputMLModelPipeline.load_checkpoint(ckpt_data)
        obj.models = restored_models # 復元されたモデルをobj.modelsに代入

        return obj