"""Single-output model pipeline public entrypoint."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import optuna
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.pipeline import Pipeline

from ..models.explainability import (
    get_importances,
    get_pd_and_ice,
    get_pd_and_ice_2d,
    get_pfi_values,
    get_shap_scatter,
    get_shap_values,
)
from ..models.pipelines import make_pipeline
from ..models.training import cv_fit, fit_model, tune_model
from ..models.utils import (
    feature_names_from_pipeline,
    get_cat_unique_values,
    label_encode,
)

__all__ = ["SingleOutputMLModelPipeline"]

optuna.logging.disable_default_handler()
warnings.simplefilter("ignore")


class SingleOutputMLModelPipeline:
    """Single-target machine-learning model pipeline."""

    def __init__(self):
        """機械学習モデルパイプラインの初期化。"""

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
        decomposition: bool = False,
        decomposition_method: str = "PCA",
        dec_n_components: int = 2,
        sampling_method=None,
    ) -> None:
        """モデルパイプラインをデータに適合させる。"""

        num_cols = [] if num_cols is None else num_cols
        cat_cols = [] if cat_cols is None else cat_cols
        smiles_cols = [] if (smiles_cols is None or smiles_cols == [None]) else smiles_cols
        comp_cols = [] if (comp_cols is None or comp_cols == [None]) else comp_cols

        all_cols = num_cols + cat_cols + smiles_cols + comp_cols
        if target_col is not None:
            all_cols = all_cols + [target_col]

        if impute:
            if task == "AD":
                _df = df[all_cols].dropna()
                self.X = _df[num_cols + cat_cols + smiles_cols + comp_cols]
                self.y = _df[[target_col]] if target_col is not None else None
            elif task == "regression":
                iterative_imputer = IterativeImputer(max_iter=10, random_state=0)
                _df = pd.DataFrame(
                    iterative_imputer.fit_transform(df[num_cols + [target_col]]),
                    columns=num_cols + [target_col],
                )
                self.X = df[num_cols + cat_cols + smiles_cols + comp_cols]
                self.y = _df[[target_col]] if target_col is not None else None
            else:
                imputer = SimpleImputer(strategy="most_frequent")
                self.y = pd.DataFrame(
                    imputer.fit_transform(df[[target_col]]),
                    columns=[target_col],
                )
                self.X = df[num_cols + cat_cols + smiles_cols + comp_cols]
        else:
            _df = df[all_cols].dropna()
            self.X = _df[num_cols + cat_cols + smiles_cols + comp_cols]
            self.y = _df[[target_col]] if target_col is not None else None

        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.all_cols = num_cols + cat_cols + smiles_cols + comp_cols
        self.target_col = target_col if not ad else "AD"
        self.smiles_cols = smiles_cols
        self.comp_cols = comp_cols
        self.unique_cols = get_cat_unique_values(
            self.X,
            self.cat_cols + self.smiles_cols + self.comp_cols,
        )

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
        self.decomposition = decomposition
        self.decomposition_method = decomposition_method
        self.dec_n_components = dec_n_components

        self.sampling_method = sampling_method if task == "classification" else None

        self.fingerprints = fingerprints
        self.comp_method = comp_method
        self.comp_feats = comp_feats
        self.ad = ad

        self.cat_index = [self.all_cols.index(c) for c in cat_cols + smiles_cols + comp_cols]
        self.cat_index_fit = (
            []
            if decomposition or poly or self.ensemble
            else [self.all_cols.index(c) for c in cat_cols]
        )

        self.model = self._make_pipeline()
        if self.task == "classification":
            self.target_items = np.unique(self.y.values)
            self.y, self.le = label_encode(self.y)
            self.idx2item = {k: v for k, v in zip(self.y.unique(), self.target_items)}
            self.item2idx = {k: v for k, v in zip(self.target_items, self.y.unique())}
        else:
            self.sampling_method = None

        if self.tuning:
            self.model, best_params, best_base_param = tune_model(
                X=self.X,
                y=self.y,
                model_pipeline=self.model,
                model_names=self.model_names,
                base_model=self.base_model,
                ens_type=self.ens_type,
                sampling_method=self.sampling_method,
                cat_index=self.cat_index,
                cat_index_fit=self.cat_index_fit,
                task=self.task,
                n_trials=30,
                verbose=2,
            )
            self.model_params = best_params
            self.base_model_param = best_base_param
        else:
            self.model = fit_model(
                X=self.X,
                y=self.y,
                model_pipeline=self.model,
                model_names=self.model_names,
                ensemble=self.ensemble,
                sampling_method=self.sampling_method,
                cat_index=self.cat_index,
                cat_index_fit=self.cat_index_fit,
            )

        self.feature_names = feature_names_from_pipeline(self.model)
        self.df_prerpocessed = pd.DataFrame(
            self.model["preprocess"].transform(self.X),
            columns=self.feature_names,
        )

    def _make_pipeline(self) -> Pipeline:
        """前処理とモデルパイプラインを作成する。"""

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
            base_model_params=self.base_model_param,
        )
        return model_items[0]

    def predict(
        self,
        X: Optional[pd.DataFrame] = None,
        model: Optional[Dict[str, Pipeline]] = None,
        proba=False,
        idx2item=False,
    ) -> pd.DataFrame:
        """モデルを用いて予測を行う。"""

        models_pred = self.model if model is None else model
        X_data = self.X if X is None else X
        if proba:
            predictions = models_pred.predict_proba(X_data[self.all_cols])
        elif self.task == "AD":
            predictions = models_pred.decision_function(X_data[self.all_cols])
        else:
            predictions = models_pred.predict(X_data[self.all_cols]).reshape(-1, 1)

        if self.task == "regression":
            predictions = pd.DataFrame(predictions, columns=[self.target_col])
        else:
            if proba:
                predictions = pd.DataFrame(
                    predictions,
                    columns=[
                        self.target_col + "_" + c
                        for c in self.le.inverse_transform(
                            np.arange(predictions.shape[-1])
                        ).astype(str)
                    ],
                )
            else:
                predictions = pd.DataFrame(predictions, columns=[self.target_col])
                if (idx2item) & (self.idx2item is not None):
                    predictions[self.target_col] = predictions[self.target_col].map(self.idx2item)

        return predictions

    def score(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        model: Optional[Dict[str, Pipeline]] = None,
    ) -> pd.DataFrame:
        """モデルのスコアを計算する。"""

        if X is None:
            X_data = self.X
            y_data = self.y
        else:
            X_data = X
            y_data = y

        pred = self.predict(X=X_data, model=model)

        if self.task == "regression":
            score_df = pd.DataFrame(
                {
                    "RMSE": [root_mean_squared_error(y_data, pred)],
                    "MAE": [mean_absolute_error(y_data, pred)],
                    "MAPE": [mean_absolute_percentage_error(y_data, pred)],
                    "R2": [r2_score(y_data, pred)],
                }
            )
        else:
            average = "binary" if len(np.unique(self.target_items)) <= 2 else "macro"
            score_df = pd.DataFrame(
                {
                    "ACCURACY": [accuracy_score(y_data, pred)],
                    "PRECISION": [precision_score(y_data, pred, average=average)],
                    "RECALL": [recall_score(y_data, pred, average=average)],
                    "F1": [f1_score(y_data, pred, average=average)],
                }
            )

        return score_df

    def cv_score(
        self,
        method: Optional[str] = "kfold",
        n_splits: Optional[int] = 5,
        X: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """クロスバリデーションを用いてモデルのスコアを計算する。"""

        if X is None:
            x = self.X
            Y = self.y
        else:
            x = X
            Y = y

        if method == "kfold":
            CV = KFold(n_splits=n_splits, random_state=0, shuffle=True)
        else:
            CV = LeaveOneOut()

        score_train_cv = []
        score_test_cv = []
        predicts_train_cv = []
        predicts_test_cv = []

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

        score_train_cv = pd.concat(score_train_cv).groupby(level=0).mean()
        score_test_cv = pd.concat(score_test_cv).groupby(level=0).mean()

        predicts_train_cv = (
            pd.concat(predicts_train_cv)
            .reset_index(drop=True)
            .set_index("index")
            .sort_index()
            .groupby("index")
            .mean()
        )
        predicts_test_cv = (
            pd.concat(predicts_test_cv)
            .reset_index(drop=True)
            .set_index("index")
            .sort_index()
        )

        self.cv_scores = {"train": score_train_cv, "test": score_test_cv}
        self.cv_preds = {"train": predicts_train_cv, "test": predicts_test_cv}

    def shap(self, X=None) -> None:
        """SHAP値を計算する。"""

        self.shap_values, self.base_values, self.explainer = {}, {}, {}
        if X is not None:
            df_prerpocessed = pd.DataFrame(
                self.model["preprocess"].transform(X[self.feature_names]),
                columns=self.feature_names,
            )
        else:
            df_prerpocessed = self.df_prerpocessed

        shap_values, base_values, explainer, X_sample = get_shap_values(
            self.model["predictor"],
            df_prerpocessed,
        )

        self.shap_values = shap_values
        self.base_values = base_values
        self.explainer = explainer
        self.X_sample = X_sample

    def model_importance(self):
        if self.task == "AD":
            return None
        return get_importances(self.model["predictor"])

    def pfi_importance(self):
        if self.task == "AD":
            return None
        return get_pfi_values(self.model["predictor"], self.df_prerpocessed, self.y)

    def shap_importance(self):
        shap_values = getattr(self, "shap_values", None)
        if shap_values is None:
            return None
        elif len(shap_values.shape) == 2:
            return np.sqrt((shap_values**2).sum(axis=0))
        else:
            return np.sqrt((shap_values**2).sum(axis=2).sum(axis=0))

    def _combine_cat_importance(self, imp):
        if imp is None:
            return None
        if self.decomposition:
            return imp

        feature_names = self.df_prerpocessed.columns
        num_cols = [c in self.num_cols for c in feature_names]
        mat_imp = [imp[num_cols]]

        for col in self.cat_cols:
            mat_imp.append(
                imp[[name.startswith(f"{col}_") for name in feature_names]].sum(
                    keepdims=True
                )
            )

        smiles_cols = feature_names.str.startswith("smiles__")
        comp_cols = feature_names.str.startswith("comp__")
        material_cols = feature_names.str.startswith(("smiles__", "comp__"))

        if sum(material_cols) > 0 and len(imp) > 0:
            if sum(smiles_cols) > 0:
                mat_imp.append(imp[smiles_cols].sum(keepdims=True))
            if sum(comp_cols) > 0:
                mat_imp.append(imp[comp_cols].sum(keepdims=True))

        return np.concatenate(mat_imp)

    def _combine_shap(self):
        shap_values = getattr(self, "shap_values", None)
        if shap_values is None:
            return None
        if self.decomposition:
            return shap_values

        feature_names = self.df_prerpocessed.columns
        num_cols = [i for i, c in enumerate(feature_names) if c in self.num_cols]
        mat_shap = [shap_values[:, num_cols]]

        for col in self.cat_cols:
            _idx = [i for i, name in enumerate(feature_names) if name.startswith(f"{col}_")]
            mat_shap.append(shap_values[:, _idx].sum(axis=1, keepdims=True))

        sm_idx = [i for i, name in enumerate(feature_names) if name.startswith("smiles__")]
        co_idx = [i for i, name in enumerate(feature_names) if name.startswith("comp__")]
        ma_idx = [i for i, name in enumerate(feature_names) if name.startswith(("smiles__", "comp__"))]

        if sum(ma_idx) > 0 and len(shap_values) > 0:
            if sum(sm_idx) > 0:
                mat_shap.append(shap_values[:, sm_idx].sum(axis=1, keepdims=True))
            if sum(co_idx) > 0:
                mat_shap.append(shap_values[:, co_idx].sum(axis=1, keepdims=True))

        return np.concatenate(mat_shap, axis=1)

    def get_shap_scatter_data(self, target_col):
        if getattr(self, "shap_values", None) is not None:
            return get_shap_scatter(
                X=self.X_sample,
                shap_values=self.shap_values,
                target_col=target_col,
                modelname=self.model_names[0],
                unique_dict=self.unique_cols,
                smiles_cols=self.smiles_cols,
                comp_cols=self.comp_cols,
                le=self.le,
            )
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
            bounds=None,
        )

    def get_xai(self):
        self.importances = {
            "model": self.model_importance(),
            "pfi": self.pfi_importance(),
            "shap": self.shap_importance(),
            "shap_pd": {
                feature: self.get_shap_scatter_data(feature)
                for feature in self.feature_names
            },
            "pd": {feature: self.get_pd_and_ice(feature) for feature in self.all_cols},
        }
        self.importances["model_combine"] = self._combine_cat_importance(
            self.importances["model"]
        )
        self.importances["pfi_combine"] = self._combine_cat_importance(
            self.importances["pfi"]
        )
        self.importances["shap_combine"] = self._combine_cat_importance(
            self.importances["shap"]
        )

    def get_instance_vars(self) -> Dict[str, Any]:
        """インスタンス変数を取得する。"""

        return vars(self)

    def predict_objective(
        self,
        X: Optional[pd.DataFrame] = None,
        obj_value=None,
        model: Optional[Dict[str, Pipeline]] = None,
    ) -> pd.DataFrame:
        """特定の目的変数を考慮して予測を行う。"""

        y_obj = self.predict(X, model, self.task == "classification")

        if self.task == "classification":
            if obj_value is not None:
                y_obj_result = 1 - y_obj[self.target_col + "_" + obj_value].values
            else:
                y_obj_result = 1 - y_obj.iloc[:, [-1]].values
            y_obj_result = pd.DataFrame(y_obj_result, columns=[self.target_col])
        else:
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
        """Checkpoint用の状態辞書を作成する。"""

        state: Dict[str, Any] = {}
        keys_core = [
            "num_cols",
            "cat_cols",
            "all_cols",
            "target_col",
            "smiles_cols",
            "comp_cols",
            "unique_cols",
            "task",
            "model_names",
            "ensemble",
            "ens_type",
            "base_model",
            "model_params",
            "base_model_param",
            "tuning",
            "num_impute_type",
            "num_scale_type",
            "cat_impute",
            "poly",
            "poly_degree",
            "poly_interaction_only",
            "pca",
            "pca_n_components",
            "sampling_method",
            "fingerprints",
            "comp_method",
            "comp_feats",
            "ad",
            "le",
            "target_items",
            "item2idx",
            "idx2item",
            "feature_names",
            "model",
            "cat_index",
            "cat_index_fit",
        ]
        for key in keys_core:
            if hasattr(self, key):
                state[key] = getattr(self, key)

        if include_data:
            for key in ["X", "y"]:
                if hasattr(self, key):
                    state[key] = getattr(self, key)

        if include_preprocessed:
            for key in ["df_prerpocessed"]:
                if hasattr(self, key):
                    state[key] = getattr(self, key)

        if include_cv:
            for key in ["cv_scores", "cv_preds"]:
                if hasattr(self, key):
                    state[key] = getattr(self, key)

        if include_importances and hasattr(self, "importances"):
            state["importances"] = getattr(self, "importances")

        if include_shap_arrays:
            for key in ["shap_values", "base_values", "X_sample"]:
                if hasattr(self, key):
                    state[key] = getattr(self, key)

        state.pop("explainer", None)
        return state

    @classmethod
    def load_checkpoint(cls, state: Dict[str, Any]) -> "SingleOutputMLModelPipeline":
        """Checkpoint用の状態辞書から復元する。"""

        state = state.copy()
        obj = cls.__new__(cls)
        state.pop("explainer", None)
        obj.__dict__.update(state)
        return obj
