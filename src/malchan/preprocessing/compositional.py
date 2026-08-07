"""組成データ向けのlog-ratio変換。"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

SUPPORTED_COMPOSITIONAL_METHODS = ("ILR", "CLR", "ALR")


def _as_2d_float_array(X: Any) -> np.ndarray:
    """入力を有限な2次元float配列へ変換する。"""
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise ValueError("組成データは2次元の表形式データで指定してください。")
    if not np.all(np.isfinite(array)):
        raise ValueError("組成データに欠損値または無限値が含まれています。")
    return array


def _prepare_compositions(
    X: Any,
    *,
    zero_replacement: float | None,
    closure: bool,
) -> np.ndarray:
    """非負値検証、closure、ゼロ置換を行う。"""
    array = _as_2d_float_array(X).copy()
    if np.any(array < 0):
        raise ValueError("組成データには負の値を指定できません。")

    row_sums = array.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("各サンプルの組成比には少なくとも1つ正の値が必要です。")

    closed = array / row_sums[:, None]
    if closure:
        array = closed

    zero_mask = array == 0
    if not np.any(zero_mask):
        return array
    if zero_replacement is None:
        raise ValueError(
            "ILR/CLR/ALRでは0をそのまま扱えません。zero_replacementを指定してください。"
        )
    if not 0 < zero_replacement < 1:
        raise ValueError("zero_replacementは0より大きく1より小さい値を指定してください。")

    # log-ratioは共通倍率に不変なので、ゼロを含む行はclosure後の組成上で
    # multiplicative replacementを行う。非ゼロ成分の比率は保持される。
    replaced = closed.copy()
    for row_index in range(replaced.shape[0]):
        row_zero_mask = replaced[row_index] == 0
        zero_count = int(row_zero_mask.sum())
        if zero_count == 0:
            continue
        remaining_mass = 1.0 - zero_count * zero_replacement
        if remaining_mass <= 0:
            raise ValueError(
                "zero_replacementが大きすぎます。ゼロ成分数との積が1未満になる値を指定してください。"
            )
        replaced[row_index, row_zero_mask] = zero_replacement
        replaced[row_index, ~row_zero_mask] *= remaining_mass

    return replaced


def _input_feature_names(X: Any, n_features: int) -> np.ndarray:
    """DataFrame列名があれば保存し、なければx0形式の名前を生成する。"""
    columns = getattr(X, "columns", None)
    if columns is not None and len(columns) == n_features:
        return np.asarray([str(column) for column in columns], dtype=object)
    return np.asarray([f"x{index}" for index in range(n_features)], dtype=object)


def _resolve_input_features(
    input_features: Any,
    *,
    feature_names_in: np.ndarray,
    n_features_in: int,
) -> np.ndarray:
    if input_features is None:
        return feature_names_in
    names = np.asarray(input_features, dtype=object)
    if names.shape != (n_features_in,):
        raise ValueError(f"input_featuresは{n_features_in}個指定してください。")
    return names.astype(str)


class _BaseLogRatioTransformer(BaseEstimator, TransformerMixin):
    """log-ratio変換に共通する組成前処理を提供する基底クラス。"""

    def __init__(
        self,
        *,
        zero_replacement: float | None = 1e-6,
        closure: bool = True,
    ) -> None:
        self.zero_replacement = zero_replacement
        self.closure = closure

    def _fit_common(self, X: Any, *, min_features: int = 2) -> np.ndarray:
        array = _prepare_compositions(
            X,
            zero_replacement=self.zero_replacement,
            closure=self.closure,
        )
        if array.shape[1] < min_features:
            raise ValueError(f"組成変換には{min_features}列以上の組成比が必要です。")
        self.n_features_in_ = array.shape[1]
        self.feature_names_in_ = _input_feature_names(X, self.n_features_in_)
        return array

    def _transform_common(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        array = _prepare_compositions(
            X,
            zero_replacement=self.zero_replacement,
            closure=self.closure,
        )
        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"fit時は{self.n_features_in_}列でしたが、transform時は{array.shape[1]}列です。"
            )
        return array


class ILRTransformer(_BaseLogRatioTransformer):
    """組成比をisometric log-ratio (ILR) 座標へ変換する。"""

    def __init__(
        self,
        *,
        zero_replacement: float | None = 1e-6,
        closure: bool = True,
        prefix: str = "ilr",
    ) -> None:
        super().__init__(zero_replacement=zero_replacement, closure=closure)
        self.prefix = prefix

    def fit(self, X: Any, y: Any = None) -> ILRTransformer:
        """入力次元からHelmert型の直交基底を構築する。"""
        del y
        array = self._fit_common(X)
        n_features = array.shape[1]
        basis = np.zeros((n_features - 1, n_features), dtype=float)
        for index in range(1, n_features):
            scale = np.sqrt(index * (index + 1))
            basis[index - 1, :index] = 1.0 / scale
            basis[index - 1, index] = -index / scale
        self.basis_ = basis
        return self

    def transform(self, X: Any) -> np.ndarray:
        """組成比をD-1次元のILR座標へ変換する。"""
        check_is_fitted(self, "basis_")
        array = self._transform_common(X)
        return np.log(array) @ self.basis_.T

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """ILR balance座標の特徴量名を返す。"""
        check_is_fitted(self, "basis_")
        _resolve_input_features(
            input_features,
            feature_names_in=self.feature_names_in_,
            n_features_in=self.n_features_in_,
        )
        return np.asarray(
            [f"{self.prefix}__balance_{index}" for index in range(1, self.n_features_in_)],
            dtype=object,
        )

    def get_balance_definitions(self) -> list[dict[str, Any]]:
        """各ILR座標が比較する成分群を返す。"""
        check_is_fitted(self, "basis_")
        definitions: list[dict[str, Any]] = []
        for index in range(1, self.n_features_in_):
            definitions.append(
                {
                    "feature": f"{self.prefix}__balance_{index}",
                    "positive": self.feature_names_in_[:index].tolist(),
                    "negative": [str(self.feature_names_in_[index])],
                }
            )
        return definitions


class CLRTransformer(_BaseLogRatioTransformer):
    """組成比をcentered log-ratio (CLR) 座標へ変換する。"""

    def __init__(
        self,
        *,
        zero_replacement: float | None = 1e-6,
        closure: bool = True,
        prefix: str = "clr",
    ) -> None:
        super().__init__(zero_replacement=zero_replacement, closure=closure)
        self.prefix = prefix

    def fit(self, X: Any, y: Any = None) -> CLRTransformer:
        del y
        self._fit_common(X)
        return self

    def transform(self, X: Any) -> np.ndarray:
        array = self._transform_common(X)
        log_array = np.log(array)
        return log_array - log_array.mean(axis=1, keepdims=True)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        names = _resolve_input_features(
            input_features,
            feature_names_in=self.feature_names_in_,
            n_features_in=self.n_features_in_,
        )
        return np.asarray([f"{self.prefix}__{name}" for name in names], dtype=object)


class ALRTransformer(_BaseLogRatioTransformer):
    """組成比をadditive log-ratio (ALR) 座標へ変換する。"""

    def __init__(
        self,
        *,
        reference: int | str = -1,
        zero_replacement: float | None = 1e-6,
        closure: bool = True,
        prefix: str = "alr",
    ) -> None:
        super().__init__(zero_replacement=zero_replacement, closure=closure)
        self.reference = reference
        self.prefix = prefix

    def fit(self, X: Any, y: Any = None) -> ALRTransformer:
        del y
        self._fit_common(X)
        if isinstance(self.reference, str):
            matches = np.flatnonzero(self.feature_names_in_ == self.reference)
            if len(matches) == 0:
                raise ValueError(f"ALRの基準成分 {self.reference!r} が入力列にありません。")
            reference_index = int(matches[0])
        else:
            reference_index = int(self.reference)
            if reference_index < 0:
                reference_index += self.n_features_in_
            if not 0 <= reference_index < self.n_features_in_:
                raise ValueError("ALRのreferenceが組成列の範囲外です。")
        self.reference_index_ = reference_index
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "reference_index_")
        array = self._transform_common(X)
        numerator_indices = [
            index for index in range(self.n_features_in_) if index != self.reference_index_
        ]
        return np.log(array[:, numerator_indices] / array[:, [self.reference_index_]])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "reference_index_")
        names = _resolve_input_features(
            input_features,
            feature_names_in=self.feature_names_in_,
            n_features_in=self.n_features_in_,
        )
        denominator = names[self.reference_index_]
        return np.asarray(
            [
                f"{self.prefix}__{name}_over_{denominator}"
                for index, name in enumerate(names)
                if index != self.reference_index_
            ],
            dtype=object,
        )


def make_compositional_preprocess(
    method: str | None = "ILR",
    *,
    zero_replacement: float | None = 1e-6,
    closure: bool = True,
    alr_reference: int | str = -1,
    scale_type: str | None = None,
) -> Pipeline | None:
    """組成比列向けのlog-ratio前処理Pipelineを作成する。"""
    if method is None:
        return None

    normalized_method = method.upper()
    transformer: Any
    if normalized_method == "ILR":
        transformer = ILRTransformer(
            zero_replacement=zero_replacement,
            closure=closure,
        )
    elif normalized_method == "CLR":
        transformer = CLRTransformer(
            zero_replacement=zero_replacement,
            closure=closure,
        )
    elif normalized_method == "ALR":
        transformer = ALRTransformer(
            reference=alr_reference,
            zero_replacement=zero_replacement,
            closure=closure,
        )
    else:
        raise ValueError(
            f"不正なcompositional_methodです。{list(SUPPORTED_COMPOSITIONAL_METHODS)}から指定してください。"
        )

    steps: list[tuple[str, Any]] = [(normalized_method.lower(), transformer)]
    scaler = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "centering": StandardScaler(with_std=False),
        "MaxAbsScaler": MaxAbsScaler(),
    }.get(scale_type)
    if scale_type is not None and scaler is None:
        raise ValueError(
            "不正なcompositional_scale_typeです。StandardScaler、MinMaxScaler、centering、MaxAbsScalerから指定してください。"
        )
    if scaler is not None:
        steps.append(("scaler", scaler))
    return Pipeline(steps)


__all__ = [
    "ALRTransformer",
    "CLRTransformer",
    "ILRTransformer",
    "SUPPORTED_COMPOSITIONAL_METHODS",
    "make_compositional_preprocess",
]
