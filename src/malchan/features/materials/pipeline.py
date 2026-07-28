"""材料特徴量バックエンドを選択して前処理Pipelineを構築する。"""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SUPPORTED_COMPOSITION_METHODS = ("xenonpy", "matminer", "mendeleev")


def make_comp_preprocess(
    method: str | None = "xenonpy",
    feats: list[str] | tuple[str, ...] = (),
) -> Pipeline | None:
    """組成式列向けの特徴量生成パイプラインを作成する。

    Args:
        method: ``xenonpy``、``matminer``、``mendeleev``のいずれか。
        feats: Matminer featurizer名またはMendeleev元素プロパティ名。

    Returns:
        選択した材料特徴量パイプライン。methodがNoneの場合はNone。

    Raises:
        ValueError: 未対応のmethodが指定された場合。
    """
    if method is None:
        return None

    normalized_method = method.lower()
    if normalized_method == "xenonpy":
        from .composition import FormulaToFractionDict
        from .xenonpy import XenoCompositionsTransformer, prepare_xenonpy_preset

        prepare_xenonpy_preset()
        return Pipeline(
            [
                ("formula2dict", FormulaToFractionDict(invalid="empty")),
                (
                    "xeno",
                    XenoCompositionsTransformer(
                        prefix="comp__",
                        sample_in_fit=512,
                    ),
                ),
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=False)),
            ]
        )

    if normalized_method == "matminer":
        from .composition import FormulaToComposition
        from .matminer import (
            MatminerCompositionFeaturizer,
            resolve_matminer_featurizers,
        )

        return Pipeline(
            [
                ("f2c", FormulaToComposition(invalid="empty")),
                (
                    "mm",
                    MatminerCompositionFeaturizer(
                        featurizers=resolve_matminer_featurizers(feats),
                        prefix="comp__",
                    ),
                ),
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=False)),
            ]
        )

    if normalized_method == "mendeleev":
        from .composition import FormulaToComposition
        from .mendeleev import MendeleevCompositionFeaturizer

        return Pipeline(
            [
                ("f2c", FormulaToComposition(invalid="empty")),
                (
                    "md",
                    MendeleevCompositionFeaturizer(
                        props=feats,
                        prefix="comp__",
                    ),
                ),
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=False)),
            ]
        )

    raise ValueError(
        "不正なcomp_methodです。xenonpy、matminer、mendeleevから指定してください。"
    )


__all__ = ["SUPPORTED_COMPOSITION_METHODS", "make_comp_preprocess"]
