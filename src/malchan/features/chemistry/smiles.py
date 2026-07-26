"""SMILES文字列を分子オブジェクトへ変換するTransformer。"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def _load_smiles_transformers() -> tuple[type[Any], type[Any]]:
    """scikit-fingerprintsのSMILES変換器を遅延importする。"""
    try:
        from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer
    except ImportError as exc:
        raise ImportError(
            "SMILES特徴量を使用するにはscikit-fingerprintsを導入してください。"
        ) from exc
    return MolFromSmilesTransformer, ConformerGenerator


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
        """内部Transformerとキャッシュを初期化する。"""
        mol_transformer_cls, conformer_generator_cls = _load_smiles_transformers()
        self.mol_transformer_ = mol_transformer_cls(
            sanitize=self.sanitize,
            valid_only=False,
            n_jobs=self.mol_n_jobs,
        )
        self.conf_transformer_ = conformer_generator_cls(
            num_conformers=self.conf_num_conformers,
            errors=self.conf_errors,
            n_jobs=self.conf_n_jobs,
            random_state=self.random_state,
        )
        self._mol_cache: dict[Any, Any] | None = {} if self.use_cache else None
        self._conf_cache: dict[Any, Any] | None = {} if self.use_cache else None
        return self

    def transform(self, X: Any) -> list[Any]:
        """入力順を維持してSMILESをMolへ変換する。"""
        smiles = np.asarray(X).ravel().tolist()
        unique_smiles = list(dict.fromkeys(smiles))

        if self.use_cache:
            missing = [
                value
                for value in unique_smiles
                if value not in self._mol_cache
            ]
            if missing:
                generated = self.mol_transformer_.transform(missing)
                self._mol_cache.update(
                    dict(zip(missing, generated, strict=False))
                )
            mol_by_smiles = self._mol_cache
        else:
            generated = self.mol_transformer_.transform(unique_smiles)
            mol_by_smiles = dict(zip(unique_smiles, generated, strict=False))

        molecules = [mol_by_smiles[value] for value in smiles]
        if self.error_on_invalid and any(molecule is None for molecule in molecules):
            invalid_rows = [
                index
                for index, molecule in enumerate(molecules)
                if molecule is None
            ][:10]
            raise ValueError(f"Invalid SMILES at rows: {invalid_rows}")

        if not self.generate_conformers:
            self._clear_cache_if_needed()
            return molecules

        if self.use_cache:
            missing_conformers = [
                value
                for value in unique_smiles
                if value not in self._conf_cache
            ]
            valid_pairs = [
                (value, mol_by_smiles[value])
                for value in missing_conformers
                if mol_by_smiles[value] is not None
            ]
            if valid_pairs:
                values, valid_molecules = zip(*valid_pairs, strict=False)
                generated = self.conf_transformer_.transform(list(valid_molecules))
                self._conf_cache.update(
                    dict(zip(values, generated, strict=False))
                )
            for value in missing_conformers:
                self._conf_cache.setdefault(value, None)

            result = [self._conf_cache[value] for value in smiles]
            self._clear_cache_if_needed()
            return result

        valid_pairs = [
            (value, mol_by_smiles[value])
            for value in unique_smiles
            if mol_by_smiles[value] is not None
        ]
        conformers: dict[Any, Any] = {value: None for value in unique_smiles}
        if valid_pairs:
            values, valid_molecules = zip(*valid_pairs, strict=False)
            generated = self.conf_transformer_.transform(list(valid_molecules))
            conformers.update(dict(zip(values, generated, strict=False)))
        return [conformers[value] for value in smiles]

    def _clear_cache_if_needed(self) -> None:
        """設定上限を超えたキャッシュを解放する。"""
        if not self.use_cache or not self.cache_max:
            return
        if len(self._mol_cache) <= self.cache_max:
            return
        self._mol_cache.clear()
        self._conf_cache.clear()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """後段へ渡す分子列名を返す。"""
        return np.array(["mol"], dtype=object)


__all__ = ["SmilesToMol"]
