# Features module layout

`malchan.features` contains chemistry and materials feature generation that is independent from model construction.

Heavy optional dependencies are loaded only when the corresponding descriptor backend is selected.

## Materials

- `malchan.features.materials.composition`: formula parsing, `pymatgen.Composition` conversion, and composition cache keys.
- `malchan.features.materials.xenonpy`: XenonPy preset setup and `XenoCompositionsTransformer`.
- `malchan.features.materials.matminer`: Matminer featurizer resolution and `MatminerCompositionFeaturizer`.
- `malchan.features.materials.mendeleev`: composition-weighted Mendeleev element properties.
- `malchan.features.materials.pipeline`: backend selection and composition preprocessing pipelines.

## Chemistry

- `malchan.features.chemistry.smiles`: SMILES parsing, molecule conversion, optional conformer generation, and conversion caches.
- `malchan.features.chemistry.fingerprints`: fingerprint name resolution, 3D requirement detection, and output feature-name handling.
- `malchan.features.chemistry.pipeline`: SMILES conversion, fingerprint union, scaling, and feature-name preservation.

## Model preprocessing

`malchan.models.pipelines.preprocess_pipeline` only combines numeric, categorical, chemistry, and materials pipelines.

- The chemistry pipeline is imported only when `smiles_cols` is specified.
- The materials pipeline is imported only when `comp_cols` is specified.
- Chemistry and materials implementations are not re-exported from the model preprocessing module.
