# Features module layout

`malchan.features` contains chemistry and materials feature generation that is independent from model construction.

Heavy optional dependencies are loaded only when the corresponding descriptor backend is selected.

## Materials

- `malchan.features.materials.composition`: formula parsing, `pymatgen.Composition` conversion, and composition cache keys.
- `malchan.features.materials.xenonpy`: XenonPy preset setup and `XenoCompositionsTransformer`.
- `malchan.features.materials.matminer`: Matminer featurizer resolution and `MatminerCompositionFeaturizer`.
- `malchan.features.materials.mendeleev`: composition-weighted Mendeleev element properties.
- `malchan.features.materials.pipeline`: backend selection and composition preprocessing pipelines.

`malchan.models.pipelines.preprocess_pipeline` only combines numeric, categorical, SMILES, and optional composition pipelines. It imports the materials pipeline only when `comp_cols` is specified.

## Chemistry

The following migration remains:

- move SMILES parsing into `malchan.features.chemistry.smiles`;
- move RDKit and scikit-fingerprints descriptors into `malchan.features.chemistry.fingerprints`;
- keep chemistry backends optional in the same way as materials backends.
