# Features module layout

`malchan.features` is the intended home for chemistry and materials feature generation.

The goal is to keep heavy optional dependencies out of generic preprocessing and model code.  Feature generators should be imported only when a workflow explicitly asks for chemistry or materials descriptors.

Target responsibilities:

- `malchan.features.chemistry.smiles`: SMILES parsing and molecule conversion.
- `malchan.features.chemistry.fingerprints`: molecular fingerprints and descriptor backends such as RDKit and scikit-fingerprints.
- `malchan.features.materials.composition`: composition parsing and common composition helpers.
- `malchan.features.materials.xenonpy`: XenonPy-specific composition descriptors.
- `malchan.features.materials.matminer`: Matminer-specific composition descriptors.

Follow-up migration direction:

1. move chemistry-specific transformers out of preprocessing modules;
2. move composition-specific transformers out of preprocessing modules;
3. keep optional backend imports inside the concrete backend modules;
4. expose stable feature builders through `malchan.features` only after their APIs are settled.
