import { useSyncExternalStore } from "react";

export const FEATURE_REPRESENTATIONS = [
  ["categorical", "通常カテゴリ"],
  ["composition", "組成式"],
  ["smiles", "分子表記（SMILES）"],
];

export const SMILES_FINGERPRINTS = [
  "ECFP",
  "MACCS",
  "RDKit",
  "PhysChem",
  "AtomPair",
  "Avalon",
  "PubChem",
  "Autocorr",
  "E3FP",
  "MORSE",
  "RDF",
];

export const THREE_DIMENSIONAL_FINGERPRINTS = ["Autocorr", "E3FP", "MORSE", "RDF"];

export const PYMATGEN_PROPERTIES = [
  "atomic_number",
  "atomic_mass",
  "atomic_radius",
  "electronegativity",
  "row",
  "group",
];

export const MATMINER_DESCRIPTORS = [
  "ElementProperty",
  "Stoichiometry",
  "ValenceOrbital",
  "IonProperty",
  "YangSolidSolution",
  "TMetalFraction",
  "Meredig",
  "BandCenter",
  "Miedema",
  "ElementFraction",
];

export const MENDELEEV_PROPERTIES = [
  "atomic_number",
  "atomic_weight",
  "atomic_radius",
  "covalent_radius_cordero",
  "electron_affinity",
  "boiling_point",
  "density",
  "block",
];

const DEFAULT_SETTINGS = {
  kinds: {},
  fingerprints: ["ECFP"],
  compMethod: "pymatgen",
  pymatgenProperties: [...PYMATGEN_PROPERTIES],
  matminerDescriptors: ["ElementProperty", "Stoichiometry"],
  mendeleevProperties: [
    "atomic_number",
    "atomic_weight",
    "atomic_radius",
    "electron_affinity",
    "density",
  ],
};

let settings = DEFAULT_SETTINGS;
const listeners = new Set();

function emit() {
  listeners.forEach((listener) => listener());
}

function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function getSnapshot() {
  return settings;
}

function unique(values) {
  return [...new Set((values || []).filter(Boolean))];
}

export function useMaterialFeatureSettings() {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

export function materialFeatureKind(column) {
  return settings.kinds[column] || "categorical";
}

export function setMaterialFeatureKind(column, kind) {
  if (!column) return;
  if (kind === "categorical") {
    const nextKinds = { ...settings.kinds };
    delete nextKinds[column];
    settings = { ...settings, kinds: nextKinds };
  } else if (kind === "composition" || kind === "smiles") {
    settings = {
      ...settings,
      kinds: {
        ...settings.kinds,
        [column]: kind,
      },
    };
  } else {
    throw new Error(`未対応の材料列種別です: ${kind}`);
  }
  emit();
}

export function pruneMaterialFeatureKinds(columns) {
  const allowed = new Set(columns || []);
  const nextKinds = Object.fromEntries(
    Object.entries(settings.kinds).filter(([column]) => allowed.has(column)),
  );
  if (JSON.stringify(nextKinds) === JSON.stringify(settings.kinds)) return;
  settings = { ...settings, kinds: nextKinds };
  emit();
}

export function patchMaterialFeatureSettings(patch) {
  const next = { ...settings, ...patch };
  if (Object.prototype.hasOwnProperty.call(patch, "fingerprints")) {
    next.fingerprints = unique(patch.fingerprints);
  }
  if (Object.prototype.hasOwnProperty.call(patch, "pymatgenProperties")) {
    next.pymatgenProperties = unique(patch.pymatgenProperties);
  }
  if (Object.prototype.hasOwnProperty.call(patch, "matminerDescriptors")) {
    next.matminerDescriptors = unique(patch.matminerDescriptors);
  }
  if (Object.prototype.hasOwnProperty.call(patch, "mendeleevProperties")) {
    next.mendeleevProperties = unique(patch.mendeleevProperties);
  }
  settings = next;
  emit();
}

export function applyMaterialFeatureTrainingPayload(payload) {
  const categoricalColumns = unique(payload?.cat_cols);
  const smilesColumns = categoricalColumns.filter(
    (column) => settings.kinds[column] === "smiles",
  );
  const compositionColumns = categoricalColumns.filter(
    (column) => settings.kinds[column] === "composition",
  );
  const materialColumns = new Set([...smilesColumns, ...compositionColumns]);
  const normalCategoricalColumns = categoricalColumns.filter(
    (column) => !materialColumns.has(column),
  );

  if (smilesColumns.length && !settings.fingerprints.length) {
    throw new Error("SMILES列に使用する分子記述子を1件以上選択してください。");
  }

  let compositionDescriptors = [];
  if (compositionColumns.length) {
    if (settings.compMethod === "pymatgen") {
      compositionDescriptors = settings.pymatgenProperties;
    } else if (settings.compMethod === "mendeleev") {
      compositionDescriptors = settings.mendeleevProperties;
    } else {
      compositionDescriptors = settings.matminerDescriptors;
    }
    if (!compositionDescriptors.length) {
      throw new Error("組成式列に使用する記述子を1件以上選択してください。");
    }
  }

  return {
    ...payload,
    cat_cols: normalCategoricalColumns,
    smiles_cols: smilesColumns,
    fingerprints: smilesColumns.length ? settings.fingerprints : [],
    comp_cols: compositionColumns,
    comp_method: compositionColumns.length ? settings.compMethod : null,
    comp_feats: compositionColumns.length ? compositionDescriptors : [],
  };
}
