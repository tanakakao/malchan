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

export const COMPOSITIONAL_METHODS = ["ILR", "CLR", "ALR"];

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
  compositionalEnabled: false,
  compositionalGroups: [],
  compositionalMethod: "ILR",
  compositionalZeroReplacement: 1e-6,
  compositionalClosure: true,
  compositionalAlrReference: -1,
  compositionalScaleType: "",
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

function normalizeCompositionalGroups(groups) {
  return (groups || []).map((group) => unique(group));
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

export function pruneCompositionalGroups(columns) {
  const allowed = new Set(columns || []);
  const nextGroups = settings.compositionalGroups
    .map((group) => unique(group).filter((column) => allowed.has(column)))
    .filter((group) => group.length > 0);
  if (JSON.stringify(nextGroups) === JSON.stringify(settings.compositionalGroups)) return;
  settings = { ...settings, compositionalGroups: nextGroups };
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
  if (Object.prototype.hasOwnProperty.call(patch, "compositionalGroups")) {
    next.compositionalGroups = normalizeCompositionalGroups(patch.compositionalGroups);
  }
  if (Object.prototype.hasOwnProperty.call(patch, "compositionalZeroReplacement")) {
    next.compositionalZeroReplacement = Number(patch.compositionalZeroReplacement);
  }
  settings = next;
  emit();
}

function compositionalTrainingSettings(numericColumns) {
  if (!settings.compositionalEnabled) {
    return {
      compositional_groups: [],
      compositional_method: null,
      compositional_zero_replacement: null,
      compositional_closure: true,
      compositional_alr_reference: -1,
      compositional_scale_type: null,
    };
  }

  const numericSet = new Set(numericColumns);
  const groups = normalizeCompositionalGroups(settings.compositionalGroups)
    .map((group) => group.filter((column) => numericSet.has(column)));

  if (!groups.length) {
    throw new Error("組成比前処理を使用する場合は、組成グループを1件以上追加してください。");
  }

  const usedColumns = new Set();
  groups.forEach((group, index) => {
    if (group.length < 2) {
      throw new Error(`組成グループ${index + 1}には2列以上を選択してください。`);
    }
    const overlap = group.filter((column) => usedColumns.has(column));
    if (overlap.length) {
      throw new Error(`同じ列を複数の組成グループへ指定できません: ${overlap.join(", ")}`);
    }
    group.forEach((column) => usedColumns.add(column));
  });

  if (!COMPOSITIONAL_METHODS.includes(settings.compositionalMethod)) {
    throw new Error("組成比変換はILR、CLR、ALRから選択してください。");
  }

  const zeroReplacement = Number(settings.compositionalZeroReplacement);
  if (!Number.isFinite(zeroReplacement) || zeroReplacement <= 0) {
    throw new Error("ゼロ置換値は0より大きい数値を指定してください。");
  }

  const alrReference = Number(settings.compositionalAlrReference);
  if (settings.compositionalMethod === "ALR") {
    for (const [index, group] of groups.entries()) {
      if (!Number.isInteger(alrReference) || alrReference < -group.length || alrReference >= group.length) {
        throw new Error(`ALR基準成分の位置が組成グループ${index + 1}の範囲外です。`);
      }
    }
  }

  return {
    compositional_groups: groups,
    compositional_method: settings.compositionalMethod,
    compositional_zero_replacement: zeroReplacement,
    compositional_closure: Boolean(settings.compositionalClosure),
    compositional_alr_reference: settings.compositionalMethod === "ALR" ? alrReference : -1,
    compositional_scale_type: settings.compositionalScaleType || null,
  };
}

export function applyMaterialFeatureTrainingPayload(payload) {
  const numericColumns = unique(payload?.num_cols);
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
    ...compositionalTrainingSettings(numericColumns),
    cat_cols: normalCategoricalColumns,
    smiles_cols: smilesColumns,
    fingerprints: smilesColumns.length ? settings.fingerprints : [],
    comp_cols: compositionColumns,
    comp_method: compositionColumns.length ? settings.compMethod : null,
    comp_feats: compositionColumns.length ? compositionDescriptors : [],
  };
}