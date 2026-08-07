import assert from "node:assert/strict";

import {
  applyMaterialFeatureTrainingPayload,
  patchMaterialFeatureSettings,
} from "./materialFeatures.js";

const basePayload = {
  data: [
    { a: 0.2, b: 0.3, c: 0.5, temperature: 100, y: 1 },
    { a: 0.4, b: 0.2, c: 0.4, temperature: 200, y: 2 },
  ],
  target_col: "y",
  task: "regression",
  num_cols: ["a", "b", "c", "temperature"],
  cat_cols: [],
  model_names: ["Ridge"],
};

patchMaterialFeatureSettings({
  compositionalEnabled: true,
  compositionalGroups: [["a", "b", "c"]],
  compositionalMethod: "ILR",
  compositionalZeroReplacement: 1e-5,
  compositionalClosure: true,
  compositionalAlrReference: -1,
  compositionalScaleType: "StandardScaler",
});

const payload = applyMaterialFeatureTrainingPayload(basePayload);
assert.deepEqual(payload.compositional_groups, [["a", "b", "c"]]);
assert.equal(payload.compositional_method, "ILR");
assert.equal(payload.compositional_zero_replacement, 1e-5);
assert.equal(payload.compositional_closure, true);
assert.equal(payload.compositional_alr_reference, -1);
assert.equal(payload.compositional_scale_type, "StandardScaler");
assert.deepEqual(payload.num_cols, ["a", "b", "c", "temperature"]);

patchMaterialFeatureSettings({
  compositionalGroups: [["a", "b"], ["b", "c"]],
});
assert.throws(
  () => applyMaterialFeatureTrainingPayload(basePayload),
  /同じ列を複数の組成グループへ指定できません/,
);

patchMaterialFeatureSettings({
  compositionalGroups: [["a"]],
});
assert.throws(
  () => applyMaterialFeatureTrainingPayload(basePayload),
  /2列以上/,
);

patchMaterialFeatureSettings({
  compositionalGroups: [["a", "b"]],
  compositionalMethod: "ALR",
  compositionalAlrReference: 3,
});
assert.throws(
  () => applyMaterialFeatureTrainingPayload(basePayload),
  /ALR基準成分の位置/,
);

patchMaterialFeatureSettings({ compositionalEnabled: false });
const disabledPayload = applyMaterialFeatureTrainingPayload(basePayload);
assert.deepEqual(disabledPayload.compositional_groups, []);
assert.equal(disabledPayload.compositional_method, null);
