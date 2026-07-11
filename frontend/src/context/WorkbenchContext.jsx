import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import {
  CLASSIFICATION_MODELS,
  REGRESSION_MODELS,
  coerceRows,
  columnStats,
  numericSummary,
  parseTabularFile,
  uniqueValues,
} from "../data";

const WorkbenchContext = createContext(null);

export const STEPS = [
  ["data", "Data", "読込・確認"],
  ["explore", "Explore", "統計・可視化"],
  ["prepare", "Prepare", "変数・前処理"],
  ["model", "Model", "学習・比較"],
  ["explain", "Explain", "精度・挙動"],
  ["optimize", "Optimize", "予測・逆解析"],
  ["report", "Report", "レポート"],
];

const REGRESSION_DEFAULTS = ["線形回帰", "Ridge", "ランダムフォレスト回帰", "LightGBM"];
const CLASSIFICATION_DEFAULTS = ["ロジスティック回帰", "ランダムフォレスト", "LightGBM", "サポートベクターマシン"];

export const modelsFor = (task) =>
  task === "classification" ? CLASSIFICATION_MODELS : REGRESSION_MODELS;

const defaultModel = (task) =>
  task === "classification" ? "ランダムフォレスト" : "ランダムフォレスト回帰";

const defaultCandidates = (task) =>
  task === "classification" ? CLASSIFICATION_DEFAULTS : REGRESSION_DEFAULTS;

function isIntegerColumn(rows, column) {
  const values = rows.map((row) => row[column]).filter(Number.isFinite);
  return values.length > 0 && values.every(Number.isInteger);
}

export function WorkbenchProvider({ children }) {
  const [theme, setTheme] = useState(localStorage.getItem("malchan-theme") || "dark");
  const [step, setStep] = useState("data");
  const [health, setHealth] = useState({ status: "loading", text: "FastAPIを確認しています..." });
  const [busy, setBusy] = useState("");
  const [toast, setToast] = useState(null);
  const [fileName, setFileName] = useState("");
  const [rows, setRows] = useState([]);
  const [columns, setColumns] = useState([]);
  const [numeric, setNumeric] = useState([]);
  const [categorical, setCategorical] = useState([]);
  const [targets, setTargets] = useState([]);
  const [tasks, setTasks] = useState({});
  const [numFeatures, setNumFeatures] = useState([]);
  const [catFeatures, setCatFeatures] = useState([]);
  const [modelNames, setModelNames] = useState({});
  const [candidates, setCandidates] = useState({});
  const [modelInfo, setModelInfo] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [cvSplits, setCvSplits] = useState(5);
  const [tuneBest, setTuneBest] = useState(true);
  const [activateBest, setActivateBest] = useState(true);
  const [trials, setTrials] = useState(50);
  const [chartType, setChartType] = useState("scatter");
  const [chartX, setChartX] = useState("");
  const [chartY, setChartY] = useState("");
  const [predictValues, setPredictValues] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [diagnostics, setDiagnostics] = useState([]);
  const [objectives, setObjectives] = useState({});
  const [bounds, setBounds] = useState({});
  const [inverseResult, setInverseResult] = useState(null);
  const [sampler, setSampler] = useState("TPE");
  const [inverseTrials, setInverseTrials] = useState(250);
  const [topK, setTopK] = useState(10);
  const [reportProblem, setReportProblem] = useState("");
  const [report, setReport] = useState("");

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("malchan-theme", theme);
  }, [theme]);

  useEffect(() => {
    api.health()
      .then((response) => setHealth({ status: "ready", text: `${response.service} ${response.version}` }))
      .catch((error) => setHealth({ status: "error", text: error.message }));
  }, []);

  const features = useMemo(
    () => [...numFeatures, ...catFeatures].filter((column) => !targets.includes(column)),
    [numFeatures, catFeatures, targets],
  );
  const stats = useMemo(() => columnStats(rows, columns), [rows, columns]);
  const missing = useMemo(
    () => stats.reduce((sum, item) => sum + item.missing, 0),
    [stats],
  );
  const ready = rows.length > 1 && targets.length > 0 && features.length > 0;

  function notify(text, type = "success") {
    setToast({ text, type });
    window.setTimeout(() => setToast(null), 4500);
  }

  async function run(label, operation) {
    setBusy(label);
    try {
      return await operation();
    } catch (error) {
      notify(error.message || String(error), "error");
      return null;
    } finally {
      setBusy("");
    }
  }

  async function loadFile(file) {
    if (!file) return;
    const parsed = await run("データを読み込んでいます...", () => parseTabularFile(file));
    if (!parsed) return;

    const data = coerceRows(parsed.rows);
    const target = data.columns.at(-1);
    const task = data.numericColumns.includes(target) ? "regression" : "classification";
    const featureColumns = data.columns.filter((column) => column !== target);

    setFileName(file.name);
    setRows(data.rows);
    setColumns(data.columns);
    setNumeric(data.numericColumns);
    setCategorical(data.categoricalColumns);
    setTargets([target]);
    setTasks({ [target]: task });
    setNumFeatures(data.numericColumns.filter((column) => column !== target));
    setCatFeatures(data.categoricalColumns.filter((column) => column !== target));
    setModelNames({ [target]: defaultModel(task) });
    setCandidates({ [target]: defaultCandidates(task) });
    setChartX(data.numericColumns.find((column) => column !== target) || data.numericColumns[0] || "");
    setChartY(target || "");
    setPredictValues(
      Object.fromEntries(featureColumns.map((column) => [column, data.rows[0]?.[column] ?? ""])),
    );
    setObjectives({
      [target]: task === "classification"
        ? { mode: "target", value: uniqueValues(data.rows, target)[0] ?? "" }
        : { mode: "direction", value: "max" },
    });
    setBounds(
      Object.fromEntries(
        data.numericColumns
          .filter((column) => column !== target)
          .map((column) => [column, numericSummary(data.rows, column)]),
      ),
    );
    setModelInfo(null);
    setComparison(null);
    setPrediction(null);
    setDiagnostics([]);
    setInverseResult(null);
    notify(`${data.rows.length}行 × ${data.columns.length}列を読み込みました。`);
  }

  function changeTargets(nextTargets) {
    const cleanTargets = columns.filter((column) => nextTargets.includes(column));
    const nextTasks = { ...tasks };
    const nextModels = { ...modelNames };
    const nextCandidates = { ...candidates };
    const nextObjectives = { ...objectives };

    cleanTargets.forEach((target) => {
      const task = nextTasks[target] || (numeric.includes(target) ? "regression" : "classification");
      nextTasks[target] = task;
      nextModels[target] ||= defaultModel(task);
      nextCandidates[target] ||= defaultCandidates(task);
      nextObjectives[target] ||= task === "classification"
        ? { mode: "target", value: uniqueValues(rows, target)[0] ?? "" }
        : { mode: "direction", value: "max" };
    });

    setTargets(cleanTargets);
    setTasks(nextTasks);
    setModelNames(nextModels);
    setCandidates(nextCandidates);
    setObjectives(nextObjectives);
    setNumFeatures((values) => values.filter((column) => !cleanTargets.includes(column)));
    setCatFeatures((values) => values.filter((column) => !cleanTargets.includes(column)));
  }

  function changeTask(target, task) {
    setTasks({ ...tasks, [target]: task });
    setModelNames({ ...modelNames, [target]: defaultModel(task) });
    setCandidates({ ...candidates, [target]: defaultCandidates(task) });
    setObjectives({
      ...objectives,
      [target]: task === "classification"
        ? { mode: "target", value: uniqueValues(rows, target)[0] ?? "" }
        : { mode: "direction", value: "max" },
    });
  }

  function trainingPayload() {
    const common = {
      data: rows,
      num_cols: numFeatures.filter((column) => features.includes(column)),
      cat_cols: catFeatures.filter((column) => features.includes(column)),
    };
    if (targets.length === 1) {
      const target = targets[0];
      return {
        ...common,
        target_col: target,
        task: tasks[target],
        model_names: [modelNames[target]],
      };
    }
    return {
      ...common,
      target_cols: targets,
      tasks: targets.map((target) => tasks[target]),
      model_names_by_target: Object.fromEntries(
        targets.map((target) => [target, [modelNames[target]]]),
      ),
    };
  }

  async function trainModel() {
    const response = await run("モデルを学習しています...", () => api.train(trainingPayload()));
    if (!response) return;
    setModelInfo(response);
    setComparison(null);
    notify(`モデル ${response.model_id} を登録しました。`);
  }

  async function compareModels() {
    if (!modelInfo) return notify("先にモデルを学習してください。", "error");
    const multiOutput = targets.length > 1;
    const payload = {
      model_names: multiOutput
        ? Object.fromEntries(targets.map((target) => [target, candidates[target] || []]))
        : candidates[targets[0]] || [],
      method: "kfold",
      n_splits: Number(cvSplits),
      metric: multiOutput
        ? Object.fromEntries(
            targets.map((target) => [target, tasks[target] === "classification" ? "F1" : "RMSE"]),
          )
        : tasks[targets[0]] === "classification" ? "F1" : "RMSE",
      tune_best: tuneBest,
      tuning_trials: multiOutput
        ? Object.fromEntries(targets.map((target) => [target, Number(trials)]))
        : Number(trials),
      activate_best: activateBest,
    };
    const response = await run("候補モデルを比較しています...", () =>
      api.compare(modelInfo.model_id, payload),
    );
    if (!response) return;
    setComparison(response);
    if (activateBest) setModelInfo(await api.modelInfo(modelInfo.model_id));
    notify("モデル比較が完了しました。");
  }

  async function tuneBestLater() {
    if (!comparison) return notify("先に比較してください。", "error");
    const payload = targets.length > 1
      ? {
          n_trials: Object.fromEntries(targets.map((target) => [target, Number(trials)])),
          evaluate: true,
          activate_best: activateBest,
        }
      : { n_trials: Number(trials), evaluate: true, activate_best: activateBest };
    const response = await run("ベストモデルをチューニングしています...", () =>
      api.tuneBest(modelInfo.model_id, payload),
    );
    if (!response) return;
    setComparison(response);
    if (activateBest) setModelInfo(await api.modelInfo(modelInfo.model_id));
    notify("チューニングが完了しました。");
  }

  async function predictOne() {
    if (!modelInfo) return notify("先にモデルを学習してください。", "error");
    const input = Object.fromEntries(
      features.map((column) => [
        column,
        numeric.includes(column) ? Number(predictValues[column]) : predictValues[column],
      ]),
    );
    const response = await run("予測しています...", () =>
      api.predict(modelInfo.model_id, { data: [input] }),
    );
    if (response) setPrediction(response.predictions[0]);
  }

  async function updateDiagnostics() {
    if (!modelInfo) return notify("先にモデルを学習してください。", "error");
    const data = rows.map((row) =>
      Object.fromEntries(features.map((column) => [column, row[column]])),
    );
    const response = await run("診断用予測を計算しています...", () =>
      api.predict(modelInfo.model_id, { data }),
    );
    if (response) {
      setDiagnostics(
        rows.map((actual, index) => ({ actual, predicted: response.predictions[index] || {} })),
      );
    }
  }

  async function runInverseAnalysis() {
    if (!modelInfo) return notify("先にモデルを学習してください。", "error");
    const payload = {
      objectives: targets.map((target) => {
        const objective = objectives[target] || {};
        if (objective.mode === "target") {
          return {
            target,
            target_value: tasks[target] === "regression"
              ? Number(objective.value)
              : objective.value,
          };
        }
        return { target, direction: objective.value || "max" };
      }),
      sampler_type: sampler,
      bounds: Object.fromEntries(
        numFeatures.map((column) => [
          column,
          {
            min: Number(bounds[column]?.min ?? 0),
            max: Number(bounds[column]?.max ?? 1),
            dtype: isIntegerColumn(rows, column) ? "int" : "float",
            ...(isIntegerColumn(rows, column) ? { step: 1 } : {}),
          },
        ]),
      ),
      categories: Object.fromEntries(
        catFeatures.map((column) => [column, uniqueValues(rows, column)]),
      ),
      trials: Number(inverseTrials),
      n_candidates: Number(topK),
    };
    const response = await run("逆解析を実行しています...", () =>
      api.inverse(modelInfo.model_id, payload),
    );
    if (!response) return;
    setInverseResult(response);
    notify(`${response.candidates.length}件の候補を取得しました。`);
  }

  function makeReportPrompt() {
    setReport(`あなたは材料・製造データ分析の専門家です。以下の情報から、根拠と限界を明確にした分析レポートを日本語で作成してください。

【課題】
${reportProblem || "未入力"}

【データ】
ファイル: ${fileName}
行数: ${rows.length}
説明変数: ${features.join(", ")}
目的変数: ${targets.map((target) => `${target}(${tasks[target]})`).join(", ")}
欠損セル: ${missing}

【モデル比較】
${comparison
  ? Object.entries(comparison.targets)
      .map(([target, result]) => `${target}: ${result.best_model_name}, ${result.metric}, tuned=${result.best_is_tuned}`)
      .join("\n")
  : "未実施"}

【逆解析】
${inverseResult
  ? `候補数=${inverseResult.candidates.length}, Pareto=${inverseResult.pareto_size}`
  : "未実施"}

目的、データ概要、精度、モデル選定理由、変数関係、逆解析候補、限界、次の実験提案の順にまとめてください。`);
  }

  const value = {
    theme, setTheme, step, setStep, health, busy, toast, setToast,
    fileName, rows, columns, numeric, categorical, targets, tasks,
    numFeatures, setNumFeatures, catFeatures, setCatFeatures,
    modelNames, setModelNames, candidates, setCandidates,
    modelInfo, comparison, cvSplits, setCvSplits, tuneBest, setTuneBest,
    activateBest, setActivateBest, trials, setTrials,
    chartType, setChartType, chartX, setChartX, chartY, setChartY,
    predictValues, setPredictValues, prediction, diagnostics,
    objectives, setObjectives, bounds, setBounds, inverseResult,
    sampler, setSampler, inverseTrials, setInverseTrials, topK, setTopK,
    reportProblem, setReportProblem, report, setReport,
    features, stats, missing, ready,
    loadFile, changeTargets, changeTask, trainModel, compareModels,
    tuneBestLater, predictOne, updateDiagnostics, runInverseAnalysis,
    makeReportPrompt,
  };

  return <WorkbenchContext.Provider value={value}>{children}</WorkbenchContext.Provider>;
}

export function useWorkbench() {
  const context = useContext(WorkbenchContext);
  if (!context) throw new Error("useWorkbench must be used inside WorkbenchProvider.");
  return context;
}
