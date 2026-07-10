import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

import io
import os
import sys

import warnings
warnings.simplefilter('ignore')

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "images")

def save_yy_plot_mpl(model, target):
    """
    regression: actual vs pred + residual
    classification: confusion matrix (cvなし:1枚, cvあり:train/testで2枚)
    """
    task = getattr(model.models[target], "task", "regression")

    # ---- font sizes（少し大きめ）----
    title_fs = 14
    label_fs = 13
    tick_fs = 12
    cell_fs = 12
    legend_fs = 12

    # --- true / pred ---
    y_true = np.asarray(model.models[target].y.values).ravel()
    y_pred = np.asarray(model.predict()[target].values)

    # --- cv ---
    cv_preds = getattr(model.models[target], "cv_preds", None)
    cv = cv_preds is not None
    if cv:
        cv_train_pred = np.asarray(cv_preds["train"].values)
        cv_test_pred = np.asarray(cv_preds["test"].values)

    # --- label mapping (e.g., {1:'x',0:'y'}) ---
    idx2item = getattr(model.models[target], "idx2item", None) or {}

    # =========================
    # regression
    # =========================
    if task == "regression":
        y_pred_1d = y_pred.ravel()
        if cv:
            cv_train_1d = cv_train_pred.ravel()
            cv_test_1d = cv_test_pred.ravel()

        ymin = float(np.nanmin(y_true))
        ymax = float(np.nanmax(y_true))

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        # act vs pred
        if cv:
            ax[0].scatter(y_true, cv_train_1d, color="blue", alpha=0.75, label="train")
            ax[0].scatter(y_true, cv_test_1d, color="orange", alpha=0.75, label="test")
        else:
            ax[0].scatter(y_true, y_pred_1d, color="blue", alpha=0.75)
        ax[0].plot([ymin, ymax], [ymin, ymax], color="black")
        ax[0].set_xlabel("actual", fontsize=label_fs)
        ax[0].set_ylabel("predict", fontsize=label_fs)
        ax[0].set_title("act. vs pred.", fontsize=title_fs)
        ax[0].tick_params(axis="both", labelsize=tick_fs)

        # residual
        if cv:
            ax[1].scatter(y_true, cv_train_1d - y_true, color="blue", alpha=0.75, label="train")
            ax[1].scatter(y_true, cv_test_1d - y_true, color="orange", alpha=0.75, label="test")
        else:
            ax[1].scatter(y_true, y_pred_1d - y_true, color="blue", alpha=0.75)
        ax[1].plot([ymin, ymax], [0, 0], color="black")
        ax[1].set_xlabel("actual", fontsize=label_fs)
        ax[1].set_ylabel("residual", fontsize=label_fs)
        ax[1].set_title("act. vs res.", fontsize=title_fs)
        ax[1].tick_params(axis="both", labelsize=tick_fs)

        if cv:
            ax[0].legend(fontsize=legend_fs)
            ax[1].legend(fontsize=legend_fs)

        plt.tight_layout()

    else:
        # =========================
        # classification -> confusion matrix
        # =========================
        def _to_class_labels(y_hat, y_ref):
            """
            y_hat をクラスラベルに変換（確率/スコアっぽい場合もある程度吸収）
            """
            y_hat = np.asarray(y_hat)
            y_ref = np.asarray(y_ref).ravel()
            labels_ref = np.unique(y_ref)
    
            if y_hat.ndim == 2 and y_hat.shape[1] > 1:
                return np.argmax(y_hat, axis=1)
    
            y_hat = y_hat.ravel()
    
            # 2値確率っぽい (0..1)
            if labels_ref.size == 2 and np.issubdtype(y_hat.dtype, np.floating):
                y_min = np.nanmin(y_hat)
                y_max = np.nanmax(y_hat)
                if y_min >= 0.0 and y_max <= 1.0:
                    return np.where(y_hat >= 0.5, labels_ref[1], labels_ref[0])
    
            # 数値ラベル + float -> 最近傍ラベル
            if np.issubdtype(labels_ref.dtype, np.number) and np.issubdtype(y_hat.dtype, np.floating):
                labels_num = labels_ref.astype(float)
                idx = np.argmin(np.abs(y_hat[:, None] - labels_num[None, :]), axis=1)
                return labels_ref[idx]
    
            return y_hat
    
        def _label_to_name(lbl):
            """idx2item があればそれで表示名に置換"""
            if not idx2item:
                return str(lbl)
            try:
                return str(idx2item.get(int(lbl), lbl))
            except Exception:
                return str(lbl)
    
        def _plot_cm(ax, y_t, y_p, title):
            y_t = np.asarray(y_t).ravel()
            y_p = np.asarray(y_p).ravel()
    
            labels_raw = np.unique(np.concatenate([y_t, y_p]))
            display_labels = [_label_to_name(l) for l in labels_raw]
    
            cm = confusion_matrix(y_t, y_p, labels=labels_raw)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            disp.plot(ax=ax, colorbar=False, values_format="d", cmap="Reds")
    
            # フォント調整（タイトル/軸/目盛/セル内）
            acc = float(np.mean(y_p == y_t))
            ax.set_title(f"{title}\nacc={acc:.3f}", fontsize=title_fs)
            ax.set_xlabel("Predicted label", fontsize=label_fs)
            ax.set_ylabel("True label", fontsize=label_fs)
            ax.tick_params(axis="both", labelsize=tick_fs)
    
            if getattr(disp, "text_", None) is not None:
                for t in np.asarray(disp.text_).ravel():
                    t.set_fontsize(cell_fs)
    
        if cv:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            y_pred_train_lbl = _to_class_labels(cv_train_pred, y_true)
            y_pred_test_lbl = _to_class_labels(cv_test_pred, y_true)
            _plot_cm(axes[0], y_true, y_pred_train_lbl, "train")
            _plot_cm(axes[1], y_true, y_pred_test_lbl, "test")
            plt.tight_layout()
    
        else:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            y_pred_lbl = _to_class_labels(y_pred, y_true)
            _plot_cm(ax, y_true, y_pred_lbl, "confusion matrix")
            plt.tight_layout()

    plot_image_path = f"{IMAGE_PATH}/yyplot_image_{target}.png"
    fig.savefig(plot_image_path)
    plt.close(fig)  # グラフを閉じてメモリを解放

def importance_array(model, target):
    importances = model.models[target].importances
    importance1 = importances["model"]
    feature_names = model.feature_names[target]
    if importance1 is not None:
        if len(importance1)>0:
            importance1 = importance1.astype(float)
            importance1 /= importance1.max()
        else:
            importance1 = np.zeros(len(feature_names))
    else:
        importance1 = np.zeros(len(feature_names))

    importance2 = importances["shap"]
    if importance2 is not None:
        importance2 = importance2
        importance2 /= importance2.max()
    else:
        importance2=np.zeros(len(feature_names))
        
    importance3 = importances["pfi"]
    if (len(importance3)>0)and all(importance3):
        importance3 /= importance3.max()
    else:
        importance3 = np.zeros(len(feature_names))

    importance = np.concatenate([
        importance1[np.newaxis,:],
        importance2[np.newaxis,:],
        importance3[np.newaxis,:]
    ],axis=0)
    return importance

def save_importances_plot_mpl(model, target, max_cols=10):
    importance = importance_array(model, target)
    sort_idx = np.argsort(np.abs(importance).mean(axis=0))
    features = model.models[target].df_prerpocessed.columns
    features_list = list(features[sort_idx])
    
    K = len(features_list)
    ncols = min(max_cols, K)
    K = ncols
    
    features_list = features_list[-ncols:]
    nrows = int(np.ceil(K / ncols))

    sort_idx = sort_idx[-ncols:]
    height = 0.4
    y = np.arange(len(features))[-ncols:]
    
    fig, ax = plt.subplots(1,1,figsize=(5, 6*nrows), sharey=True)
    ax.barh(y=y+(height/2), width=importance[0][sort_idx], height=height/2, label='model')
    ax.barh(y=y , width=importance[1][sort_idx], height=height/2, tick_label=features[sort_idx], label='shap')
    ax.barh(y=y-(height/2) , width=importance[2][sort_idx], height=height/2, label='pfi')
    ax.set_xlabel('importance')
    ax.set_ylabel('feature name')
    ax.legend(loc='lower right');
    ax.set_title('feature importance')
    
    plt.tight_layout();
    
    plot_image_path = f"{IMAGE_PATH}/importances_{target}.png"
    fig.savefig(plot_image_path)
    plt.close(fig)  # グラフを閉じてメモリを解放

def _add_jitter(x, scale=0.12, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    return x + rng.normal(0.0, scale, size=x.shape)

def _shap_to_class_arrays(shap_values, n_samples, n_features):
    """shap_values -> (K, n, f) に正規化（回帰は K=1）"""
    if shap_values is None:
        return np.zeros((1, n_samples, n_features), dtype=float)

    if hasattr(shap_values, "values"):  # shap.Explanation 等
        shap_values = shap_values.values

    if isinstance(shap_values, (list, tuple)):
        arrs = [np.asarray(a) for a in shap_values if a is not None]
        if len(arrs) == 0:
            return np.zeros((1, n_samples, n_features), dtype=float)
        norm = []
        for a in arrs:
            a = np.asarray(a).squeeze()
            if a.ndim != 2:
                a = a.reshape(n_samples, n_features)
            norm.append(a)
        return np.stack(norm, axis=0)  # (K,n,f)

    a = np.asarray(shap_values)
    if a.ndim == 2:
        if a.shape != (n_samples, n_features):
            a = a.reshape(n_samples, n_features)
        return a[None, :, :]

    if a.ndim == 3:
        # (n,f,k)
        if a.shape[0] == n_samples and a.shape[1] == n_features:
            return np.transpose(a, (2, 0, 1))  # (k,n,f)
        # (k,n,f)
        if a.shape[1] == n_samples and a.shape[2] == n_features:
            return a
        # (n,k,f)
        if a.shape[0] == n_samples and a.shape[2] == n_features:
            return np.transpose(a, (1, 0, 2))  # (k,n,f)
        # fallback
        k = a.shape[0]
        return a.reshape(k, n_samples, n_features)

    return np.zeros((1, n_samples, n_features), dtype=float)

def df_shap_beeswarm_panels(model, target, *, jitter_seed=0):
    """
    回帰/分類共通：
      - 回帰: K=1
      - 多クラス分類: K=クラス数
      - 2クラスで shap が1つだけ: K=1
    さらに、特徴量は mean(|SHAP|) 降順に並べ替える（全パネル共通）。
    """
    df_pre = model.models[target].df_prerpocessed
    task = getattr(model.models[target], "task", "regression")
    idx2item = getattr(model.models[target], "idx2item", None) or {}

    # 可能ならこちらの順序/名称を優先（元コード互換）
    feature_names = list(model.feature_names[target]) if hasattr(model, "feature_names") else []
    cols = list(df_pre.columns)

    # SHAP
    n_samples, n_features = df_pre.shape
    shap_values = getattr(model.models[target], "shap_values", None)
    shap_k = _shap_to_class_arrays(shap_values, n_samples, n_features)  # (K,n,f)
    K = shap_k.shape[0]

    # --- パネル名 ---
    note = ""
    if task == "regression":
        panel_names = ["regression"]
    else:
        if K > 1:
            panel_names = [str(idx2item.get(i, i)) for i in range(K)]
        else:
            if (0 in idx2item) or (1 in idx2item):
                panel_names = [str(idx2item.get(1, idx2item.get(0, "class")))]
                note = " (only one SHAP array)"
            else:
                panel_names = ["class"]

    # --- 特徴量リスト（feature_namesがdf_pre.columnsに全部含まれるならそれを使う）---
    if feature_names and all(f in cols for f in feature_names):
        base_features = feature_names
    else:
        base_features = cols

    # --- 重要度で並べ替え（全パネル共通）---
    # base_features に対応する列index
    col_idx = [cols.index(f) for f in base_features]
    imp = np.mean(np.abs(shap_k[:, :, col_idx]), axis=(0, 1))  # (F,)
    order = np.argsort(-imp)  # 降順
    ordered_features = [base_features[i] for i in order]

    # --- long dataframe を作成（ordered_features順に feature index を 0..F-1 にする）---
    df_list_all = []
    for k in range(K):
        shap_nf = shap_k[k]  # (n,f)

        rows = []
        for rank, feature in enumerate(ordered_features):
            j = cols.index(feature)  # df_pre / shapの列位置
            col = df_pre[feature].to_numpy()

            cmin, cmax = np.nanmin(col), np.nanmax(col)
            if np.isfinite(cmin) and np.isfinite(cmax) and (cmax - cmin) != 0:
                stdv = (col - cmin) / (cmax - cmin) - 0.5
            else:
                stdv = np.zeros_like(col, dtype=float)

            rows.append(pd.DataFrame({
                "shap value": shap_nf[:, j],
                "feature name": feature,
                "standardize value": stdv,
                "feature index": rank,
            }))

        df_k = pd.concat(rows, axis=0, ignore_index=True)
        df_k["feature index"] = _add_jitter(df_k["feature index"].to_numpy(), seed=jitter_seed + k)
        df_k["panel_index"] = k
        df_k["panel_name"] = panel_names[k] if k < len(panel_names) else str(k)
        df_list_all.append(df_k)

    df_all = pd.concat(df_list_all, axis=0, ignore_index=True)
    return df_all, panel_names, note, ordered_features, task

def plot_shap_beeswarm_mpl(model, target, *, max_cols=10):
    """回帰/分類共通 Beeswarm（カラーバー無し / 特徴量はmean(|SHAP|)降順）"""
    df_all, panel_names, note, ordered_features, task = df_shap_beeswarm_panels(model, target)

    ordered_features = ordered_features[:max_cols]
    K = len(panel_names)
    ncols = min(max_cols, K)
    nrows = int(np.ceil(K / ncols))

    title_fs = 14
    label_fs = 12
    tick_fs = 11

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

    for k in range(K):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        d = df_all[df_all["panel_index"] == k]
        d = d[d["feature name"].apply(lambda x: x in ordered_features)]

        ax.scatter(
            d["shap value"].to_numpy(),
            d["feature index"].to_numpy(),
            c=d["standardize value"].to_numpy(),
            cmap="coolwarm",
            vmin=-0.5, vmax=0.5,
            s=10, alpha=0.75,
            linewidths=0,
        )
        ax.axvline(0.0, color="black", linewidth=1)

        if task == "regression":
            ax.set_title(f"SHAP beeswarm (regression): {target}", fontsize=title_fs)
        else:
            ax.set_title(f"SHAP beeswarm: {panel_names[k]}{note if K==1 else ''}", fontsize=title_fs)

        ax.set_xlabel("SHAP value", fontsize=label_fs)
        ax.set_ylabel("Feature", fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)

        yticks = np.arange(len(ordered_features))
        ax.set_yticks(yticks)
        ax.set_yticklabels(ordered_features, fontsize=tick_fs)
        ax.invert_yaxis()

    for kk in range(K, nrows * ncols):
        r, c = divmod(kk, ncols)
        axes[r][c].axis("off")

    plt.tight_layout()

    plot_image_path = f"{IMAGE_PATH}/beeswarm_{target}.png"
    fig.savefig(plot_image_path)
    plt.close(fig)  # グラフを閉じてメモリを解放

# -----------------------------
# small helpers
# -----------------------------
def _to_numpy(x):
    """shap.Explanation / pandas / list などを numpy へ寄せる。None は None のまま。"""
    if x is None:
        return None
    if hasattr(x, "values"):  # shap.Explanation / pandas
        x = x.values
    return np.asarray(x)


def _stack_shap_list_per_class(shap_list):
    """
    shap が [class0_arr, class1_arr, ...] の形式で返ってくる場合を (n,f,k) に統一。
    各要素は (n,f) または (n,...) を想定し、f 側は flatten する。
    """
    arrs = [_to_numpy(a) for a in shap_list if a is not None]
    if len(arrs) == 0:
        return None

    # 各 class: (n, f_like) に整形
    arrs2 = [a.reshape(a.shape[0], -1) if a.ndim >= 2 else a.reshape(-1, 1) for a in arrs]
    # (k,n,f) -> (n,f,k)
    shap_knf = np.stack(arrs2, axis=0)
    shap_nfk = np.transpose(shap_knf, (1, 2, 0))
    return shap_nfk


def _reduce_importance_to_1d(importance):
    """
    importance が (f,) / (k,f) / (n,f) 等でも、features 次元(f)の 1D に潰す。
    """
    imp = _to_numpy(importance)
    if imp is None:
        return None
    imp = np.abs(imp)
    if imp.ndim == 1:
        return imp
    # 末尾が features である想定で平均（一般に pfi_combine が (k,f) 等を取りうるため）
    return imp.mean(axis=0)


def _infer_shap_to_nfk(shap_arr, base_arr=None, *, positive_class_index=1):
    """
    shap_arr を (n,f,k) へ寄せる。
    - shap_arr.ndim==2 -> (n,f,1)
    - shap_arr.ndim==3 -> base_arr を手掛かりに (n,f,k) へ転置
    """
    info = {}

    if shap_arr is None:
        return None, info

    if shap_arr.ndim == 2:
        # (n,f) -> (n,f,1)
        info["shap_layout"] = "nf"
        return shap_arr[:, :, None], info

    if shap_arr.ndim != 3:
        info["shap_layout"] = f"unsupported_ndim={shap_arr.ndim}"
        return shap_arr, info  # そのまま（呼び元で扱う）

    s0, s1, s2 = shap_arr.shape

    # base が (n,k) ならそれに合わせる
    if base_arr is not None and base_arr.ndim == 2:
        n, k = base_arr.shape

        # (n,f,k)
        if s0 == n and s2 == k:
            info["shap_layout"] = "n_f_k"
            return shap_arr, info

        # (n,k,f) -> (n,f,k)
        if s0 == n and s1 == k:
            info["shap_layout"] = "n_k_f"
            return np.transpose(shap_arr, (0, 2, 1)), info

        # (k,n,f) -> (n,f,k)
        if s0 == k and s1 == n:
            info["shap_layout"] = "k_n_f"
            return np.transpose(shap_arr, (1, 2, 0)), info

    # fallback heuristic: 既に (n,f,k) 扱い
    info["shap_layout"] = "3d_fallback_assume_n_f_k"
    return shap_arr, info


def _normalize_base_to_n1k(base_arr, *, n, k, shap_is_single_class=False, positive_class_index=1):
    """
    base_arr を (n,1,k) にする（broadcast 可能な形）。
    shap が単一クラス (k=1相当) のとき、base が (n,k_all) 等ならクラスを pick して k=1 に落とす。
    """
    info = {}

    if base_arr is None:
        return None, {"base_mode": "none", **info}

    # scalar
    if base_arr.ndim == 0:
        base_exp = float(base_arr)  # scalar broadcast
        return base_exp, {"base_mode": "scalar", **info}

    # (n,) / (k,)
    if base_arr.ndim == 1:
        if base_arr.shape[0] == n:
            # sample-wise base
            base_exp = base_arr.reshape(n, 1, 1)
            return base_exp, {"base_mode": "per_sample_(n,)", **info}

        if base_arr.shape[0] == k:
            # class base
            base_exp = base_arr.reshape(1, 1, k)
            return base_exp, {"base_mode": "per_class_(k,)", **info}

        # fallback
        base_exp = base_arr.reshape(1, 1, -1)
        return base_exp, {"base_mode": "fallback_1d", "note": f"base.shape={base_arr.shape}", **info}

    # (n,k) が典型
    if base_arr.ndim == 2:
        if base_arr.shape[0] == n:
            if shap_is_single_class and base_arr.shape[1] > 1:
                # shap は (n,f) なのに base が (n,k) の場合: クラスを 1 つ選んで k=1 へ
                k_all = base_arr.shape[1]
                cls = positive_class_index if (k_all > positive_class_index) else 0
                picked = base_arr[:, cls].reshape(n, 1, 1)
                return picked, {
                    "base_mode": "picked_one_class_from_(n,k)",
                    "picked_class": cls,
                    "note": "shap was single-class; base (n,k) -> picked one class",
                    **info,
                }

            # (n,k) -> (n,1,k)
            base_exp = base_arr[:, None, :]
            return base_exp, {"base_mode": "(n,k)->(n,1,k)", **info}

        # fallback
        base_exp = base_arr.reshape(1, 1, base_arr.shape[-1])
        return base_exp, {"base_mode": "fallback_2d", "note": f"base.shape={base_arr.shape}", **info}

    # すでに (n,1,k) 等ならそのまま（雑に）
    return base_arr, {"base_mode": f"fallback_ndim={base_arr.ndim}", "note": f"base.shape={base_arr.shape}", **info}


def _normalize_pd_curve(pd_raw, ticks, *, n_classes=1):
    """
    PD の生配列を
      - 回帰: (len_ticks,)
      - 分類: (len_ticks, k)
    に正規化する。

    想定:
      pd_raw が (len_ticks, n_boot, k) / (n_boot, len_ticks, k) / (len_ticks, k) 等、いろいろ混ざりうる。
    """
    if pd_raw is None:
        return None

    arr = _to_numpy(pd_raw)
    tick_len = len(ticks)

    # ticks 軸を 0 に移動
    tick_axis = next((i for i, s in enumerate(arr.shape) if s == tick_len), None)
    if tick_axis is not None:
        arr = np.moveaxis(arr, tick_axis, 0)  # (tick_len, ...)

    # 回帰（または k=1 扱い）
    if n_classes == 1:
        # (tick_len, anything...) -> mean over rest -> (tick_len,)
        return arr.reshape(tick_len, -1).mean(axis=1)

    # 分類: (tick_len, k, rest...) に寄せて rest を平均
    # class 軸を見つける（ticks 軸以外で n_classes に一致するもの）
    class_axis = next((i for i in range(1, arr.ndim) if arr.shape[i] == n_classes), None)
    if class_axis is None:
        # fallback: 最後を class とみなす（ダメなら shape を見直す必要あり）
        class_axis = arr.ndim - 1

    arr = np.moveaxis(arr, class_axis, 1)  # (tick_len, k, ...)
    if arr.ndim == 2:
        return arr  # (tick_len, k)
    return arr.reshape(tick_len, n_classes, -1).mean(axis=2)  # (tick_len, k)


def _summarize_base_for_hline(base_value, *, n_samples, n_classes, positive_class_index=1):
    """
    hlines 用の base をクラスごとに 1 本のスカラーへ要約する。
    """
    b = _to_numpy(base_value)
    if b is None:
        return [None] * n_classes

    if b.ndim == 0:
        return [float(b)] * n_classes

    if b.ndim == 1:
        # (n,) or (k,)
        if b.shape[0] == n_samples:
            m = float(np.mean(b))
            return [m] * n_classes
        if b.shape[0] == n_classes:
            return [float(x) for x in b]
        # それ以外は平均で潰す
        m = float(np.mean(b))
        return [m] * n_classes

    if b.ndim == 2:
        # (n,k) 典型
        if b.shape[0] == n_samples and b.shape[1] == n_classes:
            return [float(x) for x in b.mean(axis=0)]
        # (n,1) 等
        m = float(np.mean(b))
        return [m] * n_classes

    # fallback
    m = float(np.mean(b))
    return [m] * n_classes


# -----------------------------
# main: normalize shap/base
# -----------------------------
def _normalize_shap_base_for_add(base_value, shap_value, *, positive_class_index=1):
    """
    base_value / shap_value を numpy にして、足し算できる形へ正規化して返す。

    Returns:
        base_expanded, shap_arr_out, info
          - 回帰/単一クラス: base_expanded=(n,1) or scalar, shap_arr_out=(n,f)
          - 分類:            base_expanded=(n,1,k) or (1,1,k) or scalar, shap_arr_out=(n,f,k)

    NOTE:
      - shap が (n,f) で base が (n,k) のときは、positive_class_index を 1 クラス pick して単一クラス扱いにする。
    """
    info = {}

    # --- shap の取り出し ---
    if isinstance(shap_value, (list, tuple)):
        shap_arr = _stack_shap_list_per_class(shap_value)
        if shap_arr is None:
            return None, None, {"mode": "empty_shap", "shap_mode": "list_per_class"}
        info["shap_mode"] = "list_per_class"
    else:
        shap_arr = _to_numpy(shap_value)
        if shap_arr is None:
            return None, None, {"mode": "no_shap", "shap_mode": "none"}
        info["shap_mode"] = "array"

    base_arr = _to_numpy(base_value)

    # --- shap を (n,f,k) に寄せる ---
    shap_nfk, layout_info = _infer_shap_to_nfk(shap_arr, base_arr, positive_class_index=positive_class_index)
    info.update(layout_info)

    if shap_nfk is None:
        return None, None, {"mode": "no_shap_after_infer", **info}

    if shap_nfk.ndim == 2:
        # 想定外だが一応
        n, f = shap_nfk.shape
        shap_out = shap_nfk
        base_exp, base_info = _normalize_base_to_n1k(
            base_arr, n=n, k=1, shap_is_single_class=True, positive_class_index=positive_class_index
        )
        info.update(base_info)
        # base_exp: (n,1,1) or scalar -> (n,1)
        if isinstance(base_exp, np.ndarray) and base_exp.ndim == 3:
            base_exp = base_exp[:, :, 0]  # (n,1)
        return base_exp, shap_out, {"mode": "2d_shap", **info}

    # shap_nfk: (n,f,k)
    n, f, k = shap_nfk.shape

    # shap が元々 (n,f) の単一クラスだったケースを検出して base を pick するためのフラグ
    shap_is_single_class = (k == 1)

    base_exp, base_info = _normalize_base_to_n1k(
        base_arr, n=n, k=k, shap_is_single_class=shap_is_single_class, positive_class_index=positive_class_index
    )
    info.update(base_info)

    # 出力: k==1 のときは (n,f) に落とす（既存コード互換）
    if k == 1:
        shap_out = shap_nfk[:, :, 0]
        if isinstance(base_exp, np.ndarray) and base_exp.ndim == 3:
            base_exp = base_exp[:, :, 0]  # (n,1)
        return base_exp, shap_out, {"mode": "nfk_but_k=1_squeezed", "k": 1, **info}

    return base_exp, shap_nfk, {"mode": "nfk", "k": k, **info}


# -----------------------------
# main: shap_and_pd_array
# -----------------------------
def shap_and_pd_array(model, target, *, positive_class_index=1, return_info=False):
    """
    SHAP + base を足した配列と、PD 用の ticks / 曲線を返す。

    Returns (default):
        shap_values, x_ticks, x_pd, base_value, features_list, sort_desc

    If return_info=True:
        (..., info)
    """
    m = model.models[target]

    # --- importance -> sort ---
    importance_raw = m.importances["pfi_combine"]
    imp_1d = _reduce_importance_to_1d(importance_raw)
    if imp_1d is None or m.decomposition:
        # フォールバック: all_cols の順
        features = np.array(m.all_cols)
        sort_desc = np.arange(len(features))
    else:
        sort_desc = np.argsort(imp_1d)[::-1]

    features = np.array(m.all_cols)
    features_list = list(features[sort_desc])  # 重要度降順

    if not m.decomposition:
        # --- SHAP / base ---
        base_value = getattr(m, "base_values", None)
        if getattr(m, "shap_values", None) is not None:
            shap_value = m._combine_shap()
        else:
            shap_value = None
    
        base_exp, shap_arr, info = _normalize_shap_base_for_add(
            base_value, shap_value, positive_class_index=positive_class_index
        )
    
        # --- base + shap ---
        if shap_arr is None:
            # shap が無い
            shap_values = np.zeros_like(model.X)
            info["fallback"] = "zeros_like_model_X"
        else:
            if base_exp is None:
                shap_values = shap_arr
                info["base_add"] = "no_base"
            else:
                shap_values = base_exp + shap_arr
                info["base_add"] = "added"
    
        # --- 特徴量順を「重要度降順」に並べ替え（※ここがズレやすいので厳密に）---
        # shap_values の feature 軸は「元の列順」前提なので sort_desc で並べ替えたら、
        # 以降は「並べ替え後の i」を使って参照する（元 index j で参照しない）。
        if isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 2:
                shap_values = shap_values[:, sort_desc]
            elif shap_values.ndim == 3:
                shap_values = shap_values[:, sort_desc, :]
    
        # --- PD / ticks（shape を整える）---
        n_classes = shap_values.shape[2] if (isinstance(shap_values, np.ndarray) and shap_values.ndim == 3) else 1
    else:
        n_classes = len(model.models[target].target_items) if model.models[target].target_items is not None else 1
        shap_values = None
        base_value = None

    x_pd = []
    x_ticks = []
    for i, feat_name in enumerate(features_list):
        pd_raw, ticks = m.importances["pd"][feat_name]
        pd_curve = _normalize_pd_curve(pd_raw, ticks, n_classes=n_classes)
        x_pd.append(pd_curve)
        x_ticks.append(ticks)

    out = (shap_values, x_ticks, x_pd, base_value, features_list, sort_desc)
    if return_info:
        return (*out, info)
    return out


# -----------------------------
# main: plotting
# -----------------------------
def save_pd_plot_mpl(
    model,
    target,
    *,
    positive_class_index=1
):
    """Save PD/ICE plots with actual-data overlays.

    Args:
        model: Fitted model pipeline that contains the target child model.
        target: Target column name to visualize.
        positive_class_index: Class index used when summarizing binary or
            multiclass outputs.
    """
    shap_values, x_ticks, x_pd, base_value, features_list, sort_desc = shap_and_pd_array(
        model, target, positive_class_index=positive_class_index, return_info=False
    )

    m = model.models[target]
    n_feat = len(features_list)

    # shap_values shape
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        n_samples, _, n_classes = shap_values.shape
    else:
        n_samples = len(m.X_sample) if m.X_sample is not None else len(m.X)
        n_classes = 1

    # クラス名（あれば）
    class_names = None
    try:
        class_names = m.target_items
        n_classes = len(class_names)
    except Exception:
        class_names = None

    
    # base(hline) をクラスごとに要約
    base_lines = _summarize_base_for_hline(
        base_value, n_samples=n_samples, n_classes=n_classes, positive_class_index=positive_class_index
    )

    fig, axes = plt.subplots(
        n_classes, n_feat,
        figsize=(4 * n_feat, 3 * n_classes),
        sharey=False
    )
    axes = np.atleast_2d(axes)  # (n_classes, n_feat)

        
    for i in range(n_feat):
        j = sort_desc[i]  # 元の列 index（X 参照用）
        feat_name = features_list[i]

        x = m.X.iloc[:, j]
        xmin = m.X.iloc[:, j].min() if feat_name not in m.cat_cols else 0
        xmax = m.X.iloc[:, j].max() if feat_name not in m.cat_cols else len(m.unique_cols.get(feat_name, []))-1
        
        # 回帰 / 単一クラス
        if n_classes == 1:
            ax = axes[0, i]

            # PD
            if x_pd[i] is not None:
                y_pd = x_pd[i] if np.asarray(x_pd[i]).ndim == 1 else np.asarray(x_pd[i])[:, 0]
                ax.plot(x_ticks[i], y_pd, color='orange')
            if not m.decomposition and not m.ensemble:
                y_shap = shap_values[:, i] if shap_values.ndim == 2 else shap_values[:, i, 0]
                ax.scatter(x, y_shap, color='blue', alpha=.75)
            ax.scatter(x, m.y, color='red', alpha=.5)

            # base
            if base_lines[0] is not None:
                ax.hlines(base_lines[0], xmin, xmax, linestyle="--", color="black", linewidth=0.5)

            ax.set_xlabel(feat_name)
            if i == 0:
                ax.set_ylabel(target)

        # マルチクラス
        else:
            for k in range(n_classes):
                ax = axes[k, i]

                if x_pd[i] is not None:
                    pd_i = np.asarray(x_pd[i])
                    # 期待: (len_ticks, k)
                    y_pd = pd_i[:, k] if (pd_i.ndim == 2 and pd_i.shape[1] == n_classes) else pd_i.reshape(len(x_ticks[i]), -1)[:, 0]
                    ax.plot(x_ticks[i], y_pd, color='orange')

                if not m.decomposition:
                    ax.scatter(x, shap_values[:, i, k], color='blue', alpha=.75)
                class_label = class_names[k] if class_names is not None and k < len(class_names) else k
                class_candidates = [class_label]
                item2idx = getattr(m, "item2idx", None)
                if item2idx and class_label in item2idx:
                    class_candidates.append(item2idx[class_label])
                class_candidates.append(k)
                y_values = np.asarray(m.y).ravel()
                actual_class = np.zeros(y_values.shape, dtype=bool)
                for class_candidate in class_candidates:
                    actual_class |= y_values == class_candidate
                ax.scatter(x, actual_class.astype(int), color='red', alpha=.5)

                if base_lines[k] is not None:
                    ax.hlines(base_lines[k], xmin, xmax, linestyle="--", color="black", linewidth=0.5)

                ax.set_xlabel(feat_name)
                if i == 0:
                    label = f"{target} (class={k})"
                    if class_names is not None and k < len(class_names):
                        label = f"{target} ({class_names[k]})"
                    ax.set_ylabel(label)

    plt.tight_layout()
    plot_image_path = f"{IMAGE_PATH}/pd_{target}.png"
    fig.savefig(plot_image_path)
    plt.close(fig)  # グラフを閉じてメモリを解放

def _plot_pd2d_heatmap(
    X_PD: np.ndarray,
    xticks1: np.ndarray,
    xticks2: np.ndarray,
    *,
    class_index: int | None = None,   # 分類probaのどのクラスを見るか
    agg: str = "mean",                # "mean" or "median"
    title: str = "PD heatmap",
    xlabel: str = "x",
    ylabel: str = "y",
    ax = None,
    vmin = None,
    vmax = None
):
    """
    X_PD: (n_grid_total, n_samples, out_dim) or (n_grid_total, n_samples, 1)
    xticks1/2: ravel済み grid (n_grid_total,)
    """

    # --- ravel grid から (nx, ny) と tick順序を復元（meshgrid(indexing="xy")前提） ---
    xticks1 = np.asarray(xticks1)
    xticks2 = np.asarray(xticks2)

    # 1行目の長さ nx を「xticks2 が変わる最初の位置」から推定
    diff = np.where(xticks2 != xticks2[0])[0]
    nx = int(diff[0]) if len(diff) > 0 else len(xticks2)
    x_order = xticks1[:nx]
    y_order = xticks2[::nx]
    ny = len(y_order)

    # --- PD（ICEを集約） ---
    if agg == "mean":
        Z = X_PD.mean(axis=1)      # (grid, out_dim)
    elif agg == "median":
        Z = np.median(X_PD, axis=1)
    else:
        raise ValueError(f"Unknown agg: {agg}")

    # out_dim 選択
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    if Z.shape[-1] == 1:
        z1 = Z[:, 0]
    else:
        if class_index is None:
            class_index = 1  # 迷ったら positive を 1 にする想定
        z1 = Z[:, class_index]

    # --- 2Dへ戻す (ny, nx) ---
    Z2 = z1.reshape(ny, nx)

    # --- 描画 ---
    # fig, ax = plt.subplots()
    im = ax.imshow(Z2, origin="lower", aspect="auto", interpolation="nearest", cmap="bwr", vmin=vmin, vmax=vmax)
    # fig.colorbar(im, ax=ax)

    # 数値軸かカテゴリ軸かをざっくり判定（object/strはカテゴリとして扱う）
    is_x_cat = (x_order.dtype == object) or isinstance(x_order[0], str)
    is_y_cat = (y_order.dtype == object) or isinstance(y_order[0], str)

    if is_x_cat:
        ax.set_xticks(np.arange(nx))
        ax.set_xticklabels([str(v) for v in x_order], rotation=45, ha="right")
    else:
        ax.set_xticks(np.linspace(0, nx - 1, 6))
        ax.set_xticklabels([f"{v:.3g}" for v in np.linspace(float(x_order[0]), float(x_order[-1]), 6)])

    if is_y_cat:
        ax.set_yticks(np.arange(ny))
        ax.set_yticklabels([str(v) for v in y_order])
    else:
        ax.set_yticks(np.linspace(0, ny - 1, 6))
        ax.set_yticklabels([f"{v:.3g}" for v in np.linspace(float(y_order[0]), float(y_order[-1]), 6)])

def plot_pd_heatmap_matrix_lower(
    model,
    target,
    # features: list[str],
    *,
    unique_dict: dict | None = None,
    bounds: dict | None = None,
    n_grid_num: int = 30,
    max_samples: int = 150,
    random_state: int = 0,
    class_label: str | None = None,
    figsize_scale: float = 2.6,     # 1要素あたりのサイズ
    show_feature_name_on_diag: bool = True,
):
    """
    features の全ペアについて、下三角のみ PD ヒートマップを描く。
    """
    X = model.X
    features = model.models[target].all_cols
    unique_dict = unique_dict or {}
    n = len(features)
    class_index = model.models[target].item2idx[class_label] if class_label is not None else None

    xmax = model.models[target].cv_preds["train"].max().item()
    xmin = model.models[target].cv_preds["train"].min().item()
    d = (xmax - xmin)*0.1
    xmax += d
    xmin -= d

    if model.models[target].task=="classification":    
        xmax = min(xmax, 1)
        xmin = min(xmin, 0)
    
    fig, axes = plt.subplots(n, n, figsize=(figsize_scale * n, figsize_scale * n))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i <= j:
                ax.axis("off")
                if i == j and show_feature_name_on_diag:
                    ax.axis("on")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.text(0.5, 0.5, features[i], ha="center", va="center")
                continue

            fx = features[j]  # x: 列
            fy = features[i]  # y: 行
            Z2, x_ticks, y_ticks = model.models[target].get_pd_2d(target_cols=[fx, fy])
            _plot_pd2d_heatmap(Z2, x_ticks, y_ticks, class_index=class_index, ax=ax, vmin=xmin, vmax=xmax)

            # 軸ラベルは外側だけ（ごちゃつき防止）
            if i == n - 1:
                ax.set_xlabel(fx)
            else:
                ax.set_xticks([])

            if j == 0:
                ax.set_ylabel(fy)
            else:
                ax.set_yticks([])

    # カラーバーは共有で1本（見た目重視）。スケールは各subplotで違う可能性あり。
    # 厳密に揃えたいなら vmin/vmax を全ペアで事前計算してください。
    title = "probability of class " + str(class_label) if class_label is not None else None
    mappable = axes[1, 0].images[0]   # 例：色を付けたimshowの戻り
    
    # 右上に固定するカラーバー専用Axes（[left, bottom, width, height] は 0〜1 のFigure座標）
    cax = fig.add_axes([0.92, 0.58, 0.02, 0.32])  # ←ここで位置/サイズ調整
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(title)
    
    # fig.colorbar(axes[1, 0].images[0], ax=axes, shrink=0.4, label=title)
    plt.tight_layout()

    if class_label is not None:
        plot_image_path = f"{IMAGE_PATH}/pd2d_{target}_{class_label}.png"
    else:
        plot_image_path = f"{IMAGE_PATH}/pd2d_{target}.png"
    fig.savefig(plot_image_path)
    plt.close(fig)  # グラフを閉じてメモリを解放

def save_pca_mpl(model, target):
    if (model.models[target].decomposition)and(model.models[target].decomposition_method in ["PCA","NMF","ICA"]):
        feature_names = [
            c.replace("num_cat__num__", "").replace("num_cat__cat__", "") \
            for c in model.models[target].model["preprocess"]["column_preprocess"].get_feature_names_out()
        ]
        
        pca_int = model.models[target].model["preprocess"]["common_preprocess"].components_
        fig, ax = plt.subplots(1,len(pca_int), figsize=(2+2.5*len(pca_int),4), sharey=True)
        
        for i in range(len(pca_int)):
            ax[i].barh(y=feature_names[::-1], width=pca_int[i][::-1], )
            ax[i].set_title("dec"+str(i))
        plt.tight_layout();
        
        plot_image_path = f"{IMAGE_PATH}/dec_{target}.png"
        fig.savefig(plot_image_path)
        plt.close(fig)  # グラフを閉じてメモリを解放

def save_corrmatrix_mpl(model):
    num_cols = model.num_cols
    target_cols = [c for c in model.target_cols if c not in list(model.target_items.keys())]
    labels = list(target_cols)+list(num_cols)
    n = len(labels)
    df = pd.concat([model.models[target_cols[0]].X]+[model.models[target_cols[i]].y for i in range(len(target_cols))],axis=1).dropna()
    
    # 相関係数の計算
    cor_matrix = np.corrcoef(df[labels].T)
    
    # VIFの計算
    X = df[list(num_cols)].dropna().to_numpy()
    vif_values = [0]*len(target_cols)+[variance_inflation_factor(X, i) for i in range(len(list(num_cols)))]
    
    # 相互情報量の計算（対称化）
    mi_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                mi_matrix[i, j] = 0  # VIFが入るので0
            else:
                mi_ij = mutual_info_regression(df[[labels[i]]], df[labels[j]], discrete_features=False)[0]
                mi_ji = mutual_info_regression(df[[labels[j]]], df[labels[i]], discrete_features=False)[0]
                mi_sym = (mi_ij + mi_ji) / 2  # 対称化
                mi_matrix[i, j] = mi_sym
                mi_matrix[j, i] = mi_sym
    
    # vmin, vmax を個別に取得
    cor_min, cor_max = np.min(cor_matrix), np.max(cor_matrix)
    vif_min, vif_max = np.min(vif_values), np.max(vif_values)
    mi_min, mi_max = np.min(mi_matrix), np.max(mi_matrix)
    
    # カラーマップの設定
    cor_cmap = cm.get_cmap('coolwarm')
    vif_cmap = cm.get_cmap('Greens')
    mi_cmap = cm.get_cmap('Purples')
    
    # 正規化オブジェクト
    cor_norm = Normalize(vmin=cor_min, vmax=cor_max)
    vif_norm = Normalize(vmin=vif_min, vmax=vif_max)
    mi_norm = Normalize(vmin=mi_min, vmax=mi_max)
    
    # プロットの作成
    fig, ax = plt.subplots(figsize=(n+2, n+2))
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    
    # **y 軸を反転**
    ax.invert_yaxis()
    
    # 各セルの描画
    for i in range(n):
        for j in range(n):
            if i == j:
                # 対角成分 (VIF)
                color = vif_cmap(vif_norm(vif_values[i]))
                text = f"{vif_values[i]:.2f}"
            elif i > j:
                # 左三角 (相関係数)
                color = cor_cmap(cor_norm(cor_matrix[i, j]))
                text = f"{cor_matrix[i, j]:.2f}"
            else:
                # 右三角 (相互情報量)
                color = mi_cmap(mi_norm(mi_matrix[i, j]))
                text = f"{mi_matrix[i, j]:.2f}"
    
            # セルの描画
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color=color))
    
            # テキストの色を適切に設定
            text_color = 'black' if (i == j or abs(cor_matrix[i, j]) < 0.5) else 'white'
            ax.text(j, i, text, ha='center', va='center', color=text_color)
    
    # グリッド線を追加
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    
    # 領域ラベルの追加
    ax.text(n-0.75, n-0.75, 'VIF', ha='center', va='center', color='black', fontsize=14, fontweight='bold')
    ax.text(0.25, n-0.75, 'Correlation', ha='center', va='center', color='black', fontsize=14, fontweight='bold')
    ax.text(n-1.5, -0.3, 'Mutual Information', ha='center', va='center', color='black', fontsize=14, fontweight='bold')
    
    # タイトル
    plt.title("Correlation (Lower) / VIF (Diagonal) / Mutual Info (Upper)")
    
    plot_image_path = f"{IMAGE_PATH}/corrmatrix.png"
    fig.savefig(plot_image_path)
    plt.close(fig)  # グラフを閉じてメモリを解放

def save_scattermatrix_mpl(model):
    num_cols = model.num_cols
    target_cols = [c for c in model.target_cols if c not in list(model.target_items.keys())]
    collist = list(target_cols)+list(num_cols)
    n = len(collist)
    
    df = pd.concat([model.models[target_cols[0]].X]+[model.models[target_cols[i]].y for i in range(len(target_cols))],axis=1).dropna()
    
    fig ,ax = plt.subplots(
        len(collist), len(collist),
        figsize=(2*len(collist),2*len(collist)),
        # sharex='col', sharey='row'
    )
    for i1, c1 in enumerate(collist):
        for i2, c2 in enumerate(collist):
            if i1==i2:
                ax[i1][i2].hist(df[c1], bins=12, alpha=0.7)
            elif i1>i2:
                ax[i1][i2].scatter(df[c2], df[c1], alpha=0.5, s=15, color="blue")
            else:
                # 上三角 → 回帰直線 + 誤差範囲
                x = df[c2]
                y = df[c1]
    
                # 1次の線形回帰（共分散行列も取得）
                (slope, intercept), cov = np.polyfit(x, y, 1, cov=True)
                slope_std_err = np.sqrt(cov[0, 0])
                intercept_std_err = np.sqrt(cov[1, 1])
    
                # 回帰直線の計算
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = slope * x_fit + intercept
    
                # 残差標準偏差
                y_pred = slope * x + intercept
                residuals = y - y_pred
                sigma = np.sqrt(np.sum(residuals**2) / (len(y) - 2))
    
                # 標準誤差の計算（sns.regplot の Wald信頼区間に一致）
                mean_x = np.mean(x)
                n_samples = len(x)
                se_fit = sigma * np.sqrt(
                    1/n_samples + (x_fit - mean_x)**2 / np.sum((x - mean_x)**2)
                )
                
                # 95% 信頼区間（±1.96 SE）
                lower_bound = y_fit - 1.96 * se_fit
                upper_bound = y_fit + 1.96 * se_fit
    
                # 回帰直線のプロット
                ax[i1][i2].plot(x_fit, y_fit, color="red", linewidth=2)
    
                # 信頼区間のプロット
                ax[i1][i2].fill_between(x_fit, lower_bound, upper_bound, color="red", alpha=0.2)
    
                ax[i1][i2].scatter(df[c2], df[c1], alpha=0.5, s=15, color="blue")
            # 軸ラベルの設定
            if (i1 == n - 1):
                ax[i1][i2].set_xlabel(collist[i2], size=10)
            else:
                ax[i1][i2].set_xticklabels([])
            
            if i2 == 0:
                ax[i1][i2].set_ylabel(collist[i1], size=10)
            else:
                ax[i1][i2].set_yticklabels([])
                
    plot_image_path = f"{IMAGE_PATH}/scattermatrix.png"
    fig.savefig(plot_image_path)
    plt.close(fig)  # グラフを閉じてメモリを解放