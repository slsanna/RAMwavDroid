#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
APK-level classification from audio-feature CSVs (benign + malware), APK-ONLY MODE.

- Expects final CSVs (preferred):
    features/benign_apk_features.csv
    features/malware_apk_features.csv
  Falls back to older names (benign_<sonify>_features.csv, malware_<sonify>_features.csv,
  benign_dumps_<sonify>_features.csv, dumps_dataset_<sonify>_features.csv), or keyword search.

- Labels normalized to: 0 (benign) / 1 (malware)
- Derives 'apk' from filename/path if missing
- Aggregates per APK via mean
- Train/test split BY APK (no leakage)

Models:
  KNN, SVM(RBF), LinearSVC, RandomForest, ExtraTrees, HistGradientBoosting,
  GradientBoosting, AdaBoost, LogisticRegression, MLP, GaussianNB, QDA
+ Soft-voting ensemble over probabilistic base-learners
+ Optional: XGBoost / LightGBM / CatBoost if available

Python 3.8+ compatible.
"""

import os
import csv
from glob import glob
from collections import Counter
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from pandas.errors import ParserError
from loguru import logger

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, average_precision_score

# Base & extended classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    HistGradientBoostingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# -------------------- CONFIG --------------------

BASE_CSV_SEARCH_ROOT = "PATH"

TARGET_COL = "category"   # 0 = benign, 1 = malware
APK_COL    = "apk"
SONIFY_COL = "sonify_type"

NON_FEATURE_COLS = {
    'filename', 'file', 'path', 'id', 'name',
    'apk', 'apk_id',
    'sample_rate', 'num_samples', 'duration_sec', 'duration',
    'length', 'size', 'timestamp', 'date',
    'num_regions', 'region_types', 'dl_embedding_dim',
    'analysis_type', 'feature_type', 'source', 'audio_path', 'wav_path',
    TARGET_COL, SONIFY_COL
}

TEST_SIZE      = 0.20
RANDOM_STATE   = 42
MODEL_DIR      = "models"
RUNS_DIR       = "./runs/apk_eval"
LOGS_DIR       = "./logs"

# -------------------- LOGGING --------------------

def setup_run_logging(sonify: str, mode_name: str) -> Tuple[str, int]:
    os.makedirs(LOGS_DIR, exist_ok=True)
    run_name = f"{sonify}__{mode_name}"
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    base_log = os.path.join(run_dir, f"{run_name}.log")
    logger.remove()
    logger.add(base_log, level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
               enqueue=True)
    stderr_sink_id = logger.add(sink=lambda msg: print(msg, end=""),
                                level="INFO", format="<lvl>{message}</lvl>\n", enqueue=True)
    logger.info("[RUN] Logging to {}", base_log)
    return run_dir, stderr_sink_id

def with_model_log(run_dir: str, sonify: str, mode_name: str, model_name: str):
    class _Sink:
        def __enter__(self_inner):
            fname = os.path.join(run_dir, f"{sonify}__{mode_name}__{model_name}.log")
            self_inner.sink_id = logger.add(fname, level="INFO",
                                            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
                                            enqueue=True)
            logger.info("[MODEL LOG] {} → {}", model_name, fname)
            return self_inner
        def __exit__(self_inner, *_):
            logger.remove(self_inner.sink_id)
    return _Sink()

# -------------------- PATH & IO HELPERS --------------------

def _normalize_root(p: str) -> str:
    """Collapse accidental .../features/features into .../features."""
    p = os.path.abspath(p)
    while os.path.basename(p) == "features" and os.path.basename(os.path.dirname(p)) == "features":
        p = os.path.dirname(p)
    return p

def _find_file(p: Optional[str]) -> Optional[str]:
    return p if (p and os.path.isfile(p)) else None

def _search_csv(filename: str, root: str) -> Optional[str]:
    pats = [os.path.join(root, "**", filename), os.path.join(root, "**", f"*{filename}")]
    hits = []
    for pat in pats:
        hits.extend(glob(pat, recursive=True))
    hits = [h for h in hits if os.path.isfile(h)]
    if not hits:
        return None
    hits.sort(key=lambda q: os.path.getmtime(q), reverse=True)
    return hits[0]

def read_csv_robust(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except ParserError as e:
        logger.warning("Fast read_csv failed for {}: {}", path, e)
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip",
                               quoting=csv.QUOTE_MINIMAL, quotechar='"', escapechar='\\')
        except TypeError:
            return pd.read_csv(path, engine="python", error_bad_lines=False, warn_bad_lines=True,
                               quoting=csv.QUOTE_MINIMAL, quotechar='"', escapechar='\\')

def load_two_csvs(sonify: str,
                  search_root: str,
                  csv_benign: Optional[str],
                  csv_mal: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    Priority:
      1) final names: benign_apk_features.csv / malware_apk_features.csv
      2) old per-sonify: benign_<sonify>_features.csv / malware_<sonify>_features.csv
      3) legacy dumps: benign_dumps_<sonify>_features.csv / dumps_dataset_<sonify>_features.csv
      4) keyword fallback
    """
    search_root = _normalize_root(search_root)
    son = (sonify or "").lower()

    new_b = os.path.join(search_root, "benign_apk_features.csv")
    new_m = os.path.join(search_root, "malware_apk_features.csv")
    old_b = os.path.join(search_root, f"benign_{son}_features.csv")
    old_m = os.path.join(search_root, f"malware_{son}_features.csv")
    legacy_b = f"benign_dumps_{son}_features.csv"
    legacy_m = f"dumps_dataset_{son}_features.csv"

    def _first_hit(explicit: Optional[str], *candidates: str, keywords: Optional[List[str]] = None) -> Optional[str]:
        if explicit and os.path.isfile(explicit):
            return explicit
        for c in candidates:
            if c and os.path.isfile(c):
                return c
        if keywords:
            for kw in keywords:
                hit = _search_csv(kw, search_root)
                if hit:
                    return hit
        return None

    benign_path = _first_hit(
        csv_benign, new_b, old_b, keywords=[os.path.basename(new_b), os.path.basename(old_b), legacy_b, "benign"]
    )
    mal_path = _first_hit(
        csv_mal, new_m, old_m, keywords=[os.path.basename(new_m), os.path.basename(old_m), legacy_m, "malware"]
    )

    if not benign_path or not mal_path:
        nearby = [p for p in glob(os.path.join(search_root, "**", "*.csv"), recursive=True)]
        nearby = sorted(nearby)[-20:]
        hint = "\n".join("  - " + p for p in nearby) if nearby else "  (no CSVs found under search root)"
        raise FileNotFoundError(
            "Could not locate the expected feature CSVs.\n"
            f" search_root = {search_root}\n"
            f" tried benign: {new_b}, {old_b}, *{legacy_b}\n"
            f" tried malware: {new_m}, {old_m}, *{legacy_m}\n"
            f" Nearby CSVs:\n{hint}"
        )

    logger.info("Using benign CSV:  {}", benign_path)
    logger.info("Using malware CSV: {}", mal_path)

    df_b = read_csv_robust(benign_path)
    df_m = read_csv_robust(mal_path)

    if TARGET_COL not in df_b.columns:
        df_b[TARGET_COL] = 0
        logger.info("Injected '{}'=0 into benign CSV.", TARGET_COL)
    if TARGET_COL not in df_m.columns:
        df_m[TARGET_COL] = 1
        logger.info("Injected '{}'=1 into malware CSV.", TARGET_COL)

    return df_b, df_m, benign_path, mal_path

# -------------------- CORE COLS & PREP --------------------

def _derive_apk_from_paths(df: pd.DataFrame) -> Optional[str]:
    for cand in ('apk', 'apk_id', 'filename', 'file', 'path'):
        if cand in df.columns:
            if cand in ('apk', 'apk_id'):
                return cand
            def to_apk(x):
                s = str(x)
                base = os.path.basename(s)
                stem = os.path.splitext(base)[0]
                return stem
            df['apk'] = df[cand].map(to_apk)
            return 'apk'
    return None

def ensure_apk_col(df: pd.DataFrame) -> str:
    col = _derive_apk_from_paths(df)
    if not col:
        raise SystemExit("No APK id column found/derivable. Need 'apk'/'apk_id' or a filename/path column.")
    if col != APK_COL:
        logger.info("Using '{}' as APK column.", col)
    return col

def normalize_labels_inplace(df: pd.DataFrame) -> None:
    if df[TARGET_COL].dtype == object:
        mapping = {'benign':0,'malware':1,'Benign':0,'Malicious':1,'0':0,'1':1,
                   'benign_dumps':0,'dumps_dataset':1}
        df[TARGET_COL] = df[TARGET_COL].map(mapping).fillna(df[TARGET_COL]).astype(int)

def build_apk_level_table(df: pd.DataFrame, apk_col: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    exclude = set(NON_FEATURE_COLS) | {apk_col}
    feat_cols = [c for c in df.columns if c not in exclude]
    if not feat_cols:
        raise SystemExit("No feature columns left after exclusions.")
    num_df = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    agg_df = num_df.groupby(df[apk_col]).mean()
    labels = df.groupby(apk_col)[TARGET_COL].agg(lambda s: Counter(s).most_common(1)[0][0])
    agg_df[TARGET_COL] = labels.astype(int)
    agg_df = agg_df.reset_index()
    apk_list = agg_df[apk_col].astype(str).tolist()
    return agg_df, feat_cols, apk_list

# -------------------- EVALUATION --------------------

def _get_probs(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        if p.ndim == 1:
            return p
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return s
    return model.predict(X).astype(float)

def evaluate_model(model, X_test, y_test, apk_test, model_name: str):
    prob = _get_probs(model, X_test)

    if len(np.unique(y_test)) > 1:
        fpr, tpr, thr = roc_curve(y_test, prob)
        j = tpr - fpr
        best_thr = float(thr[int(np.argmax(j))])
        auc = float(roc_auc_score(y_test, prob))
        ap  = float(average_precision_score(y_test, prob))
        logger.info("[{}] ROC AUC: {:.4f}, PR AUC: {:.4f}, BestThr={:.3f}", model_name, auc, ap, best_thr)
    else:
        best_thr = 0.5
        logger.info("[{}] Single-class test set; using Thr=0.5.", model_name)

    y_pred = (prob > best_thr).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    logger.info("\n[{}] Classification Report:\n{}", model_name,
                classification_report(y_test, y_pred, target_names=["Benign","Malicious"]))
    logger.info("\n[{}] Confusion Matrix:\n{}", model_name, cm)

# -------------------- TRAIN & ENSEMBLE --------------------

def tune_and_eval_models(X_train, y_train, X_test, y_test, apk_test, feat_cols, *, run_dir: str, sonify: str, mode_name: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    k_features = max(1, min(200, X_train.shape[1]))

    # Hyperparam grids (kept compact to avoid long runs)
    grids = {
        "KNN": {"n_neighbors": [3,5,7], "metric": ["euclidean","manhattan"]},
        "SVM": {"C": [0.5,1.0,2.0], "gamma": [0.01,0.001], "kernel": ["rbf"]},
        "LinearSVC": {"C": [0.5,1.0,2.0], "loss": ["hinge","squared_hinge"]},
        "RandomForest": {"n_estimators": [200,400], "max_depth": [None,20]},
        "ExtraTrees": {"n_estimators": [400,800], "max_depth": [None,20,40], "max_features": ["sqrt","log2",None]},
        "HistGradientBoosting": {"max_iter":[200,400], "learning_rate":[0.05,0.1], "max_depth":[None,10]},
        "GradientBoosting": {"n_estimators":[200,400], "learning_rate":[0.05,0.1], "max_depth":[3,5]},
        "AdaBoost": {"n_estimators":[200,400], "learning_rate":[0.5,1.0]},
        "LogisticRegression": {"C":[0.1,1.0], "solver":["saga"], "max_iter":[3000]},
        "MLP": {"hidden_layer_sizes":[(256,), (512,256)], "max_iter":[500], "activation":["relu"], "early_stopping":[True]},
        "GaussianNB": {"var_smoothing":[1e-9,1e-8,1e-7]},
        "QDA": {"reg_param":[0.0,0.1,0.5]},
    }
    models = {
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "LinearSVC": LinearSVC(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1),
        "MLP": MLPClassifier(random_state=RANDOM_STATE),
        "GaussianNB": GaussianNB(),
        "QDA": QuadraticDiscriminantAnalysis(),
    }

    tuned = {}
    for name, base in models.items():
        with with_model_log(run_dir, sonify, mode_name, name):
            logger.info("=== Tuning {} ===", name)
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(mutual_info_classif, k=k_features)),
                ("classifier", base),
            ])
            search = RandomizedSearchCV(
                pipe,
                param_distributions={f"classifier__{k}": v for k, v in grids[name].items()},
                n_iter=min(5, sum(len(v) for v in grids[name].values())),
                scoring="roc_auc",
                cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                n_jobs=-1,
                refit=True,
                random_state=RANDOM_STATE,
                verbose=1,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            tuned[name] = best
            out = os.path.join(MODEL_DIR, name.replace(" ","") + ".pkl")
            import pickle
            with open(out, "wb") as f:
                pickle.dump(best, f)
            logger.info("Saved {} -> {}", name, out)
            evaluate_model(best, X_test, y_test, apk_test, name)

    # Optional boosting (train straight without search to keep runtime sane)
    try:
        k_features = max(1, min(200, X_train.shape[1]))
        boosters = {}
        try:
            from xgboost import XGBClassifier
            boosters["XGBoost"] = XGBClassifier(
                eval_metric="logloss", tree_method="hist",
                random_state=RANDOM_STATE, n_estimators=400
            )
        except Exception as e:
            logger.info("XGBoost not available: {}", e)
        try:
            from lightgbm import LGBMClassifier
            boosters["LightGBM"] = LGBMClassifier(random_state=RANDOM_STATE)
        except Exception as e:
            logger.info("LightGBM not available: {}", e)
        try:
            from catboost import CatBoostClassifier
            boosters["CatBoost"] = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
        except Exception as e:
            logger.info("CatBoost not available: {}", e)

        for name, base in boosters.items():
            with with_model_log(run_dir, sonify, mode_name, name):
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("selector", SelectKBest(mutual_info_classif, k=k_features)),
                    ("classifier", base),
                ])
                pipe.fit(X_train, y_train)
                tuned[name] = pipe
                out = os.path.join(MODEL_DIR, name.replace(" ","") + ".pkl")
                import pickle
                with open(out, "wb") as f:
                    pickle.dump(pipe, f)
                logger.info("Saved {} -> {}", name, out)
                evaluate_model(pipe, X_test, y_test, apk_test, name)
    except Exception as e:
        logger.info("Boosting stage skipped: {}", e)

    # Soft-voting ensemble of probabilistic models only
    try:
        # Pick one tuned pipeline to grab scaler & selector (use LR)
        lr = tuned["LogisticRegression"]
        scaler   = lr.named_steps["scaler"]
        selector = lr.named_steps["selector"]

        # Models that expose predict_proba (exclude LinearSVC)
        proba_members = []
        for key in ["SVM","RandomForest","ExtraTrees","HistGradientBoosting",
                    "GradientBoosting","AdaBoost","LogisticRegression","MLP",
                    "GaussianNB","QDA"]:
            if key in tuned:
                proba_members.append( (key.lower(), tuned[key].named_steps["classifier"]) )

        ensemble = VotingClassifier(estimators=proba_members, voting="soft")
        ens_pipe = Pipeline([("scaler", scaler), ("selector", selector), ("classifier", ensemble)])
        ens_pipe.fit(X_train, y_train)
        evaluate_model(ens_pipe, X_test, y_test, apk_test, "VotingEnsemble")
    except Exception as e:
        logger.warning("Voting ensemble skipped: {}", e)

# -------------------- MAIN --------------------

def main_classify(
    selected_memoryregion: str = "all",  # unused; kept for API compatibility
    selected_sonification: str = "wav8",
    *,
    csv_benign: Optional[str] = None,
    csv_malware: Optional[str] = None,
    csv_root: Optional[str] = None
):
    sonify = selected_sonification
    mode_name = "apk_only"

    run_dir, _ = setup_run_logging(sonify, mode_name)

    root = _normalize_root(csv_root or BASE_CSV_SEARCH_ROOT)
    df_b, df_m, _, _ = load_two_csvs(sonify, root, csv_benign, csv_malware)

    # Optional filter by sonify type if column exists
    if SONIFY_COL in df_b.columns:
        before = len(df_b)
        df_b = df_b[df_b[SONIFY_COL].astype(str).str.lower() == sonify.lower()].reset_index(drop=True)
        logger.info("[benign] Filtered by {}='{}': kept {} / {} rows.", SONIFY_COL, sonify, len(df_b), before)
    if SONIFY_COL in df_m.columns:
        before = len(df_m)
        df_m = df_m[df_m[SONIFY_COL].astype(str).str.lower() == sonify.lower()].reset_index(drop=True)
        logger.info("[malware] Filtered by {}='{}': kept {} / {} rows.", SONIFY_COL, sonify, len(df_m), before)

    # Combine & sanity
    df = pd.concat([df_b, df_m], ignore_index=True)
    logger.info("Combined rows: {}", len(df))
    logger.info("CSV columns: {}", list(df.columns))

    # Ensure APK id & labels
    apk_col = ensure_apk_col(df)
    normalize_labels_inplace(df)

    # APK-only: keep rows as-is
    df_filtered = df.reset_index(drop=True)
    logger.info("[APK-ONLY MODE] Using all rows from CSVs (no region filtering).")

    # APK-level aggregation
    apk_df, feat_cols, _ = build_apk_level_table(df_filtered, apk_col)

    # Split by APK
    apk_counts = Counter(apk_df[TARGET_COL].tolist())
    logger.info("APK-level class balance (0=Benign, 1=Malicious): {}", dict(apk_counts))

    apks = apk_df[apk_col].astype(str).values
    y_apk = apk_df[TARGET_COL].values
    try:
        apks_train, apks_test = train_test_split(
            apks, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=y_apk if len(np.unique(y_apk)) > 1 else None
        )
    except ValueError:
        apks_train, apks_test = train_test_split(apks, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    train_df = apk_df[apk_df[apk_col].astype(str).isin(apks_train)].reset_index(drop=True)
    test_df  = apk_df[apk_df[apk_col].astype(str).isin(apks_test)].reset_index(drop=True)

    logger.info("APK samples -> Train: {}, Test: {}", len(train_df), len(test_df))
    logger.info("Train class balance: {}", dict(Counter(train_df[TARGET_COL].tolist())))
    logger.info("Test  class balance: {}", dict(Counter(test_df[TARGET_COL].tolist())))

    # Feature matrices
    exclude_now = set(NON_FEATURE_COLS) | {apk_col}
    feat_cols_final = [c for c in train_df.columns if c not in exclude_now]
    if not feat_cols_final:
        raise SystemExit("No features found after exclusions. Check your CSVs.")
    X_train = train_df[feat_cols_final].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[feat_cols_final].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    y_test  = test_df[TARGET_COL].values
    apk_test = test_df[apk_col].astype(str).tolist()

    # Train & evaluate
    tune_and_eval_models(X_train, y_train, X_test, y_test, apk_test, feat_cols_final,
                         run_dir=run_dir, sonify=sonify, mode_name=mode_name)

if __name__ == "__main__":
    # Example: keep your explicit CSVs if you prefer fixed paths
    main_classify(
        "all", "wav8",
        csv_benign="PATH",
        csv_malware="PATH")
