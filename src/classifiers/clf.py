#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
APK-level classification from audio-feature CSVs (benign+malware), with maps.txt-driven region selection.
- Loads TWO CSVs: benign_dumps_<sonify>_features.csv and dumps_dataset_<sonify>_features.csv
- Robust CSV loading (skips malformed lines that break the parser)
- Filters rows by selectionMode using <apk>/maps.txt
- Aggregates to one vector per APK (mean over rows)
- Train/test split BY APK (no leakage)
- Hyperparameter tuning with scikit-learn (KNN, SVM, RF, LR, MLP)
- Soft-voting ensemble (uses a shared scaler+selector)
- Boosting (XGBoost/LightGBM/CatBoost if installed)
- Prints confusion matrix & classification report
- Prints misclassified APKs with regions converted to friendly names via maps.txt/SELECTION_MAP

No command-line args required — call main(selected_memoryregion, selected_sonification, ...).
"""

import ast
import csv
import os
import re
import pickle
import warnings
from glob import glob
from functools import lru_cache
from collections import Counter
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.errors import ParserError

from loguru import logger

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    average_precision_score,
)

# Base classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# -------------------- CONFIG DEFAULTS --------------------
warnings.filterwarnings("ignore")

CSV_SEARCH_ROOT = "/home/ssanna/Desktop/malware_ram/Android/audio/malware-sonification-signal-processing"

ROOT_DIR        = "/home/ssanna/nas/RAM/Android"
BENIGN_DIR_NAME = "benign_dumps"
MAL_DIR_NAME    = "dumps_dataset"

TARGET_COL = "category"
APK_COL    = "apk"            # fallback to 'apk_id'
ADDR_COL   = "address"        # derive from 'memory_region' if missing
REGION_COL = "region_desc"    # fallback to 'memory_region'
SONIFY_COL = "sonify_type"

NON_FEATURE_COLS = {
    'filename', 'file', 'path', 'id', 'name',
    'apk', 'apk_id',
    'sample_rate', 'num_samples', 'duration_sec', 'duration',
    'length', 'size', 'timestamp', 'date',
    'num_regions', 'region_types', 'dl_embedding_dim'
}

SELECTION_MAP = {
    "stack": [r"\[stack\]"],
    "memory_regions": [r"\[stack\]", r"\[vvar\]"],
    "stack_vdso": [r"\[stack\]", r"\[vdso\]"],
    "base_odex-apk": [r"base\.odex", r"base\.apk"],
    "base_vdex-apk": [r"base\.vdex", r"base\.apk"],
    "base_odex-vdex": [r"base\.odex", r"base\.vdex"],
    "base_odex": [r"base\.odex"],
    "base_vdex": [r"base\.vdex"],
    "base_apk": [r"base\.apk"],
    "all_data_paths": [r"^/data/.*"],
    "data_stack": [r"^/data/.*", r"\[stack\]"],
    "complete_memory_regions": [
        r"\[heap\]", r"\[anon:libc_malloc\]", r"\[stack\]",
        r"\[vdso\]", r"\[vvar\]", r"\[vsyscall\]",
        r"\(deleted\)", r"u:object_r.*", r"data@resource-cache@"
    ],
    "base_apk_memory": [r"\[stack\]", r"\[vvar\]", r"base\.apk", r"base\.vdex", r"base\.odex"],
    "base_apk_stack": [r"\[stack\]", r"base\.apk", r"base\.vdex", r"base\.odex"],
    "apk_full": [
        r"base\.apk", r"base\.vdex", r"base\.odex", r"/data/.*/lib/.+\.so$",
        r"\[anon:dalvik.*\]", r"\[anon:.*\.apk.*\]", r"\.db$", r"\.db-shm$", r"\.db-wal$",
        r"Cookies", r"Preferences", r"ua_preferences", r"data@resource-cache@", r"\.dex$",
        r"/data/.+\.odex$"
    ],
    "apk_full_memory": [
        r"\[stack\]", r"\[vvar\]",
        r"base\.apk", r"base\.vdex", r"base\.odex", r"/data/.*/lib/.+\.so$",
        r"\[anon:dalvik.*\]", r"\[anon:.*\.apk.*\]", r"\.db$", r"\.db-shm$", r"\.db-wal$",
        r"Cookies", r"Preferences", r"ua_preferences", r"data@resource-cache@", r"\.dex$",
        r"/data/.+\.odex$"
    ],
    "apk_full_stack": [
        r"\[stack\]", r"base\.apk", r"base\.vdex", r"base\.odex", r"/data/.*/lib/.+\.so$",
        r"\[anon:dalvik.*\]", r"\[anon:.*\.apk.*\]", r"\.db$", r"\.db-shm$", r"\.db-wal$",
        r"Cookies", r"Preferences", r"ua_preferences", r"data@resource-cache@", r"\.dex$",
        r"/data/.+\.odex$"
    ],
    "system": [r"framework\.jar", r"boot\.vdex", r"boot-framework\.vdex", r"base\.odex"],
    "system_full": [
        r"dalvik-classes", r"framework\.jar", r"boot-framework\.art",
        r"boot-core-libart\.art", r"boot\.oat", r"boot-framework\.oat",
        r"boot\.vdex", r"boot-framework\.vdex", r"base\.odex"
    ],
    "libraries": [r"\.so", r"\[anon:lib.*"]
}

TEST_SIZE      = 0.20
RANDOM_STATE   = 42
MODEL_DIR      = "models"
SAVE_PLOTS     = False
RUNS_DIR       = "./runs/apk_eval"  # each run gets its own subfolder
LOGS_DIR       = "./logs"           # root for logs

# -------------------- LOGGING HELPERS --------------------

def setup_run_logging(sonify: str, selection_mode: str) -> Tuple[str, int]:
    """Create run folder + base log file; returns run_dir and base sink id."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    safe_mode = selection_mode.replace(os.sep, "_")
    run_name = f"{sonify}__{safe_mode}"
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    base_log = os.path.join(run_dir, f"{run_name}.log")
    # Remove default stderr sink and re-add with nicer format
    logger.remove()
    logger.add(
        base_log,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
        rotation=None,
        retention=None,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    # Also mirror to stderr in a minimal format
    stderr_sink_id = logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        format="<lvl>{message}</lvl>\n",
        enqueue=True,
    )
    logger.info("[RUN] Logging to {}", base_log)
    return run_dir, stderr_sink_id

def with_model_log(run_dir: str, sonify: str, selection_mode: str, model_name: str):
    """Context manager to add/remove a per-model sink."""
    class _ModelSink:
        def __enter__(self_inner):
            safe_mode = selection_mode.replace(os.sep, "_")
            fname = os.path.join(run_dir, f"{sonify}__{safe_mode}__{model_name}.log")
            self_inner.path = fname
            self_inner.sink_id = logger.add(
                fname,
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
                enqueue=True,
            )
            logger.info("[MODEL LOG] {} → {}", model_name, fname)
            return self_inner
        def __exit__(self_inner, exc_type, exc, tb):
            logger.remove(self_inner.sink_id)
    return _ModelSink()

# -------------------- IO / DATA HELPERS --------------------

def _find_file(p: str) -> Optional[str]:
    return p if os.path.isfile(p) else None

def _search_csv(filename: str, root: str) -> Optional[str]:
    pats = [
        os.path.join(root, "**", filename),
        os.path.join(root, "**", f"*{filename}"),
    ]
    hits = []
    for pat in pats:
        hits.extend(glob(pat, recursive=True))
    hits = [h for h in hits if os.path.isfile(h)]
    if not hits:
        return None
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]

def read_csv_robust(path: str) -> pd.DataFrame:
    """Try fast C-engine; on failure, use python engine and skip bad lines."""
    try:
        return pd.read_csv(path)
    except ParserError as e:
        logger.warning("Standard read_csv failed for {}: {}", path, e)
        try:
            df = pd.read_csv(
                path,
                engine="python",
                on_bad_lines="skip",
                quoting=csv.QUOTE_MINIMAL,
                quotechar='"',
                escapechar='\\',
            )
            logger.warning("Loaded {:,} rows from {} with on_bad_lines='skip'.", len(df), path)
            return df
        except TypeError:
            df = pd.read_csv(
                path,
                engine="python",
                error_bad_lines=False,   # legacy pandas
                warn_bad_lines=True,
                quoting=csv.QUOTE_MINIMAL,
                quotechar='"',
                escapechar='\\',
            )
            logger.warning("Loaded {:,} rows from {} with error_bad_lines=False (legacy).", len(df), path)
            return df

def load_benign_and_mal_csvs(sonify: str,
                             search_root: str,
                             csv_benign: Optional[str],
                             csv_mal: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    son = sonify.lower()
    benign_name = f"benign_dumps_{son}_features.csv"
    mal_name    = f"dumps_dataset_{son}_features.csv"

    benign_path = csv_benign or _find_file(os.path.join(search_root, benign_name)) or _search_csv(benign_name, search_root)
    mal_path    = csv_mal    or _find_file(os.path.join(search_root, mal_name))    or _search_csv(mal_name, search_root)

    if not benign_path:
        raise FileNotFoundError(f"Cannot find benign CSV '{benign_name}' under {search_root}")
    if not mal_path:
        raise FileNotFoundError(f"Cannot find malware CSV '{mal_name}' under {search_root}")

    logger.info("Using benign CSV:  {}", benign_path)
    logger.info("Using malware CSV: {}", mal_path)

    df_b = read_csv_robust(benign_path)
    df_m = read_csv_robust(mal_path)

    if TARGET_COL not in df_b.columns:
        df_b[TARGET_COL] = "benign_dumps"
        logger.info("Injected '{}'='benign_dumps' into benign CSV.", TARGET_COL)
    if TARGET_COL not in df_m.columns:
        df_m[TARGET_COL] = "dumps_dataset"
        logger.info("Injected '{}'='dumps_dataset' into malware CSV.", TARGET_COL)

    return df_b, df_m, benign_path, mal_path

def ensure_core_columns(df: pd.DataFrame) -> Tuple[str, Optional[str], Optional[str]]:
    apk_col = APK_COL if APK_COL in df.columns else ('apk_id' if 'apk_id' in df.columns else None)
    if not apk_col:
        raise SystemExit("No APK column found. Expected 'apk' or 'apk_id' in the CSVs.")
    if apk_col != APK_COL:
        logger.info("Using '{}' as APK column.", apk_col)

    region_col = REGION_COL if REGION_COL in df.columns else ('memory_region' if 'memory_region' in df.columns else None)
    if region_col and region_col != REGION_COL:
        logger.info("Using '{}' as region description column.", region_col)

    addr_col = ADDR_COL if ADDR_COL in df.columns else None
    if not addr_col and 'memory_region' in df.columns:
        def _extract_start_addr(s: str) -> str:
            m = re.search(r'(?:0x)?([0-9a-fA-F]+)', str(s))
            return m.group(1).lower() if m else ""
        df['address'] = df['memory_region'].apply(_extract_start_addr)
        addr_col = 'address'
        logger.info("Created 'address' from 'memory_region'.")

    return apk_col, addr_col, region_col

def normalize_labels_inplace(df: pd.DataFrame) -> None:
    if df[TARGET_COL].dtype == 'object':
        uniq = set(df[TARGET_COL].unique())
        mapping = None
        if      uniq <= {'benign', 'malicious'}: mapping = {'benign': 0, 'malicious': 1}
        elif    uniq <= {'Benign', 'Malicious'}: mapping = {'Benign': 0, 'Malicious': 1}
        elif    uniq <= {'0', '1'}:              mapping = {'0': 0, '1': 1}
        elif    uniq <= {'benign_dumps', 'dumps_dataset'}:
            mapping = {'benign_dumps': 0, 'dumps_dataset': 1}
        if mapping:
            df[TARGET_COL] = df[TARGET_COL].map(mapping).astype(int)
        else:
            logger.info("Target '{}' non-numeric; factorizing.", TARGET_COL)
            df[TARGET_COL] = pd.factorize(df[TARGET_COL])[0]

def _compile_patterns(selection_mode: str,
                      custom_patterns: Optional[List[str]]) -> Optional[List[re.Pattern]]:
    pats = []
    if selection_mode and selection_mode != "all":
        if selection_mode not in SELECTION_MAP:
            raise ValueError(f"Unknown selectionMode '{selection_mode}'")
        pats.extend(SELECTION_MAP[selection_mode])
    if custom_patterns:
        pats.extend(custom_patterns)
    return [re.compile(p) for p in pats] if pats else None

@lru_cache(maxsize=None)
def _read_maps_cached(apk: str) -> Dict[str, str]:
    def _load(path) -> Dict[str,str]:
        d = {}
        if not os.path.exists(path):
            return d
        with open(path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                start = parts[0].split('-')[0].lower()
                if start.startswith("0x"):
                    start = start[2:]
                desc = line.split(None, 5)[-1] if len(line.split(None, 5)) == 6 else ""
                d[start] = desc
        return d
    p_b = os.path.join(ROOT_DIR, BENIGN_DIR_NAME, apk, "maps.txt")
    p_m = os.path.join(ROOT_DIR, MAL_DIR_NAME,    apk, "maps.txt")
    d = _load(p_b)
    if not d:
        d = _load(p_m)
    return d

def normalize_addr_token(x: str) -> str:
    s = str(x).strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    m = re.match(r'^([0-9a-f]+)', s)
    return m.group(1) if m else ""

def filter_rows_by_maps(df: pd.DataFrame,
                        apk_col: str,
                        addr_col: Optional[str],
                        region_col: Optional[str],
                        selection_mode: str,
                        custom_patterns: Optional[List[str]]) -> pd.DataFrame:
    compiled = _compile_patterns(selection_mode, custom_patterns)
    keep = np.zeros(len(df), dtype=bool)
    addr_series = None

    if addr_col and addr_col in df.columns:
        addr_series = df[addr_col].astype(str).map(normalize_addr_token)

    unique_apks = df[apk_col].astype(str).unique().tolist()
    maps_found = sum(1 for a in unique_apks if _read_maps_cached(a))
    logger.info("maps.txt found for {} / {} APKs", maps_found, len(unique_apks))

    fill_region = (region_col is not None and region_col not in df.columns)
    if fill_region:
        df[region_col] = ""

    for apk in unique_apks:
        addr2desc = _read_maps_cached(apk)
        idxs = df.index[df[apk_col].astype(str) == str(apk)]

        if addr_series is not None:
            sub_addrs = addr_series.loc[idxs]
            descs = sub_addrs.map(lambda a: addr2desc.get(a, "")) if addr2desc else pd.Series([""]*len(sub_addrs), index=sub_addrs.index)
            if fill_region:
                df.loc[idxs, region_col] = descs.values
            if compiled is None:
                row_keep = sub_addrs.map(lambda a: a in addr2desc) if addr2desc else pd.Series([False]*len(sub_addrs), index=sub_addrs.index)
            else:
                row_keep = descs.map(lambda d: any(p.search(d) for p in compiled))
            keep[idxs] = row_keep.to_numpy()
        else:
            if region_col and region_col in df.columns:
                descs = df.loc[idxs, region_col].astype(str)
                if compiled is None:
                    keep[idxs] = True
                else:
                    keep[idxs] = descs.apply(lambda d: any(p.search(d) for p in compiled)).to_numpy()
            else:
                keep[idxs] = False

    filtered = df[keep].reset_index(drop=True)
    logger.info("Region selection kept {} / {} rows ({:.1f}%).",
                len(filtered), len(df),
                100.0 * len(filtered) / max(1, len(df)))
    return filtered

def build_apk_level_table(df: pd.DataFrame,
                          apk_col: str,
                          target_col: str,
                          addr_col: Optional[str],
                          region_col: Optional[str]) -> Tuple[pd.DataFrame, List[str]]:
    exclude = set(NON_FEATURE_COLS) | {target_col, apk_col}
    if addr_col:   exclude.add(addr_col)
    if region_col: exclude.add(region_col)
    feat_cols = [c for c in df.columns if c not in exclude]

    num_df = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    agg_df = num_df.groupby(df[apk_col]).mean()

    labels = df.groupby(apk_col)[target_col].agg(lambda s: Counter(s).most_common(1)[0][0])
    agg_df[target_col] = labels.astype(int)

    agg_df = agg_df.reset_index()
    apk_list = agg_df[apk_col].astype(str).tolist()
    return agg_df, apk_list

def get_probs(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        if p.ndim == 2 and p.shape[1] == 1:
            return np.zeros((p.shape[0],), dtype=float)
        if p.ndim == 1:
            return p
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)  # MinMax to [0,1]
        return s
    preds = model.predict(X).astype(float)
    return preds

# ---------- Region-name mapping & misclassification details ----------

REGION_NAME_PRIORITY = [
    "stack", "stack_vdso", "memory_regions",
    "base_apk_stack", "base_apk_memory", "base_apk", "base_vdex", "base_odex",
    "base_odex-apk", "base_vdex-apk", "base_odex-vdex",
    "apk_full_stack", "apk_full_memory", "apk_full",
    "system_full", "system", "libraries",
    "data_stack", "all_data_paths", "complete_memory_regions"
]
_COMPILED_NAME_PATS: Dict[str, List[re.Pattern]] = {
    k: [re.compile(p) for p in SELECTION_MAP[k]] for k in SELECTION_MAP
}

def region_name_from_desc(desc: str) -> str:
    d = str(desc or "")
    for key in REGION_NAME_PRIORITY:
        if key not in _COMPILED_NAME_PATS:
            continue
        if any(p.search(d) for p in _COMPILED_NAME_PATS[key]):
            return key
    return "other"

def evaluate_apk(
    model,
    X_test,
    y_test,
    apk_test,
    model_name="Model",
    df_rows: Optional[pd.DataFrame] = None,
    feat_cols: Optional[List[str]] = None,
    apk_col: Optional[str] = None,
    addr_col: Optional[str] = None,
    region_col: Optional[str] = None,
    best_thr_override: Optional[float] = None,
):
    """Pretty console output + per-region analysis using maps.txt names."""
    prob = get_probs(model, X_test)

    if best_thr_override is not None:
        best_thr = float(best_thr_override)
        auc = ap = None
    else:
        if len(np.unique(y_test)) > 1:
            fpr, tpr, thr = roc_curve(y_test, prob)
            j = tpr - fpr
            best_thr = float(thr[int(np.argmax(j))])
            auc = float(roc_auc_score(y_test, prob))
            ap  = float(average_precision_score(y_test, prob))
        else:
            best_thr = 0.5
            auc = ap = None
            logger.warning("[{}] Single-class test set. Using Thr=0.5; ROC/PR not meaningful.", model_name)

    y_pred = (prob > best_thr).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    if auc is not None and ap is not None:
        logger.info("ROC AUC: {:.4f}, PR AUC: {:.4f}", auc, ap)
    else:
        logger.info("ROC AUC: n/a (single-class), PR AUC: n/a (single-class)")

    # Classification report
    target_names = ["Benign", "Malicious"]
    labels_present = sorted(np.unique(y_test).tolist())
    if len(labels_present) == 2:
        report = classification_report(y_test, y_pred, target_names=target_names)
    else:
        report = classification_report(
            y_test, y_pred, labels=labels_present,
            target_names=[target_names[i] for i in labels_present]
        )
    logger.info("\nClassification Report:\n{}", report)

    # Confusion matrix block
    logger.info("\nConfusion Matrix:\n{}", cm)

    # Misclassified APKs
    mis_idx = np.where(y_test != y_pred)[0]
    if mis_idx.size == 0:
        logger.info("[SUMMARY] APK-level misclassifications -> FP=0, FN=0")
        return

    # Row scoring helper
    def score_rows_block(block_df: pd.DataFrame) -> np.ndarray:
        if block_df.empty or feat_cols is None:
            return np.array([])
        Xr = block_df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        return get_probs(model, Xr)

    def addr_to_maps_desc(apk: str, addr: str) -> str:
        a = normalize_addr_token(addr)
        mp = _read_maps_cached(apk)
        if not mp:
            return ""
        return mp.get(a, "")

    fp_count = 0
    fn_count = 0

    for i in mis_idx:
        apk = apk_test[i]
        yt  = int(y_test[i]); yp = int(y_pred[i]); p = float(prob[i])
        typ = "FP" if (yt == 0 and yp == 1) else "FN"
        if typ == "FP": fp_count += 1
        else: fn_count += 1

        rows_apk = pd.DataFrame()
        if df_rows is not None and apk_col is not None and apk_col in df_rows.columns:
            rows_apk = df_rows[df_rows[apk_col].astype(str) == str(apk)]

        logger.info(
            "[APK_MISCLASS] APK={} | Type={} | True={} | Pred={} | MeanProb={:.4f} | Thr={:.4f} | Regions={}",
            apk, typ, yt, yp, p, best_thr, len(rows_apk)
        )

        if rows_apk.empty:
            continue

        row_probs = score_rows_block(rows_apk)
        if row_probs.size != len(rows_apk):
            continue

        # For FP: list rows with prob >= thr; for FN: prob < thr
        take_mask = (row_probs >= best_thr) if yp == 1 else (row_probs < best_thr)

        for idx, rp, take in zip(rows_apk.index.to_list(), row_probs, take_mask):
            if not take:
                continue
            desc = ""
            if addr_col and addr_col in rows_apk.columns:
                desc = addr_to_maps_desc(apk, rows_apk.loc[idx, addr_col])
            if not desc and region_col and region_col in rows_apk.columns:
                desc = str(rows_apk.loc[idx, region_col]).strip()
            if not desc and "memory_region" in rows_apk.columns:
                desc = str(rows_apk.loc[idx, "memory_region"]).strip()
            if not desc:
                desc = "<unknown>"

            logger.info(
                "[REGION_MISCLASS@THR] APK={} | Region={} | True={} | Pred={} | Prob={:.4f} | Thr={:.4f}",
                apk, desc, yt, yp, float(rp), best_thr
            )

    logger.info("[SUMMARY] APK-level misclassifications -> FP={}, FN={}", fp_count, fn_count)

def run_hyperparameter_tuning(
    X_train, y_train, X_test, y_test, apk_test, *,
    cv_jobs=1,
    df_rows_for_regions: pd.DataFrame,
    feat_cols: List[str],
    apk_col: str,
    addr_col: Optional[str],
    region_col: Optional[str],
    sonify: str,
    selection_mode: str,
    run_dir: str,
):
    """Tunes KNN/SVM/RF/LR/MLP and writes a per-model log file for each."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    param_grids = {
        "KNN": {"n_neighbors": [3, 5, 7], "metric": ["euclidean", "manhattan"]},
        "SVM": {"C": [0.5, 1.0, 2.0], "gamma": [0.01, 0.001], "kernel": ["rbf"]},
        "RandomForest": {"n_estimators": [200, 400], "max_depth": [None, 20], "class_weight": [None]},
        "LogisticRegression": {"C": [0.1, 1.0], "solver": ["saga"], "max_iter": [3000]},
        "MLP": {"hidden_layer_sizes": [(256,), (512, 256)], "max_iter": [500], "activation": ["relu"], "early_stopping": [True]},
    }
    model_objs = {
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "LogisticRegression": LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1),
        "MLP": MLPClassifier(random_state=RANDOM_STATE),
    }
    k_features = min(200, X_train.shape[1])

    for name, clf in model_objs.items():
        with with_model_log(run_dir, sonify, selection_mode, name):
            logger.info("=== Tuning {} ===", name)
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(mutual_info_classif, k=k_features)),
                ('classifier', clf),
            ])
            search = RandomizedSearchCV(
                pipe,
                param_distributions={f'classifier__{k}': v for k, v in param_grids[name].items()},
                n_iter=min(5, sum(len(v) for v in param_grids[name].values())),
                scoring='roc_auc',
                cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                n_jobs=cv_jobs,
                refit=True,
                random_state=RANDOM_STATE,
                verbose=1,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            out = os.path.join(MODEL_DIR, name.replace(" ", "") + ".pkl")
            with open(out, 'wb') as f:
                pickle.dump(best, f)
            logger.info("✅ Saved {} to {}", name, out)

            evaluate_apk(
                best, X_test, y_test, apk_test,
                model_name=name,
                df_rows=df_rows_for_regions,
                feat_cols=[c for c in feat_cols],
                apk_col=apk_col, addr_col=addr_col, region_col=region_col
            )

def run_voting_ensemble(
    X_train, y_train, X_test, y_test, apk_test,
    df_rows_for_regions: pd.DataFrame,
    feat_cols: List[str],
    apk_col: str,
    addr_col: Optional[str],
    region_col: Optional[str],
    sonify: str,
    selection_mode: str,
    run_dir: str,
):
    """Build a Pipeline(scaler+selector+VotingClassifier), log to its own file, evaluate."""
    def load_model(name_no_space):
        path = os.path.join(MODEL_DIR, name_no_space + ".pkl")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing tuned model: {path}")
        with open(path, 'rb') as f:
            return pickle.load(f)

    with with_model_log(run_dir, sonify, selection_mode, "VotingEnsemble"):
        # Load tuned pipelines to reuse scaler+selector
        lr  = load_model('LogisticRegression')
        knn = load_model('KNN')
        svm = load_model('SVM')
        rf  = load_model('RandomForest')
        mlp = load_model('MLP')

        scaler   = lr.named_steps['scaler']
        selector = lr.named_steps['selector']

        ensemble = VotingClassifier(
            estimators=[
                ('knn', knn.named_steps['classifier']),
                ('svm', svm.named_steps['classifier']),
                ('rf',  rf .named_steps['classifier']),
                ('lr',  lr .named_steps['classifier']),
                ('mlp', mlp.named_steps['classifier']),
            ],
            voting='soft'
        )

        # Wrap in a pipeline so evaluate_apk can score per-row features consistently
        ens_pipe = Pipeline([
            ('scaler', scaler),
            ('selector', selector),
            ('classifier', ensemble),
        ])
        ens_pipe.fit(X_train, y_train)

        evaluate_apk(
            ens_pipe, X_test, y_test, apk_test,
            model_name="VotingEnsemble",
            df_rows=df_rows_for_regions,
            feat_cols=[c for c in feat_cols],
            apk_col=apk_col, addr_col=addr_col, region_col=region_col
        )

def run_boosting_models(
    X_train, y_train, X_test, y_test, apk_test, *,
    df_rows_for_regions: pd.DataFrame,
    feat_cols: List[str],
    apk_col: str,
    addr_col: Optional[str],
    region_col: Optional[str],
    sonify: str,
    selection_mode: str,
    run_dir: str,
):
    """Train XGBoost/LightGBM/CatBoost if available; each model logs to its own file."""
    models = {}
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(
            eval_metric='logloss', tree_method='hist',
            random_state=RANDOM_STATE, n_estimators=400
        )
    except Exception as e:
        logger.info("XGBoost not available: {}", e)

    try:
        from lightgbm import LGBMClassifier
        models['LightGBM'] = LGBMClassifier(random_state=RANDOM_STATE)
    except Exception as e:
        logger.info("LightGBM not available: {}", e)

    try:
        from catboost import CatBoostClassifier
        models['CatBoost'] = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
    except Exception as e:
        logger.info("CatBoost not available: {}", e)

    if not models:
        logger.info("No boosting libraries available; skipping boosting stage.")
        return

    k_features = min(200, X_train.shape[1])

    for name, base_model in models.items():
        with with_model_log(run_dir, sonify, selection_mode, name):
            logger.info("=== Training {} ===", name)
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(mutual_info_classif, k=k_features)),
                ('classifier', base_model),
            ])
            pipe.fit(X_train, y_train)
            evaluate_apk(
                pipe, X_test, y_test, apk_test,
                model_name=name,
                df_rows=df_rows_for_regions,
                feat_cols=[c for c in feat_cols],
                apk_col=apk_col, addr_col=addr_col, region_col=region_col
            )


class SampleCNN(nn.Module):
    DROPOUT = 0.5
    def __init__(self, numClasses: int = 2):
        super().__init__()
        # Fewer pools + AdaptiveAvgPool1d to be length-agnostic
        self.conv1 = nn.Conv1d(1,   128, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm1d(256)

        self.conv5 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5   = nn.BatchNorm1d(512)

        self.pool  = nn.MaxPool1d(3, stride=2, padding=1)  # gentle downsampling
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.drop  = nn.Dropout(self.DROPOUT)
        self.fc    = nn.Linear(512, numClasses)

    def forward(self, x):
        # x: [B, C(=1), L]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.drop(x)

        x = self.gap(x).squeeze(-1)   # [B, 512]
        x = self.fc(x)                # [B, numClasses]
        return x


class TorchProbWrapper:
    """
    Minimal sklearn-like estimator for a PyTorch 1D CNN.

    Exposes: fit(X, y), predict_proba(X) -> np.array[N, 2]
    Accepts tabular X: [N, D] -> reshaped to [N, 1, D]
    """
    def __init__(self,
                 model_ctor=lambda: SampleCNN(numClasses=2),
                 epochs: int = 15,
                 batch_size: int = 64,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 device: Optional[str] = None,
                 seed: int = 42):
        self.model_ctor = model_ctor
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.model = None

    def _prepare(self, X, y=None):
        # X is [N, D] -> torch [N, 1, D]
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X).unsqueeze(1)  # add channel
        y_t = None
        if y is not None:
            y = np.asarray(y, dtype=np.int64)
            y_t = torch.from_numpy(y)
        return X_t, y_t

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        num_classes = len(np.unique(y))
        self.model = self.model_ctor()
        # if model factory didn't set numClasses correctly, patch last layer:
        if isinstance(self.model.fc, nn.Linear) and self.model.fc.out_features != num_classes:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.model.to(self.device)

        X_t, y_t = self._prepare(X, y)
        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            running = 0.0
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)
            avg_loss = running / len(ds)
            logger.info("[TorchCNN] epoch {}/{} - loss {:.4f}", epoch+1, self.epochs, avg_loss)

        return self

    @torch.no_grad()
    def predict_proba(self, X):
        assert self.model is not None, "Call fit() first."
        self.model.eval()
        X_t, _ = self._prepare(X, None)
        dl = DataLoader(TensorDataset(X_t), batch_size=self.batch_size, shuffle=False)
        probs = []
        for (xb,) in dl:
            xb = xb.to(self.device)
            logits = self.model(xb)
            pb = torch.softmax(logits, dim=1)  # [B, C]
            probs.append(pb.cpu().numpy())
        probs = np.concatenate(probs, axis=0)
        # If binary, ensure shape [:, 2]
        if probs.shape[1] == 1:
            probs = np.hstack([1.0 - probs, probs])
        return probs

    @torch.no_grad()
    def decision_function(self, X):
        # Optional; not required since evaluate_apk uses predict_proba first
        probs = self.predict_proba(X)
        if probs.shape[1] >= 2:
            return probs[:, 1]
        return probs.squeeze(-1)


def run_cnn_1d_model(
    X_train, y_train, X_test, y_test, apk_test, *,
    df_rows_for_regions: pd.DataFrame,
    feat_cols: List[str],
    apk_col: str,
    addr_col: Optional[str],
    region_col: Optional[str],
    sonify: str,
    selection_mode: str,
    run_dir: str,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    """
    Trains a 1D CNN on standardized APK-level features, reshaped to [N, 1, D].
    """
    from sklearn.preprocessing import StandardScaler

    with with_model_log(run_dir, sonify, selection_mode, "CNN1D"):
        logger.info("=== Training {} ===", "CNN1D")

        # Standardize features (important for stable training)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)

        # Build and train the wrapper
        clf = TorchProbWrapper(
            model_ctor=lambda: SampleCNN(numClasses=len(np.unique(y_train))),
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
        )
        clf.fit(Xtr, y_train)

        # Evaluate using your existing helper
        evaluate_apk(
            clf, Xte, y_test, apk_test,
            model_name="CNN1D",
            df_rows=df_rows_for_regions,
            feat_cols=[c for c in feat_cols],
            apk_col=apk_col, addr_col=addr_col, region_col=region_col
        )



# -------------------- MAIN --------------------

def main_classify(
    selected_memoryregion: str = "data_stack",
    selected_sonification: str = "wav8",
    *,
    csv_benign: Optional[str] = None,
    csv_malware: Optional[str] = None,
    cv_jobs: int = 1,
    run_ensemble: bool = True,
    run_boosting: bool = True,
):
    """
    Run APK-level classification for the given memory region selection and sonification type.

    Example:
        main("data_stack", "wav8")
        main("base_apk", "mp3", run_ensemble=False, run_boosting=False)
    """
    sonify         = selected_sonification
    selection_mode = selected_memoryregion

    # ---- Logging setup (per run) ----
    run_dir, _stderr_sink = setup_run_logging(sonify, selection_mode)
    os.makedirs(RUNS_DIR, exist_ok=True)

    # ---- Load CSVs ----
    df_b, df_m, _, _ = load_benign_and_mal_csvs(sonify, CSV_SEARCH_ROOT, csv_benign, csv_malware)

    # Optional sanity filter by sonify column
    if SONIFY_COL in df_b.columns:
        before = len(df_b)
        df_b = df_b[df_b[SONIFY_COL].astype(str).str.lower() == sonify.lower()].reset_index(drop=True)
        logger.info("[benign] Filtered by {}='{}': kept {} / {} rows.", SONIFY_COL, sonify, len(df_b), before)
    if SONIFY_COL in df_m.columns:
        before = len(df_m)
        df_m = df_m[df_m[SONIFY_COL].astype(str).str.lower() == sonify.lower()].reset_index(drop=True)
        logger.info("[malware] Filtered by {}='{}': kept {} / {} rows.", SONIFY_COL, sonify, len(df_m), before)

    df = pd.concat([df_b, df_m], ignore_index=True)
    logger.info("Combined rows: {}", len(df))
    logger.info("CSV columns: {}", list(df.columns))

    apk_col, addr_col, region_col = ensure_core_columns(df)
    normalize_labels_inplace(df)

    # maps-based row filtering
    df_filtered = filter_rows_by_maps(
        df=df,
        apk_col=apk_col,
        addr_col=addr_col,
        region_col=region_col,
        selection_mode=selection_mode,
        custom_patterns=None
    )
    if df_filtered.empty:
        logger.error("After region selection, no rows remain. Adjust selectionMode or verify maps/columns.")
        raise SystemExit(1)

    # APK-level aggregation (mean features)
    apk_df, _ = build_apk_level_table(df_filtered, apk_col, TARGET_COL, addr_col, region_col)

    # class balance at APK level
    apk_counts = Counter(apk_df[TARGET_COL].tolist())
    logger.info("APK-level class balance (0=Benign, 1=Malicious): {}", dict(apk_counts))
    if len(apk_counts) < 2:
        logger.warning("APK-level data collapsed to a single class ({}). Results will be limited.", list(apk_counts.keys())[0])

    # Train/test split by APK
    apks = apk_df[apk_col].astype(str).values
    y_apk = apk_df[TARGET_COL].values
    try:
        apks_train, apks_test = train_test_split(
            apks,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_apk if len(np.unique(y_apk)) > 1 else None
        )
    except ValueError:
        apks_train, apks_test = train_test_split(apks, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    train_df = apk_df[apk_df[apk_col].astype(str).isin(apks_train)].reset_index(drop=True)
    test_df  = apk_df[apk_df[apk_col].astype(str).isin(apks_test)].reset_index(drop=True)

    logger.info("APK samples -> Train: {}, Test: {}", len(train_df), len(test_df))
    logger.info("Train class balance: {}", dict(Counter(train_df[TARGET_COL].tolist())))
    logger.info("Test  class balance: {}", dict(Counter(test_df[TARGET_COL].tolist())))

    # Build feature matrices at APK level
    exclude = set(NON_FEATURE_COLS) | {TARGET_COL, apk_col}
    feat_cols = [c for c in train_df.columns if c not in exclude]
    X_train = train_df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    y_train = train_df[TARGET_COL].values
    X_test  = test_df [feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    y_test  = test_df [TARGET_COL].values
    apk_test = test_df[apk_col].astype(str).tolist()
    df_rows_for_regions = df_filtered  # pre-aggregation, maps-filtered rows

    # Base models (always)
    run_hyperparameter_tuning(
        X_train, y_train, X_test, y_test, apk_test,
        cv_jobs=cv_jobs,
        df_rows_for_regions=df_rows_for_regions,
        feat_cols=feat_cols,
        apk_col=apk_col,
        addr_col=addr_col,
        region_col=region_col,
        sonify=sonify,
        selection_mode=selection_mode,
        run_dir=run_dir,
    )

    # Voting ensemble (default: on)
    if run_ensemble:
        try:
            run_voting_ensemble(
                X_train, y_train, X_test, y_test, apk_test,
                df_rows_for_regions=df_rows_for_regions,
                feat_cols=feat_cols,
                apk_col=apk_col, addr_col=addr_col, region_col=region_col,
                sonify=sonify, selection_mode=selection_mode, run_dir=run_dir
            )
        except Exception as e:
            logger.warning("Voting ensemble skipped/failed: {}", e)

    # Boosting (default: on)
    if run_boosting:
        run_boosting_models(
            X_train, y_train, X_test, y_test, apk_test,
            df_rows_for_regions=df_rows_for_regions,
            feat_cols=feat_cols,
            apk_col=apk_col, addr_col=addr_col, region_col=region_col,
            sonify=sonify, selection_mode=selection_mode, run_dir=run_dir
        )


    # CNN (optional but recommended to see how it compares)
    try:
        run_cnn_1d_model(
            X_train, y_train, X_test, y_test, apk_test,
            df_rows_for_regions=df_rows_for_regions,
            feat_cols=feat_cols,
            apk_col=apk_col, addr_col=addr_col, region_col=region_col,
            sonify=sonify, selection_mode=selection_mode, run_dir=run_dir,
            epochs=20, batch_size=64, lr=1e-3, weight_decay=1e-4
        )
    except Exception as e:
        logger.warning("CNN1D skipped/failed: {}", e)

    # Optional plots (example with LR if saved)
    if SAVE_PLOTS and len(np.unique(y_test)) > 1:
        try:
            with open(os.path.join(MODEL_DIR, "LogisticRegression.pkl"), "rb") as f:
                lr = pickle.load(f)
            # LR here is a Pipeline; get_probs handles both pipeline and clf
            prob = get_probs(lr, X_test)
            fpr, tpr, _ = roc_curve(y_test, prob)
            fig = plt.figure(figsize=(5,5))
            plt.plot(fpr, tpr, label=f"LR AUC={roc_auc_score(y_test, prob):.3f}")
            plt.plot([0,1],[0,1],"--")
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("APK-level ROC (LR)")
            plt.legend(); plt.tight_layout()
            plot_path = os.path.join(run_dir, "roc_lr.png")
            plt.savefig(plot_path, dpi=150); plt.close(fig)
            logger.info("Plot saved: {}", plot_path)
        except Exception as e:
            logger.info("Plotting skipped: {}", e)
if __name__ == "__main__":
    try:
        # Default call; change here or import main elsewhere.
        main_classify("data_stack", "wav8")
    except Exception as e:
        logger.opt(exception=True).critical("Critical error: {}", e)
