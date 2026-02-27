#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
APK-level classification from audio-feature CSVs (benign+malware), with maps.txt-driven region selection.
- Loads TWO CSVs: benign_dumps_<sonify>_features.csv and dumps_dataset_<sonify>_features.csv
- Robust CSV loading (skips malformed lines).
- Filters rows by selectionMode using descriptions from <apk>/maps.txt.
- Aggregates features to one vector per APK (mean).
- Trains & evaluates at APK level.
- Logs everything; does NOT write CSV/TXT outputs.

Usage:
  python main_classify.py --sonify {wav8,wav16,mp3,opus,flac,midi} [--selectionMode data_stack]
  # optional explicit CSVs:
  python main_classify.py --sonify wav8 --csvBenign /path/benign_dumps_wav8_features.csv --csvMalware /path/dumps_dataset_wav8_features.csv
"""

import argparse
import ast
import csv
import logging
import os
import re
from glob import glob
from functools import lru_cache
from collections import Counter
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.errors import ParserError
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

from implementations.classifier_rf import RandomForestTrainer

# -------------------- CONFIG DEFAULTS --------------------

# Folder that contains your CSVs (we search here if explicit paths not provided)
CSV_SEARCH_ROOT = "/home/ssanna/Desktop/malware_ram/Android/audio/malware-sonification-signal-processing"

# Dataset root containing benign_dumps/ and dumps_dataset/ with per-APK folders and maps.txt
ROOT_DIR        = "/mnt/malware_ram/Android"
BENIGN_DIR_NAME = "benign_dumps"
MAL_DIR_NAME    = "dumps_dataset"

# Column defaults (auto-detected if missing)
TARGET_COL = "category"
APK_COL    = "apk"            # will fallback to 'apk_id'
ADDR_COL   = "address"        # if missing, derived from 'memory_region'
REGION_COL = "region_desc"    # will fallback to 'memory_region'
SONIFY_COL = "sonify_type"    # filtered to --sonify if present

# Feature columns to ignore
NON_FEATURE_COLS = {
    'filename', 'file', 'path', 'id', 'name',
    'apk', 'apk_id',  # identity columns (we remove the one we actually use too)
    'sample_rate', 'num_samples', 'duration_sec', 'duration',
    'length', 'size', 'timestamp', 'date',
    'num_regions', 'region_types', 'dl_embedding_dim'
}

# Selection presets (mirrors your PyTorch script)
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

# Training/eval defaults
TEST_SIZE      = 0.20
RANDOM_STATE   = 42
AGG_FEATURES   = "mean"   # aggregation for features per APK
MODEL_PATH     = "models/model.joblib"
SAVE_PLOTS     = False
OUT_DIR        = "./runs/apk_eval"

# -------------------- LOGGING --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------- HELPERS --------------------

def _parse_array_string(s):
    try:
        return ast.literal_eval(str(s))
    except (ValueError, SyntaxError):
        return np.fromstring(str(s).strip(' []'), sep=' ')

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
        logging.warning(f"Standard read_csv failed for {path}: {e}")
        try:
            df = pd.read_csv(
                path,
                engine="python",
                on_bad_lines="skip",
                quoting=csv.QUOTE_MINIMAL,
                quotechar='"',
                escapechar='\\',
            )
            logging.warning(f"Loaded {len(df):,} rows from {path} with on_bad_lines='skip'.")
            return df
        except TypeError:
            # legacy pandas fallback
            df = pd.read_csv(
                path,
                engine="python",
                error_bad_lines=False,
                warn_bad_lines=True,
                quoting=csv.QUOTE_MINIMAL,
                quotechar='"',
                escapechar='\\',
            )
            logging.warning(f"Loaded {len(df):,} rows from {path} with error_bad_lines=False (legacy).")
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

    logging.info(f"Using benign CSV:  {benign_path}")
    logging.info(f"Using malware CSV: {mal_path}")

    df_b = read_csv_robust(benign_path)
    df_m = read_csv_robust(mal_path)

    # Ensure target column exists with recognizable labels
    if TARGET_COL not in df_b.columns:
        df_b[TARGET_COL] = "benign_dumps"
        logging.info(f"Injected '{TARGET_COL}'='benign_dumps' into benign CSV.")
    if TARGET_COL not in df_m.columns:
        df_m[TARGET_COL] = "dumps_dataset"
        logging.info(f"Injected '{TARGET_COL}'='dumps_dataset' into malware CSV.")

    return df_b, df_m, benign_path, mal_path

def ensure_core_columns(df: pd.DataFrame) -> Tuple[str, Optional[str], Optional[str]]:
    """Return (apk_col, addr_col, region_col); create addr/region if needed."""
    apk_col = APK_COL if APK_COL in df.columns else ('apk_id' if 'apk_id' in df.columns else None)
    if not apk_col:
        raise SystemExit("No APK column found. Expected 'apk' or 'apk_id' in the CSVs.")
    if apk_col != APK_COL:
        logging.info(f"Using '{apk_col}' as APK column.")

    region_col = REGION_COL if REGION_COL in df.columns else ('memory_region' if 'memory_region' in df.columns else None)
    if region_col and region_col != REGION_COL:
        logging.info(f"Using '{region_col}' as region description column.")

    addr_col = ADDR_COL if ADDR_COL in df.columns else None
    if not addr_col and 'memory_region' in df.columns:
        def _extract_start_addr(s: str) -> str:
            m = re.search(r'(?:0x)?([0-9a-fA-F]+)', str(s))
            return m.group(1).lower() if m else ""
        df['address'] = df['memory_region'].apply(_extract_start_addr)
        addr_col = 'address'
        logging.info("Created 'address' from 'memory_region'.")

    return apk_col, addr_col, region_col

def normalize_labels_inplace(df: pd.DataFrame) -> None:
    """Map category → {0,1} if possible; else factorize."""
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
            logging.info(f"Target '{TARGET_COL}' non-numeric; factorizing.")
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
    """Return dict start_addr(hex lowercase no '0x') -> description from maps.txt (tries benign, then malware)."""
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
    """Keep only rows whose address description in maps.txt matches the selection patterns."""
    compiled = _compile_patterns(selection_mode, custom_patterns)
    keep = np.zeros(len(df), dtype=bool)
    addr_series = None

    if addr_col and addr_col in df.columns:
        addr_series = df[addr_col].astype(str).map(normalize_addr_token)

    # Count maps availability for info
    unique_apks = df[apk_col].astype(str).unique().tolist()
    maps_found = 0
    for apk in unique_apks:
        if _read_maps_cached(apk):
            maps_found += 1
    logging.info(f"maps.txt found for {maps_found} / {len(unique_apks)} APKs")

    # Fill missing region_col from maps if needed:
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
    logging.info("Region selection kept %d / %d rows (%.1f%%).",
                 len(filtered), len(df),
                 100.0 * len(filtered) / max(1, len(df)))
    return filtered

def build_apk_level_table(df: pd.DataFrame,
                          apk_col: str,
                          target_col: str,
                          addr_col: Optional[str],
                          region_col: Optional[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Aggregate row features to one vector per APK (mean)."""
    exclude = set(NON_FEATURE_COLS) | {target_col, apk_col}
    if addr_col:   exclude.add(addr_col)
    if region_col: exclude.add(region_col)
    feat_cols = [c for c in df.columns if c not in exclude]

    # numeric-only for safety
    num_df = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    agg_df = num_df.groupby(df[apk_col]).mean()

    # label per APK by majority vote in filtered rows
    labels = df.groupby(apk_col)[target_col].agg(lambda s: Counter(s).most_common(1)[0][0])
    agg_df[target_col] = labels.astype(int)

    # preserve apk ids as a column too
    agg_df = agg_df.reset_index()
    apk_list = agg_df[apk_col].astype(str).tolist()
    return agg_df, apk_list

def get_positive_probs(trainer, X, positive_label=1) -> np.ndarray:
    """Return P(y==positive_label). Handles single-class models/ordering."""
    if hasattr(trainer, "predict_proba"):
        proba = trainer.predict_proba(X)
        classes_ = getattr(trainer, "classes_", None)
    elif hasattr(trainer, "model") and hasattr(trainer.model, "predict_proba"):
        proba = trainer.model.predict_proba(X)
        classes_ = getattr(trainer.model, "classes_", None)
    else:
        raise AttributeError("Trainer/model must expose predict_proba.")
    proba = np.asarray(proba)
    if proba.ndim == 1:  # some models
        return proba
    if proba.shape[1] == 1 or classes_ is None or positive_label not in classes_:
        if classes_ is not None and len(classes_) == 1 and classes_[0] == positive_label:
            return np.ones((proba.shape[0],), dtype=float)
        return np.zeros((proba.shape[0],), dtype=float)
    pos_idx = int(np.where(classes_ == positive_label)[0][0])
    return proba[:, pos_idx]

# -------------------- MAIN --------------------

def main():
    parser = argparse.ArgumentParser(description="APK-level classifier (maps.txt region selection, logs-only).")
    parser.add_argument("--sonify", required=True, choices=["wav8", "wav16", "mp3", "opus", "flac", "midi"],
                        help="Which sonification CSV pair to use.")
    parser.add_argument("--selectionMode", default="data_stack",
                        help=f"Region preset: one of {', '.join(sorted(SELECTION_MAP.keys()))} or 'all'.")
    parser.add_argument("--csvBenign", default=None, help="Optional explicit benign CSV path.")
    parser.add_argument("--csvMalware", default=None, help="Optional explicit malware CSV path.")
    args = parser.parse_args()

    # Load two CSVs robustly
    df_b, df_m, benign_path, mal_path = load_benign_and_mal_csvs(args.sonify, CSV_SEARCH_ROOT, args.csvBenign, args.csvMalware)

    # Filter to requested sonify type if the column exists (defensive)
    if SONIFY_COL in df_b.columns:
        before = len(df_b); df_b = df_b[df_b[SONIFY_COL].astype(str).str.lower() == args.sonify.lower()].reset_index(drop=True)
        logging.info(f"[benign] Filtered by {SONIFY_COL}='{args.sonify}': kept {len(df_b)} / {before} rows.")
    if SONIFY_COL in df_m.columns:
        before = len(df_m); df_m = df_m[df_m[SONIFY_COL].astype(str).str.lower() == args.sonify.lower()].reset_index(drop=True)
        logging.info(f"[malware] Filtered by {SONIFY_COL}='{args.sonify}': kept {len(df_m)} / {before} rows.")

    # Combine
    df = pd.concat([df_b, df_m], ignore_index=True)
    logging.info("Combined rows: %d", len(df))
    logging.info("CSV columns: %s", list(df.columns))

    # Ensure core columns exist / create address if needed
    apk_col, addr_col, region_col = ensure_core_columns(df)

    # Normalize labels
    normalize_labels_inplace(df)

    # maps-based region selection
    df_filtered = filter_rows_by_maps(
        df=df,
        apk_col=apk_col,
        addr_col=addr_col,
        region_col=region_col,
        selection_mode=args.selectionMode,
        custom_patterns=None
    )
    if df_filtered.empty:
        logging.error("After region selection, no rows remain. Adjust selectionMode or verify maps/columns.")
        raise SystemExit(1)

    # APK-level aggregation (features → mean per APK)
    apk_df, apk_list = build_apk_level_table(df_filtered, apk_col, TARGET_COL, addr_col, region_col)

    # Class balance at APK level
    apk_counts = Counter(apk_df[TARGET_COL].tolist())
    logging.info("APK-level class balance (0=Benign, 1=Malicious): %s", dict(apk_counts))
    if len(apk_counts) < 2:
        logging.warning("APK-level data collapsed to a single class (%s). Results will be limited.", list(apk_counts.keys())[0])

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

    logging.info("APK samples -> Train: %d, Test: %d", len(train_df), len(test_df))
    logging.info("Train class balance: %s", dict(Counter(train_df[TARGET_COL].tolist())))
    logging.info("Test  class balance: %s", dict(Counter(test_df[TARGET_COL].tolist())))

    # Build feature matrices at APK level
    exclude = set(NON_FEATURE_COLS) | {TARGET_COL, apk_col}
    feat_cols = [c for c in train_df.columns if c not in exclude]
    X_train = train_df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    y_train = train_df[TARGET_COL].values
    X_test  = test_df [feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    y_test  = test_df [TARGET_COL].values

    # Train RF
    trainer = RandomForestTrainer()
    trainer.train(X_train, y_train)
    trainer.saveModel(MODEL_PATH)
    logging.info("Model saved to %s", MODEL_PATH)

    # Probabilities (positive class = 1)
    prob_test = get_positive_probs(trainer, X_test, positive_label=1)

    # Threshold
    if len(np.unique(y_test)) > 1:
        fpr, tpr, thr = roc_curve(y_test, prob_test)
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        best_thr = float(thr[best_idx])
        row_auc  = float(roc_auc_score(y_test, prob_test))
        row_pr   = float(average_precision_score(y_test, prob_test))
        logging.info("APK ROC-AUC=%.4f | PR-AUC=%.4f | BestThr=%.4f", row_auc, row_pr, best_thr)
    else:
        best_thr = 0.5
        logging.warning("Test set contains a single class (%d). Using threshold=0.5 and limited metrics.", y_test[0])

    # Predict
    y_pred = (prob_test > best_thr).astype(int)
    acc = 100.0 * float(np.mean(y_pred == y_test))
    logging.info("APK-level Accuracy = %.2f%% (threshold=%.3f)", acc, best_thr)

    # Reports
    labels_present = sorted(list(np.unique(y_test)))
    target_names = ["Benign", "Malicious"]
    try:
        if len(labels_present) == 2:
            report = classification_report(y_test, y_pred, target_names=target_names)
        else:
            # limited report
            report = classification_report(y_test, y_pred, labels=labels_present,
                                           target_names=[target_names[i] for i in labels_present])
        logging.info("APK-level Classification Report:\n%s", report)
    except Exception as e:
        logging.warning("Could not compute full classification_report: %s", e)

    try:
        cm = confusion_matrix(y_test, y_pred, labels=[0,1])
        logging.info("APK-level Confusion Matrix:\n%s", cm)
    except Exception as e:
        logging.warning("Could not compute confusion_matrix: %s", e)

    # Per-APK predictions
    logging.info("Per-APK predictions (apk, true, pred, prob):")
    for apk, yt, yp, pr in zip(test_df[apk_col].astype(str).tolist(), y_test.tolist(), y_pred.tolist(), prob_test.tolist()):
        logging.info("%s\ttrue=%d\tpred=%d\tprob=%.4f", apk, yt, yp, pr)

    # Misclassified
    mis = [(a, yt, yp, pr) for a, yt, yp, pr in zip(test_df[apk_col].astype(str), y_test, y_pred, prob_test) if yt != yp]
    logging.info("Total Misclassified APKs: %d / %d", len(mis), len(test_df))
    for a, yt, yp, pr in mis:
        logging.info("[MISCLASSIFIED APK] %s | True=%d | Pred=%d | Prob=%.4f", a, yt, yp, pr)

    # Optional plots (disabled by default)
    if SAVE_PLOTS and len(np.unique(y_test)) > 1:
        os.makedirs(OUT_DIR, exist_ok=True)
        # Confusion matrix
        fig = plt.figure(figsize=(5,5))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, labels=[0,1]),
                               display_labels=["Benign","Malicious"]).plot(cmap="Blues", values_format="d")
        plt.title("APK-level Confusion Matrix"); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150); plt.close(fig)
        # ROC
        fpr, tpr, _ = roc_curve(y_test, prob_test)
        fig = plt.figure(figsize=(5,5))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, prob_test):.3f}")
        plt.plot([0,1],[0,1],"--"); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title("APK-level ROC"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "roc_apk.png"), dpi=150); plt.close(fig)
        # PR
        prec, rec, _ = precision_recall_curve(y_test, prob_test)
        fig = plt.figure(figsize=(5,5))
        plt.plot(rec, prec, label=f"AP={average_precision_score(y_test, prob_test):.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("APK-level PR")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "pr_apk.png"), dpi=150); plt.close(fig)
        logging.info("Plots saved to %s", OUT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical("Critical error: %s", e)
        import traceback
        traceback.print_exc()
