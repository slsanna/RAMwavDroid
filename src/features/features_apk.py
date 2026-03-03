#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import shutil

# Make project importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from extract_features import main as extract_features_main

# --------- PATHS ----------
AUDIO_ROOT = Path("PATH")
FEATURES_ROOT = Path("PATH")
STAGING_ROOT = FEATURES_ROOT / "_staging_wav_dataset"  # symlink-only staging for extract_features

FINAL_CSV = {
    "benign": FEATURES_ROOT / "benign_apk_features.csv",
    "malware": FEATURES_ROOT / "malware_apk_features.csv",
}

# -------- SETTINGS --------
SONIFY_TYPE = os.environ.get("SONIFY_TYPE", "wav8")  # must match your sonification
ANALYSIS_TYPE = "dynamic" #"dynamic"
FEATURE_TYPE = "manual" #"wav2vec2"
WORKERS = "8"

# Per-group limits (set to None to process all files in each group)
LIMIT_BENIGN: Optional[int] = 500
LIMIT_MALWARE: Optional[int] = 500

# Clean the staging tree after finishing?
CLEAN_STAGING = True

def list_wavs(category: str) -> List[Path]:
    cat_dir = AUDIO_ROOT / category
    if not cat_dir.exists():
        return []
    return sorted(cat_dir.glob("*.wav"))

def stage_category(category: str, limit: Optional[int]) -> Path:
    """
    Create a dataset-like tree with symlinks:
      STAGING_ROOT/<category>/<apk_id>/audio/<SONIFY_TYPE>/<original_wav>
    apk_id is derived from the wav basename (before .wav).
    """
    cat_staging = STAGING_ROOT / category
    # fresh staging for this category
    if cat_staging.exists():
        shutil.rmtree(cat_staging)
    cat_staging.mkdir(parents=True, exist_ok=True)

    wavs = list_wavs(category)
    if limit is not None:
        wavs = wavs[:limit]

    pbar = tqdm(wavs, desc=f"Staging {category} WAVs", unit="wav")
    for wav in pbar:
        apk_id = wav.stem  # filename without extension, e.g., the hash
        target_dir = cat_staging / apk_id / "audio" / SONIFY_TYPE
        target_dir.mkdir(parents=True, exist_ok=True)
        link_path = target_dir / wav.name
        try:
            # remove stale link/file if present
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            os.symlink(wav, link_path)
        except FileExistsError:
            pass
        except Exception as e:
            tqdm.write(f"[WARN] Could not symlink {wav} -> {link_path}: {e}")
    return STAGING_ROOT

def find_group_csvs(output_dir: Path, category: str) -> List[Path]:
    """
    Find CSVs produced for a category.
    Priority:
      1) <output_dir>/<category>/**/*.csv
      2) <output_dir>/*{category}*.csv
      3) <output_dir>/*.csv
    """
    candidates: List[Path] = []
    cat_subdir = output_dir / category
    if cat_subdir.exists():
        candidates = [p for p in cat_subdir.rglob("*.csv")]
        if candidates:
            return sorted(candidates)
    root_csvs = [p for p in output_dir.glob("*.csv")]
    named = [p for p in root_csvs if category in p.stem.lower()]
    if named:
        return sorted(named)
    if root_csvs:
        return sorted(root_csvs)
    return []

def merge_csvs(csv_paths: List[Path], out_path: Path) -> None:
    """
    Merge multiple CSVs into one. Keeps header from the first file.
    """
    if not csv_paths:
        print(f"[WARN] No CSVs to merge for {out_path.name}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written_header = False
    with open(out_path, "w", newline="") as w:
        for p in tqdm(csv_paths, desc=f"Merging into {out_path.name}", unit="file"):
            try:
                with open(p, "r", newline="") as r:
                    first_line = True
                    for line in r:
                        if first_line:
                            if not written_header:
                                w.write(line)
                                written_header = True
                            first_line = False
                        else:
                            w.write(line)
            except Exception as e:
                print(f"[WARN] Skipping unreadable CSV {p}: {e}")

def run_group(category: str, limit: Optional[int]):
    wavs = list_wavs(category)
    if not wavs:
        print(f"[WARN] No WAV files in {AUDIO_ROOT / category}")
        return
    print(f"[INFO] {category}: {len(wavs)} WAVs found; limit={limit if limit is not None else 'ALL'}")

    # 1) Stage dataset-like view for this category
    data_root = stage_category(category, limit)

    # 2) Build argv for extract_features, pointing to staging
    argv = [
        "extract_features.py",
        "--data-dir", str(data_root),
        "--categories", category,                 # now there ARE per-APK folders
        "--sonify", SONIFY_TYPE,
        "--analysis-type", ANALYSIS_TYPE,
        "--feature-type", FEATURE_TYPE,
        "--workers", WORKERS,
        "--output-dir", str(FEATURES_ROOT),
        "--verbose",
    ]
    # We already enforced the limit by staging only N WAVs; no need to pass --limit
    print("[INFO] Running extract_features with args:")
    print(" ", " ".join(argv))

    sys.argv = argv
    extract_features_main()

    # 3) Collect CSVs and finalize to deterministic names
    csvs = find_group_csvs(FEATURES_ROOT, category)
    if not csvs:
        print(f"[WARN] No CSVs produced for {category} under {FEATURES_ROOT}")
    else:
        final_path = FINAL_CSV[category]
        if len(csvs) == 1:
            try:
                if final_path.exists():
                    final_path.unlink()
                csvs[0].replace(final_path)
                print(f"[INFO] Final CSV -> {final_path}")
            except Exception as e:
                print(f"[WARN] Rename failed ({e}); falling back to merge).")
                merge_csvs(csvs, final_path)
        else:
            merge_csvs(csvs, final_path)
            print(f"[INFO] Final CSV -> {final_path}")

    # 4) Cleanup staging for this category
    if CLEAN_STAGING:
        try:
            shutil.rmtree(STAGING_ROOT / category)
        except Exception as e:
            print(f"[WARN] Could not clean staging for {category}: {e}")

def main():
    FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
    run_group("benign", LIMIT_BENIGN)
    run_group("malware", LIMIT_MALWARE)
    # Remove empty staging root if we cleaned both
    if CLEAN_STAGING:
        try:
            if STAGING_ROOT.exists() and not any(STAGING_ROOT.iterdir()):
                STAGING_ROOT.rmdir()
        except Exception:
            pass

if __name__ == "__main__":
    main()
