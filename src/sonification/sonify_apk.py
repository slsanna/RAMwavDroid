#!/usr/bin/env python3
"""
Robust APK sonification runner (ancestor-folder aware, no SHA hashing by default).

- Dump trees: /mnt/malware_ram/Android/{benign_dumps,dumps_dataset}
  Dump folder names look like: <hash>.apk_com.pkg_.../
- APK roots:
    Benign : /mnt/AndroidProject/DatasetExplainability/labelled/Benign
    Malware: /mnt/FileSources/VirusTotal/RAM_android2/samples/dataset_apk

Matching order per token (lowercased):
  1) Filename-prefix index  (token from APK filename)
  2) Ancestor-folder index  (token from ANY ancestor under the root, e.g., <HASH>/.../base.apk)
  3) Substring fallback     (search token/variants inside filename strings)
  4) (Optional) SHA-256 fallback (disabled by default)

Outputs go to:
  /home/ssanna/Desktop/malware_ram/Android/audio/malware-sonification-signal-processing/apk/{benign,malware}

Python 3.7/3.8 compatible.
"""

import os
import sys
import re
import hashlib
from pathlib import Path
from typing import Dict, Set, Optional, Tuple, List
import logging
from tqdm import tqdm

# --- Make project 'src' importable (adjust if needed) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.sonification.sonification import Sonifier
# --------------------------------------------------------

# ---------------- CONFIG ----------------
BASE_DUMP_DIR = Path("/mnt/malware_ram/Android")
DUMP_BENIGN = BASE_DUMP_DIR / "benign_dumps"
DUMP_MALWARE = BASE_DUMP_DIR / "dumps_dataset"

APK_DIR_BENIGN = Path("/mnt/AndroidProject/DatasetExplainability/labelled/Benign")
APK_DIR_MALWARE = Path("/mnt/FileSources/VirusTotal/RAM_android2/samples/dataset_apk")

OUTPUT_ROOT    = Path("/home/ssanna/Desktop/malware_ram/Android/audio/malware-sonification-signal-processing/apk")
OUTPUT_BENIGN  = OUTPUT_ROOT / "benign"
OUTPUT_MALWARE = OUTPUT_ROOT / "malware"

SONIFICATION_TYPE = "wav8"    # "wav8" | "wav16" | "mp3"
SAMPLE_RATE = 32768
WORKERS = 8

# Limits: set to None to disable
LIMIT_TOTAL: Optional[int] = None
LIMIT_PER_GROUP: Optional[int] = 300

# Fallback modes
ENABLE_SHA256_FALLBACK = False       # ← disable slow hashing by default
ENABLE_SUBSTRING_FALLBACK = True

VERBOSE = True
# ---------------------------------------

LEADING_HEX_RE = re.compile(r"^(?P<hash>[0-9a-fA-F]{32,128})", re.ASCII)

def ensure_dirs() -> None:
    OUTPUT_BENIGN.mkdir(parents=True, exist_ok=True)
    OUTPUT_MALWARE.mkdir(parents=True, exist_ok=True)

def extract_hash_token(name: str) -> Optional[str]:
    """
    Extract token from a dump folder/filename:
      - If contains '.apk', take left side of first '.apk'
      - Else take leading 32..128 hex run
    Lowercased; None if not found.
    """
    low = name.lower()
    if ".apk" in low:
        token = low.split(".apk", 1)[0].strip()
        if token:
            return token
    m = LEADING_HEX_RE.match(low)
    return m.group("hash").lower() if m else None

def collect_hashes_from_dump_tree(root: Path) -> Tuple[Set[str], List[str]]:
    """
    Prefer top-level entries (common datasetRAM layout),
    fallback to recursive scan. Returns (hashes, sample_names).
    """
    hashes: Set[str] = set()
    samples: List[str] = []

    if not root.exists():
        return hashes, samples

    # Pass 1: top-level dirs
    for entry in root.iterdir():
        token = extract_hash_token(entry.name)
        if token:
            hashes.add(token)
            if len(samples) < 5:
                samples.append(entry.name)

    # Pass 2: recursive if needed
    if not hashes:
        for dirpath, dirnames, filenames in os.walk(str(root)):
            for d in dirnames:
                token = extract_hash_token(d)
                if token:
                    hashes.add(token)
                    if len(samples) < 5:
                        samples.append(d)
            for f in filenames:
                token = extract_hash_token(f)
                if token:
                    hashes.add(token)
                    if len(samples) < 5:
                        samples.append(f)

    return hashes, samples

def token_from_filename(fname: str) -> Optional[str]:
    """
    From an APK filename:
      - if has '.apk', token = left of '.apk'
      - else if starts with long hex, token = that hex
      - else None
    """
    low = fname.lower()
    if ".apk" in low:
        return low.split(".apk", 1)[0].strip()
    m = LEADING_HEX_RE.match(low)
    return m.group("hash").lower() if m else None

def index_apks_with_ancestors(root: Path) -> Tuple[Dict[str, Path], Dict[str, Path], List[str], List[Tuple[str, Path]]]:
    """
    Build TWO indices:
      file_idx[token]    -> path   (token from file name)
      ancestor_idx[token]-> path   (token from ANY ancestor folder name under 'root',
                                    esp. top-level <HASH> dir)
    Also returns:
      examples (some sample paths) and
      all_names ([(lower_filename, path), ...]) for substring fallback.

    Preference: if multiple candidates exist for same token, prefer a file whose
    name contains '.apk'.
    """
    file_idx: Dict[str, Path] = {}
    ancestor_idx: Dict[str, Path] = {}
    examples: List[str] = []
    all_names: List[Tuple[str, Path]] = []

    if not root.exists():
        return file_idx, ancestor_idx, examples, all_names

    root = root.resolve()

    for dirpath, _, files in os.walk(str(root)):
        dirpath_p = Path(dirpath)
        try:
            rel_parts = dirpath_p.relative_to(root).parts  # ancestor names under root
        except Exception:
            rel_parts = ()

        # collect tokens from ALL ancestors under root (e.g., <HASH>, 'original', etc.)
        ancestor_tokens: List[str] = []
        for part in rel_parts:
            tok = extract_hash_token(part)
            if tok:
                ancestor_tokens.append(tok)

        for fname in files:
            low = fname.lower()
            p = dirpath_p / fname
            all_names.append((low, p))

            # ---------- file-name token ----------
            ftok = token_from_filename(fname)
            if ftok:
                if ftok not in file_idx:
                    file_idx[ftok] = p
                    if len(examples) < 5:
                        examples.append(str(p))
                else:
                    old = file_idx[ftok]
                    if (".apk" in low) and (".apk" not in old.name.lower()):
                        file_idx[ftok] = p  # prefer .apk-named file

            # ---------- ancestor-folder tokens ----------
            for ptok in ancestor_tokens:
                if ptok not in ancestor_idx:
                    ancestor_idx[ptok] = p
                else:
                    old = ancestor_idx[ptok]
                    if (".apk" in low) and (".apk" not in old.name.lower()):
                        ancestor_idx[ptok] = p  # prefer .apk-named file

    return file_idx, ancestor_idx, examples, all_names

def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk_size)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception as e:
        logging.warning("SHA256 failed for %s: %s", path, e)
        return None

def index_apks_by_sha256(root: Path) -> Tuple[Dict[str, Path], int]:
    """
    Compute SHA-256 for every file under root and build sha->path map.
    (Only used if ENABLE_SHA256_FALLBACK=True)
    """
    index: Dict[str, Path] = {}
    count = 0
    if not root.exists():
        return index, count

    apk_paths: List[Path] = []
    for dirpath, _, files in os.walk(str(root)):
        for fname in files:
            apk_paths.append(Path(dirpath) / fname)

    for p in tqdm(apk_paths, desc=f"Hashing files in {root}", unit="file"):
        digest = sha256_of_file(p)
        if digest:
            index.setdefault(digest.lower(), p)
            count += 1
    return index, count

def token_variants(token: str) -> List[str]:
    """
    Normalized candidates:
      - full token
      - first 64, 40, 32 chars
    Deduped, lowercased.
    """
    low = token.lower()
    cands = [low]
    for n in (64, 40, 32):
        if len(low) > n:
            cands.append(low[:n])
        elif len(low) == n:
            cands.append(low)
    # dedupe
    seen = set()
    out = []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out

def main() -> None:
    logging.basicConfig(
        level=logging.INFO if VERBOSE else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    ensure_dirs()

    # 1) Collect dump tokens
    benign_hashes, benign_samples = collect_hashes_from_dump_tree(DUMP_BENIGN)
    malware_hashes, malware_samples = collect_hashes_from_dump_tree(DUMP_MALWARE)

    logging.info("Dump dirs:")
    logging.info("  benign_dumps   : %s -> hashes: %d", DUMP_BENIGN, len(benign_hashes))
    if benign_samples: logging.info("    e.g.: %s", benign_samples)
    logging.info("  dumps_dataset  : %s -> hashes: %d", DUMP_MALWARE, len(malware_hashes))
    if malware_samples: logging.info("    e.g.: %s", malware_samples)

    if not benign_hashes and not malware_hashes:
        logging.error("No hashes parsed from dumps. Check the paths/naming.")
        return

    # 2) Build indices for APK roots (file-name & ancestor-folder)
    b_file_idx, b_anc_idx, b_examples, b_all_names = index_apks_with_ancestors(APK_DIR_BENIGN)
    m_file_idx, m_anc_idx, m_examples, m_all_names = index_apks_with_ancestors(APK_DIR_MALWARE)

    logging.info("APK roots (indices):")
    logging.info("  benign file-index  : %d, ancestor-index: %d", len(b_file_idx), len(b_anc_idx))
    if b_examples: logging.info("    e.g.: %s", b_examples)
    logging.info("  malware file-index : %d, ancestor-index: %d", len(m_file_idx), len(m_anc_idx))
    if m_examples: logging.info("    e.g.: %s", m_examples)

    # 3) Prepare Sonifier
    son = Sonifier(
        sonificationType=SONIFICATION_TYPE,
        sampleRate=SAMPLE_RATE,
        workers=WORKERS,
        datasetType="legacy",
    )

    processed = 0
    processed_benign = 0
    processed_malware = 0
    errors = 0
    matched = {"benign": 0, "malware": 0}
    missing = {"benign": 0, "malware": 0}

    # Lazy SHA indices (only if enabled)
    b_sha_idx: Dict[str, Path] = {}
    m_sha_idx: Dict[str, Path] = {}
    b_sha_built = False
    m_sha_built = False

    def hit_limit(group: str) -> bool:
        if LIMIT_TOTAL is not None and processed >= LIMIT_TOTAL:
            return True
        if LIMIT_PER_GROUP is not None:
            if group == "benign" and processed_benign >= LIMIT_PER_GROUP:
                return True
            if group == "malware" and processed_malware >= LIMIT_PER_GROUP:
                return True
        return False

    def process_one(apk_path: Path, out_dir: Path, group: str) -> bool:
        nonlocal processed, processed_benign, processed_malware, errors
        base_no_ext = apk_path.name
        base_no_ext = base_no_ext[: base_no_ext.lower().rfind(".apk")] if ".apk" in base_no_ext.lower() else base_no_ext
        out_path = out_dir / (base_no_ext + son.fileExtension)

        # Skip if exists and non-empty
        try:
            if out_path.exists() and out_path.stat().st_size > 0:
                tqdm.write(f"[SKIP] {group}: {apk_path.name} (exists)")
                processed += 1
                if group == "benign":
                    processed_benign += 1
                else:
                    processed_malware += 1
                return True
        except Exception:
            pass

        try:
            binaryData, extracted = son.extractor.extract(str(apk_path))
            if not extracted or not binaryData:
                tqdm.write(f"[WARN] {group}: extractor returned no data for {apk_path}")
                return False
            ok = son.sonifier.sonify(binaryData, str(out_path))
            if ok:
                processed += 1
                if group == "benign":
                    processed_benign += 1
                else:
                    processed_malware += 1
                tqdm.write(f"[OK] {group}: {apk_path.name} -> {out_path}")
                return True
            else:
                errors += 1
                tqdm.write(f"[FAIL] {group}: sonify failed for {apk_path}")
                return False
        except Exception as e:
            errors += 1
            tqdm.write(f"[ERR] {group}: {apk_path} -> {e}")
            return False

    def build_sha_index_if_needed(group: str) -> None:
        nonlocal b_sha_built, m_sha_built, b_sha_idx, m_sha_idx
        if not ENABLE_SHA256_FALLBACK:
            return
        if group == "benign" and not b_sha_built:
            tqdm.write("[INFO] Building SHA-256 index for benign APKs (fallback)…")
            b_sha_idx, _ = index_apks_by_sha256(APK_DIR_BENIGN)
            b_sha_built = True
        elif group == "malware" and not m_sha_built:
            tqdm.write("[INFO] Building SHA-256 index for malware APKs (fallback)…")
            m_sha_idx, _ = index_apks_by_sha256(APK_DIR_MALWARE)
            m_sha_built = True

    def resolve_by_prefix_and_ancestors(token: str, file_idx: Dict[str, Path], anc_idx: Dict[str, Path]) -> Optional[Path]:
        # try variants on file index
        for cand in token_variants(token):
            p = file_idx.get(cand)
            if p: return p
        # then ancestor-folder index
        for cand in token_variants(token):
            p = anc_idx.get(cand)
            if p: return p
        return None

    def resolve_by_sha(token: str, group: str) -> Optional[Path]:
        if not ENABLE_SHA256_FALLBACK:
            return None
        cands = token_variants(token)
        if not cands:
            return None
        build_sha_index_if_needed(group)
        idx = b_sha_idx if group == "benign" else m_sha_idx
        for c in cands:
            p = idx.get(c)
            if p:
                return p
        return None

    def resolve_by_substring(token: str, all_names: List[Tuple[str, Path]]) -> Optional[Path]:
        for c in token_variants(token):
            for lowname, p in all_names:
                if c in lowname:
                    return p
        return None

    def resolve_apk(token: str, group: str) -> Optional[Path]:
        if group == "benign":
            p = resolve_by_prefix_and_ancestors(token, b_file_idx, b_anc_idx)
            if p: return p
            p = resolve_by_sha(token, "benign")
            if p: return p
            if ENABLE_SUBSTRING_FALLBACK and b_all_names:
                return resolve_by_substring(token, b_all_names)
            return None
        else:
            p = resolve_by_prefix_and_ancestors(token, m_file_idx, m_anc_idx)
            if p: return p
            p = resolve_by_sha(token, "malware")
            if p: return p
            if ENABLE_SUBSTRING_FALLBACK and m_all_names:
                return resolve_by_substring(token, m_all_names)
            return None

    # 4) Process benign with progress bar
    benign_list = sorted(benign_hashes)
    pbar_b = tqdm(benign_list, desc="Processing benign", unit="apk")
    for token in pbar_b:
        if (LIMIT_PER_GROUP is not None and processed_benign >= LIMIT_PER_GROUP) \
           or (LIMIT_TOTAL is not None and processed >= LIMIT_TOTAL):
            break
        apk = resolve_apk(token, "benign")
        pbar_b.set_postfix_str((token[:16] + "…") if len(token) > 16 else token)
        if apk is None:
            missing["benign"] += 1
            continue
        matched["benign"] += 1
        pbar_b.set_description(f"Processing benign ({apk.name[:32]}…)")
        process_one(apk, OUTPUT_BENIGN, "benign")

    # 5) Process malware with progress bar
    malware_list = sorted(malware_hashes)
    pbar_m = tqdm(malware_list, desc="Processing malware", unit="apk")
    for token in pbar_m:
        if (LIMIT_PER_GROUP is not None and processed_malware >= LIMIT_PER_GROUP) \
           or (LIMIT_TOTAL is not None and processed >= LIMIT_TOTAL):
            break
        apk = resolve_apk(token, "malware")
        pbar_m.set_postfix_str((token[:16] + "…") if len(token) > 16 else token)
        if apk is None:
            missing["malware"] += 1
            continue
        matched["malware"] += 1
        pbar_m.set_description(f"Processing malware ({apk.name[:32]}…)")
        process_one(apk, OUTPUT_MALWARE, "malware")

    # 6) Summary
    logging.info("-------- Summary --------")
    logging.info("Benign  - hashes: %d  matched: %d  missing: %d", len(benign_list), matched["benign"],  missing["benign"])
    logging.info("Malware - hashes: %d  matched: %d  missing: %d", len(malware_list), matched["malware"], missing["malware"])
    logging.info("Processed benign     : %d", processed_benign)
    logging.info("Processed malware    : %d", processed_malware)
    logging.info("Processed outputs    : %d", processed)
    logging.info("Errors               : %d", errors)
    logging.info("Outputs saved under  : %s", OUTPUT_ROOT)

if __name__ == "__main__":
    main()
