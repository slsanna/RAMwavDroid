import os
import re
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Callable, Iterable, Tuple, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# =========================
# File enumeration helpers
# =========================
def _iter_audio_files(audio_dir: str, allowed_types: Optional[Set[str]] = None) -> Iterable[Tuple[str, str]]:
    """
    Yields (sonifyType, fullPath) for supported audio files under audio_dir/*/.
    If allowed_types is provided, only yields those subfolders (e.g., {'wav8'}).
    Also yields files inside audio/<sonify>/<subdir>/...
    """
    if not os.path.isdir(audio_dir):
        return
    exts = {'.wav', '.mp3', '.opus', '.flac', '.mid'}
    try:
        for sonifyType in os.listdir(audio_dir):
            if allowed_types is not None and sonifyType not in allowed_types:
                continue
            sonifyPath = os.path.join(audio_dir, sonifyType)
            if not os.path.isdir(sonifyPath):
                continue

            # files directly under audio/<sonify>/
            with os.scandir(sonifyPath) as it:
                for entry in it:
                    if entry.is_file() and os.path.splitext(entry.name)[1].lower() in exts:
                        yield sonifyType, entry.path

            # files inside audio/<sonify>/<subdir>/
            with os.scandir(sonifyPath) as it:
                for sub in it:
                    if sub.is_dir():
                        with os.scandir(sub.path) as it2:
                            for entry in it2:
                                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in exts:
                                    yield sonifyType, entry.path
    except FileNotFoundError:
        return


def _write_rows_csv(path: Path, rows: List[Dict[str, Any]], header_order: Optional[List[str]] = None) -> None:
    """
    Append rows to CSV at `path`. Creates file + header if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    if header_order is None:
        # stable header: common meta first, then discovered feature keys in first-seen order
        meta = ['apk_id', 'memory_region', 'category', 'sonify_type']
        others: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in meta and k not in others:
                    others.append(k)
        header_order = meta + others

    file_exists = path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header_order, extrasaction='ignore')
        if not file_exists:
            w.writeheader()
        w.writerows(rows)


# =========================
# maps.txt parsing + lookup
# =========================
def _find_maps_file(apk_folder: str) -> Optional[str]:
    """
    Locate maps.txt for an APK. Checks common places, then falls back to a shallow walk.
    """
    candidates = [
        os.path.join(apk_folder, "maps.txt"),
        os.path.join(apk_folder, "audio", "maps.txt"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # fallback: first match under apk_folder
    for root, _, files in os.walk(apk_folder):
        if "maps.txt" in files:
            return os.path.join(root, "maps.txt")
    return None


def _parse_maps_file(maps_path: str) -> Tuple[List[int], List[Tuple[int, int, str]]]:
    """
    Parse a Linux /proc/*/maps-like file into sorted ranges.
    Returns (starts, ranges) where ranges[i] = (start, end, name) and starts[i] == start.
    """
    ranges: List[Tuple[int, int, str]] = []
    hexrng = re.compile(r'^([0-9A-Fa-f]+)-([0-9A-Fa-f]+)$')

    try:
        with open(maps_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                m = hexrng.match(parts[0])
                if not m:
                    continue
                start = int(m.group(1), 16)
                end = int(m.group(2), 16)

                name = ""
                if len(parts) >= 6:
                    name = parts[-1]
                # Normalize name
                if not name:
                    name = "anon"
                elif name.startswith('/'):
                    name = os.path.basename(name)
                elif name.startswith('[') and name.endswith(']'):
                    name = name[1:-1]  # e.g., [heap] -> heap

                ranges.append((start, end, name))
    except Exception as e:
        logging.warning(f"Failed to parse maps file {maps_path}: {e}")
        return [], []

    # sort by start and build starts array
    ranges.sort(key=lambda x: x[0])
    starts = [r[0] for r in ranges]
    return starts, ranges


def _extract_address_from_path(path: str) -> Optional[int]:
    """
    Extract first 0x... hex address from filename or any parent part. Returns int or None.
    """
    # filename first
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r'0x[0-9A-Fa-f]+', stem)
    if m:
        return int(m.group(0), 16)
    # parents as fallback
    parts = os.path.normpath(path).split(os.sep)[::-1]
    for p in parts:
        m = re.search(r'0x[0-9A-Fa-f]+', p)
        if m:
            return int(m.group(0), 16)
    return None


def _lookup_area_name(addr: int, starts: List[int], ranges: List[Tuple[int, int, str]]) -> Optional[str]:
    """
    Fast lookup: find name for address 'addr' using starts/ranges (binary search).
    """
    if addr is None or not starts:
        return None
    import bisect
    i = bisect.bisect_right(starts, addr) - 1
    if i >= 0:
        start, end, name = ranges[i]
        if start <= addr < end:
            return name
    return None


def _fallback_area_name(fullPath: str, sonifyType: str) -> str:
    """
    Fallback if no address match: use subfolder under audio/<sonify>/<area>/file.wav if present,
    else heuristic from filename.
    """
    parent = os.path.basename(os.path.dirname(fullPath))
    if parent and parent != sonifyType:
        return parent
    stem = os.path.splitext(os.path.basename(fullPath))[0]
    if '_' in stem:
        return stem.split('_', 1)[0]
    return stem


# =========================================
# Worker callable (thread/process friendly)
# =========================================
_GLOBAL_EXTRACTOR = None  # set in _proc_init_with_factory for process backend


def _proc_init_with_factory(factory: Callable[[], Any]):
    """
    Initializer executed once per worker process. 'factory' MUST be picklable (top-level callable).
    """
    global _GLOBAL_EXTRACTOR
    _GLOBAL_EXTRACTOR = factory()


def _worker_process_file(fullPath: str,
                         sonifyType: str,
                         apkFolderPath: str,
                         categoryName: str,
                         extractor_callable: Callable[[str], Optional[Dict[str, Any]]]
                         ) -> Optional[Dict[str, Any]]:
    """
    Extract features for a single file and return a minimal row; memory_area is resolved later via maps.
    """
    try:
        features = extractor_callable(fullPath)
        if not features:
            return None
        addr = _extract_address_from_path(fullPath)  # int or None
        # store a fallback name now; we'll overwrite with maps lookup if available
        fallback = _fallback_area_name(fullPath, sonifyType)
        row = {
            'apk_id': os.path.basename(apkFolderPath),
            'memory_region': fallback,   # placeholder; may be overwritten with maps match
            'category': categoryName,
            'sonify_type': sonifyType,
            '_addr': addr,               # internal use in parent
            **features
        }
        return row
    except Exception as e:
        logging.warning(f"Error extracting from {fullPath}: {e}")
        return None


# ================
# DynamicProcessor
# ================
class DynamicProcessor:
    """
    Parallel feature extraction with APK-level progress, per-APK CSV writes, maps.txt-based
    memory area resolution, and resume support.
    """

    def __init__(self,
                 featureExtractor=None,
                 memoryRegions: str = "all",
                 maxWorkers: int = 4,
                 limit: Optional[int] = None,
                 backend: str = "thread",
                 extractorFactory: Optional[Callable[[], Any]] = None,
                 batch_size: int = 512):
        """
        Args:
            featureExtractor: Extractor instance with .extractFeatures(path) -> dict (thread backend).
            memoryRegions: kept for API compatibility.
            maxWorkers: workers for thread/process pool.
            limit: optional limit of APK folders per category.
            backend: "thread" or "process".
            extractorFactory: top-level callable returning a new extractor (process backend).
            batch_size: not used in per-APK write path; retained for API compatibility.
        """
        self.featureExtractor = featureExtractor
        self.memoryRegions = memoryRegions
        self.maxWorkers = maxWorkers
        self.limit = limit
        self.backend = backend
        self.extractorFactory = extractorFactory
        self.batch_size = max(64, int(batch_size))

        if backend not in {"thread", "process"}:
            raise ValueError('backend must be "thread" or "process"')
        if backend == "process" and extractorFactory is None:
            logging.warning("Process backend selected but no extractorFactory provided. Falling back to thread.")
            self.backend = "thread"

    # ---------- IO helpers ----------
    @staticmethod
    def _ensure_parent(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(path)
        except Exception as e:
            logging.error(f"Reading {path}: {e}")
            return None

    # ---------- Resume ----------
    @staticmethod
    def _load_processed_apk_ids(features_dir: Path, category: str, allowed_types: Optional[Set[str]]) -> Set[str]:
        """
        Build a set of APK ids already present in the category CSV(s) so we can resume.
        """
        done: Set[str] = set()

        def add_from_csv(csv_path: Path):
            try:
                if csv_path.is_file():
                    s = pd.read_csv(csv_path, usecols=['apk_id'])['apk_id']
                    done.update(s.dropna().astype(str).tolist())
            except Exception as e:
                logging.warning(f"Resume: failed reading {csv_path}: {e}")

        if allowed_types:
            for st in allowed_types:
                add_from_csv(features_dir / f"{category}_{st}_features.csv")
        else:
            for p in features_dir.glob(f"{category}_*_features.csv"):
                add_from_csv(p)
        return done

    # ---------- Core ----------
    def processCategoryFolder(self,
                              categoryFolder: str,
                              categoryName: str,
                              allowed_types: Optional[Set[str]],
                              features_dir: Path,
                              processed_apks: Set[str]) -> None:
        """
        Processes all APK folders in a category, extracting features in parallel.
        Writes CSVs once per APK (so we can resume and keep memory small).
        Resolves memory area names via maps.txt.
        """
        if not os.path.isdir(categoryFolder):
            logging.warning(f"Category folder doesn't exist: {categoryFolder}")
            return

        apkFolders = [os.path.join(categoryFolder, item) for item in os.listdir(categoryFolder)
                      if os.path.isdir(os.path.join(categoryFolder, item))]
        apkFolders.sort()
        if self.limit:
            apkFolders = apkFolders[:self.limit]

        # Resume: skip already processed APKs
        to_run = []
        for apkFolder in apkFolders:
            apk_id = os.path.basename(apkFolder)
            if apk_id in processed_apks:
                logging.info(f"[resume] skip already processed APK: {apk_id}")
                continue
            to_run.append(apkFolder)

        selected_apk = len(to_run)
        logging.info(f"Processing {selected_apk} APK folders from {categoryName}")

        # Executor + extractor callable
        if self.backend == "thread":
            executor_cls = ThreadPoolExecutor
            if self.featureExtractor is None:
                raise ValueError("featureExtractor must be provided for thread backend.")
            extractor_callable = (lambda p: self.featureExtractor.extractFeatures(p))
        else:
            # process backend: module-level initializer (picklable!)
            executor_cls = partial(ProcessPoolExecutor,
                                   initializer=_proc_init_with_factory,
                                   initargs=(self.extractorFactory,))
            extractor_callable = (lambda p: _GLOBAL_EXTRACTOR.extractFeatures(p))  # type: ignore

        with executor_cls(max_workers=self.maxWorkers) as ex, \
             tqdm(total=selected_apk, desc=f"Extracting {categoryName}", unit="apk") as apk_pbar:

            for apkFolder in to_run:
                apk_id = os.path.basename(apkFolder)
                audio_dir = os.path.join(apkFolder, "audio")

                # Prepare maps lookup for this APK
                maps_file = _find_maps_file(apkFolder)
                starts, ranges = _parse_maps_file(maps_file) if maps_file else ([], [])
                if maps_file:
                    logging.debug(f"[maps] Using {maps_file} for {apk_id} ({len(ranges)} ranges)")

                futures = [
                    ex.submit(_worker_process_file, fullPath, sonifyType, apkFolder, categoryName, extractor_callable)
                    for (sonifyType, fullPath) in _iter_audio_files(audio_dir, allowed_types)
                ]

                # Collect results for THIS APK only and resolve memory area via maps
                per_apk_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                resolved_count = 0

                for fut in as_completed(futures):
                    row = fut.result()
                    if not row:
                        continue
                    st = row.get('sonify_type')
                    if allowed_types and st not in allowed_types:
                        continue

                    addr = row.pop('_addr', None)
                    if addr is not None:
                        name = _lookup_area_name(addr, starts, ranges)
                        if name:
                            row['memory_region'] = name
                            resolved_count += 1
                    # else keep fallback already in row['memory_region']
                    per_apk_buckets[st].append(row)

                # Write THIS APK's rows to the category CSV(s)
                total_rows = 0
                for st, rows in per_apk_buckets.items():
                    if not rows:
                        continue
                    csvPath = features_dir / f"{categoryName}_{st}_features.csv"
                    _write_rows_csv(csvPath, rows)
                    total_rows += len(rows)

                logging.info(f"[{categoryName}] {apk_id} → wrote {total_rows} rows (maps matched: {resolved_count})")
                apk_pbar.update(1)

    def processDataset(self, baseDir: str, categories: List[str], sonifyTypes: List[str], featuresDir: str) -> None:
        """
        Extract features per category and write per-sonify CSVs at the end of each APK.
        Resume is supported by reading existing CSVs and skipping APKs already present.
        """
        features_dir = Path(featuresDir)
        self._ensure_parent(features_dir / "dummy.txt")  # ensure dir exists

        allowed_types = set(sonifyTypes) if sonifyTypes else None  # e.g., {"wav8"}

        for category in categories:
            categoryPath = os.path.join(baseDir, category)
            logging.info(f"Processing category: {category}")

            # Build 'processed_apks' set from existing CSVs for resume
            processed_apks = self._load_processed_apk_ids(features_dir, category, allowed_types)
            if processed_apks:
                logging.info(f"[resume] Found {len(processed_apks)} APK(s) already in CSV for {category}")

            # Parallel extraction with per-APK writes
            self.processCategoryFolder(
                categoryFolder=categoryPath,
                categoryName=category,
                allowed_types=allowed_types,
                features_dir=features_dir,
                processed_apks=processed_apks
            )

    def createCombinedDatasets(self, featuresDir: str, categories: List[str], sonifyTypes: List[str]) -> None:
        """
        Concatenate per-category CSVs into all_{sonifyType}_features.csv files.
        """
        logging.info("Creating combined datasets")
        features_dir = Path(featuresDir)
        self._ensure_parent(features_dir / "dummy.txt")

        for sonifyType in sonifyTypes:
            dfs: List[pd.DataFrame] = []
            for category in categories:
                csvPath = features_dir / f"{category}_{sonifyType}_features.csv"
                if csvPath.is_file():
                    df = self._safe_read_csv(csvPath)
                    if df is not None and not df.empty:
                        dfs.append(df)
                        logging.info(f"Added {category} ({len(df)} samples)")

            if not dfs:
                logging.info(f"No files found for {sonifyType}")
                continue

            try:
                combined = pd.concat(dfs, ignore_index=True)
            except Exception as e:
                logging.error(f"Concatenation failed for {sonifyType}: {e}")
                continue

            outputPath = features_dir / f"all_{sonifyType}_features.csv"
            outputPath.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(outputPath, index=False)
            logging.info(f"Created {outputPath} with {len(combined)} samples")
