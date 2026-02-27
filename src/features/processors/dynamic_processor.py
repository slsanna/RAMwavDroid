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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ----------------------------
# Path + enumeration helpers
# ----------------------------
def _resolve_allowed_types(audio_dir: str, allowed_types: Optional[Set[str]]) -> Optional[Set[str]]:
    """
    If allowed_types is provided, map it to the *actual* folder names under audio_dir
    in a case-insensitive way.
    Example: allowed_types={'wav8'} and audio_dir contains 'WAV8' -> returns {'WAV8'}.
    If nothing matches, returns an empty set.
    """
    if allowed_types is None:
        return None
    if not os.path.isdir(audio_dir):
        return set()

    try:
        entries = [e for e in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, e))]
    except FileNotFoundError:
        return set()

    mapping = {e.casefold(): e for e in entries}
    resolved: Set[str] = set()
    for t in allowed_types:
        real = mapping.get(t.casefold())
        if real:
            resolved.add(real)
    return resolved


def _iter_audio_files(audio_dir: str, allowed_types: Optional[Set[str]] = None) -> Iterable[Tuple[str, str]]:
    """
    Yields (sonifyType, fullPath) for supported audio files under audio_dir/*/.
    If allowed_types is provided, only yields those subfolders (exact names).
    NOTE: It yields files directly under audio/<sonify> and also inside any subfolders.
    """
    if not os.path.isdir(audio_dir):
        return

    exts = {".wav", ".mp3", ".opus", ".flac", ".mid"}
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
        meta = ["apk_id", "memory_region", "category", "sonify_type"]
        others: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in meta and k not in others and not k.startswith("_"):
                    others.append(k)
        header_order = meta + others

    file_exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header_order, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        cleaned = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        w.writerows(cleaned)


# ----------------------------
# Address extraction + fallback
# ----------------------------
_ADDR_REGEX = re.compile(r"0x[0-9A-Fa-f]+")


def _extract_address_str_from_path(path: str) -> Optional[str]:
    """
    Extract the first 0x... hex address from filename OR any parent directories.
    Returns a normalized lowercase '0x...' string, or None.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _ADDR_REGEX.search(stem)
    if m:
        return m.group(0).lower()

    for comp in os.path.normpath(path).split(os.sep):
        m = _ADDR_REGEX.search(comp)
        if m:
            return m.group(0).lower()
    return None


def _fallback_region_label(fullPath: str, sonifyType: str) -> str:
    """
    If we can't find an address, use subfolder name under audio/<sonify>/<area>/file.wav,
    else use '<prefix>_...' before underscore, else the stem.
    """
    parent = os.path.basename(os.path.dirname(fullPath))
    if parent and parent != sonifyType:
        return parent
    stem = os.path.splitext(os.path.basename(fullPath))[0]
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


# =========================================
# Worker functions (thread/process friendly)
# =========================================
_GLOBAL_EXTRACTOR = None  # set in _proc_init_with_factory (process backend)


def _proc_init_with_factory(factory: Callable[[], Any]):
    global _GLOBAL_EXTRACTOR
    _GLOBAL_EXTRACTOR = factory()


def _proc_worker_file(fullPath: str, sonifyType: str, apkFolderPath: str, categoryName: str) -> Optional[Dict[str, Any]]:
    try:
        features = _GLOBAL_EXTRACTOR.extractFeatures(fullPath)  # type: ignore
        if not features:
            return None
        addr_str = _extract_address_str_from_path(fullPath)
        region = addr_str if addr_str is not None else _fallback_region_label(fullPath, sonifyType)
        return {
            "apk_id": os.path.basename(apkFolderPath),
            "memory_region": region,
            "category": categoryName,
            "sonify_type": sonifyType,
            **features,
        }
    except Exception as e:
        logging.warning(f"Error extracting from {fullPath}: {e}")
        return None


def _thread_worker_file(
    fullPath: str,
    sonifyType: str,
    apkFolderPath: str,
    categoryName: str,
    extractor_callable: Callable[[str], Optional[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    try:
        features = extractor_callable(fullPath)
        if not features:
            return None
        addr_str = _extract_address_str_from_path(fullPath)
        region = addr_str if addr_str is not None else _fallback_region_label(fullPath, sonifyType)
        return {
            "apk_id": os.path.basename(apkFolderPath),
            "memory_region": region,
            "category": categoryName,
            "sonify_type": sonifyType,
            **features,
        }
    except Exception as e:
        logging.warning(f"Error extracting from {fullPath}: {e}")
        return None


def _load_processed_apk_ids(features_dir: Path, category: str, allowed_types: Optional[Set[str]]) -> Set[str]:
    done: Set[str] = set()

    def add_from_csv(csv_path: Path):
        try:
            if csv_path.is_file():
                s = pd.read_csv(csv_path, usecols=["apk_id"])["apk_id"]
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


# ----------------------------
# Processor
# ----------------------------
class DynamicProcessor:
    """
    Parallel feature extraction with APK-level progress, per-APK CSV writes,
    raw-address memory_region, and resume support.
    """

    def __init__(
        self,
        featureExtractor=None,
        memoryRegions: str = "all",
        maxWorkers: int = 4,
        limit: Optional[int] = None,
        backend: str = "thread",
        extractorFactory: Optional[Callable[[], Any]] = None,
        batch_size: int = 512,
    ):
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

    @staticmethod
    def _apply_slice(apk_folders: List[str], start: Optional[int], end: Optional[int]) -> List[str]:
        """
        Apply [start:end) slicing safely after sorting.
        """
        s = 0 if start is None else max(0, start)
        e = len(apk_folders) if end is None else min(len(apk_folders), end)
        if s >= e:
            return []
        return apk_folders[s:e]

    # ---------- Core ----------
    def processCategoryFolder(
        self,
        categoryFolder: str,
        categoryName: str,
        allowed_types: Optional[Set[str]],
        features_dir: Path,
        processed_apks: Set[str],
        *,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        if not os.path.isdir(categoryFolder):
            logging.warning(f"Category folder doesn't exist: {categoryFolder}")
            return

        apkFolders = [
            os.path.join(categoryFolder, item)
            for item in os.listdir(categoryFolder)
            if os.path.isdir(os.path.join(categoryFolder, item))
        ]
        apkFolders.sort()

        if self.limit:
            apkFolders = apkFolders[: self.limit]

        # ✅ apply slice BEFORE resume filtering (your CLI semantics)
        apkFolders = self._apply_slice(apkFolders, start, end)

        to_run = []
        for apkFolder in apkFolders:
            apk_id = os.path.basename(apkFolder)
            if apk_id in processed_apks:
                # keep it concise; otherwise it spams
                logging.info(f"[resume] skip already processed APK: {apk_id}")
                continue
            to_run.append(apkFolder)

        selected_apk = len(to_run)
        logging.info(f"Processing {selected_apk} APK folders from {categoryName}")

        if self.backend == "thread":
            executor_cls = ThreadPoolExecutor
            if self.featureExtractor is None:
                raise ValueError("featureExtractor must be provided for thread backend.")
            extractor_callable = lambda p: self.featureExtractor.extractFeatures(p)
            worker_fn = _thread_worker_file
            worker_args_extra = (extractor_callable,)
        else:
            executor_cls = partial(
                ProcessPoolExecutor, initializer=_proc_init_with_factory, initargs=(self.extractorFactory,)
            )
            worker_fn = _proc_worker_file
            worker_args_extra = tuple()

        with executor_cls(max_workers=self.maxWorkers) as ex, tqdm(
            total=selected_apk, desc=f"Extracting {categoryName}", unit="apk"
        ) as apk_pbar:

            for apkFolder in to_run:
                apk_id = os.path.basename(apkFolder)
                audio_dir = os.path.join(apkFolder, "audio")

                # ✅ case-insensitive mapping of allowed_types against real folder names
                resolved_allowed = _resolve_allowed_types(audio_dir, allowed_types)

                # If user requested a type but it doesn't exist (case-insensitive), we can skip fast.
                if allowed_types is not None and (resolved_allowed is None or len(resolved_allowed) == 0):
                    logging.info(
                        f"[{categoryName}] {apk_id} → wrote 0 rows (no matching sonify dir for {sorted(allowed_types)})"
                    )
                    apk_pbar.update(1)
                    continue

                futures = []
                for (sonifyType, fullPath) in _iter_audio_files(audio_dir, resolved_allowed):
                    if worker_fn is _thread_worker_file:
                        futures.append(
                            ex.submit(
                                worker_fn,
                                fullPath,
                                sonifyType,
                                apkFolder,
                                categoryName,
                                *worker_args_extra,
                            )
                        )
                    else:
                        futures.append(ex.submit(worker_fn, fullPath, sonifyType, apkFolder, categoryName))

                if not futures:
                    logging.info(f"[{categoryName}] {apk_id} → wrote 0 rows (no audio files found)")
                    apk_pbar.update(1)
                    continue

                per_apk_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

                for fut in as_completed(futures):
                    row = fut.result()
                    if not row:
                        continue
                    st = row.get("sonify_type")
                    if resolved_allowed and st not in resolved_allowed:
                        continue
                    per_apk_buckets[st].append(row)

                total_rows = 0
                for st, rows in per_apk_buckets.items():
                    if not rows:
                        continue
                    csvPath = features_dir / f"{categoryName}_{st}_features.csv"
                    _write_rows_csv(csvPath, rows)
                    total_rows += len(rows)

                logging.info(f"[{categoryName}] {apk_id} → wrote {total_rows} rows")
                apk_pbar.update(1)

    def processDataset(
        self,
        baseDir: str,
        categories: List[str],
        sonifyTypes: List[str],
        featuresDir: str,
        *,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        features_dir = Path(featuresDir)
        self._ensure_parent(features_dir / "dummy.txt")

        allowed_types = set(sonifyTypes) if sonifyTypes else None

        for category in categories:
            categoryPath = os.path.join(baseDir, category)
            logging.info(f"Processing category: {category}")

            processed_apks = _load_processed_apk_ids(features_dir, category, allowed_types)
            if processed_apks:
                logging.info(f"[resume] Found {len(processed_apks)} APK(s) already in CSV for {category}")

            self.processCategoryFolder(
                categoryFolder=categoryPath,
                categoryName=category,
                allowed_types=allowed_types,
                features_dir=features_dir,
                processed_apks=processed_apks,
                start=start,
                end=end,
            )

    def createCombinedDatasets(self, featuresDir: str, categories: List[str], sonifyTypes: List[str]) -> None:
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

    # ----------------------------
    # ✅ Compatibility wrapper
    # ----------------------------
    def run(
        self,
        data_dir: str,
        categories: List[str],
        sonify: str,
        *,
        verbose: bool = False,
        limit: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        output_dir: Optional[str] = None,
        create_combined: bool = False,
    ) -> None:
        if output_dir is None:
            raise ValueError("output_dir must be provided to DynamicProcessor.run()")

        if limit is not None:
            self.limit = limit

        if verbose:
            logging.getLogger().setLevel(logging.INFO)

        self.processDataset(
            baseDir=data_dir,
            categories=categories,
            sonifyTypes=[sonify],
            featuresDir=output_dir,
            start=start,
            end=end,
        )

        if create_combined:
            self.createCombinedDatasets(
                featuresDir=output_dir,
                categories=categories,
                sonifyTypes=[sonify],
            )
