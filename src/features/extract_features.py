#!/usr/bin/env python3
import argparse
import logging
import os

from extractors.manual_feature_extractor import ManualFeatureExtractor
from processors.static_processor import StaticProcessor
from processors.dynamic_processor import DynamicProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Extract audio features from sonified files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-d", "--data-dir", default="dataset", help="Base data directory")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["benign", "malware"],
        help="Which categories to process (subfolders of data-dir)",
    )
    parser.add_argument("--sonify", default="wav8", help="Sonification subfolder name")
    parser.add_argument(
        "--analysis-type",
        choices=["static", "dynamic"],
        default="static",
        help="Choose static or dynamic processing",
    )
    parser.add_argument(
        "--feature-type",
        choices=["manual", "wav2vec2"],
        default="manual",
        help="Type of feature extraction",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of APK folders (dynamic) / samples (static, if supported)",
    )

    # Start/end slicing: [start:end)
    parser.add_argument("--start", type=int, default=None, help="Start index (inclusive) after sorting")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive) after sorting")

    parser.add_argument("--output-dir", default="features", help="Where to store feature CSVs")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip APK folders already processed (uses a per-APK .features_done__... marker file)",
    )

    return parser.parse_args()


def _supports_start_end(processor) -> bool:
    """
    We can't reliably introspect signatures for bound methods across all styles.
    Instead, we do a cheap probe call with dry params on an empty-ish input would be risky.
    So we assume: if processor.run accepts start/end, it'll not raise TypeError.
    We'll handle TypeError at runtime and stop (to avoid recomputation).
    """
    return True


def _marker_path(apk_dir: str, analysis_type: str, feature_type: str, sonify: str) -> str:
    # Marker stored inside each APK folder
    return os.path.join(apk_dir, f".features_done__{analysis_type}__{feature_type}__{sonify}")


def _list_apk_folders(category_dir: str):
    apks = [f for f in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, f))]
    apks.sort()
    return apks


def _apply_limit_and_slice(items, limit, start, end):
    if limit is not None:
        items = items[:limit]

    s = 0 if start is None else max(0, start)
    e = len(items) if end is None else min(len(items), end)

    if s >= e:
        return [], s, e

    return items[s:e], s, e


def _run_one_index(processor, data_dir, category, sonify, output_dir, verbose, index):
    """
    Run the processor for exactly ONE APK folder via start=index, end=index+1
    (within the given category).
    """
    return processor.run(
        data_dir,
        [category],
        sonify,
        verbose=verbose,
        start=index,
        end=index + 1,
        output_dir=output_dir,
    )


def main():
    args = parseArgs()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Select feature extractor
    if args.feature_type == "manual":
        featureExtractor = ManualFeatureExtractor()
    else:
        raise ValueError("Unsupported feature type: {}".format(args.feature_type))

    # Choose processor
    if args.analysis_type == "static":
        processor = StaticProcessor(featureExtractor, maxWorkers=args.workers)
    elif args.analysis_type == "dynamic":
        processor = DynamicProcessor(featureExtractor, maxWorkers=args.workers)
    else:
        raise ValueError("Unknown analysis type: {}".format(args.analysis_type))

    # If user doesn't care about skipping, just run as before (best-effort signature)
    if not args.skip_existing:
        try:
            return processor.run(
                args.data_dir,
                args.categories,
                args.sonify,
                verbose=args.verbose,
                limit=args.limit,
                start=args.start,
                end=args.end,
                output_dir=args.output_dir,
            )
        except TypeError:
            # fallback older signatures
            try:
                return processor.run(
                    args.data_dir,
                    args.categories,
                    args.sonify,
                    verbose=args.verbose,
                    limit=args.limit,
                    output_dir=args.output_dir,
                )
            except TypeError:
                return processor.run(
                    args.data_dir,
                    args.categories,
                    args.sonify,
                    output_dir=args.output_dir,
                )
    # We enumerate APK folders ourselves so we can skip per-APK.
    # Then we invoke processor.run on one index at a time (start=i end=i+1).
    for category in args.categories:
        category_dir = os.path.join(args.data_dir, category)
        if not os.path.isdir(category_dir):
            logging.warning(f"Category directory not found: {category_dir} (skipping)")
            continue

        apks = _list_apk_folders(category_dir)
        selected, s, e = _apply_limit_and_slice(apks, args.limit, args.start, args.end)

        if not selected:
            logging.warning(f"{category}: nothing selected (range [{s}:{e}), total={len(apks)})")
            continue

        logging.info(f"{category}: selected index range [{s}:{e}) -> {len(selected)} APK(s)")

        # We need the *original* index in the full list, because processor.run does its own sorting.
        # selected corresponds to apks[s:e], so the i-th selected has global index (s + i).
        for local_i, apk_name in enumerate(selected):
            global_index = s + local_i
            apk_dir = os.path.join(category_dir, apk_name)
            marker = _marker_path(apk_dir, args.analysis_type, args.feature_type, args.sonify)

            if os.path.exists(marker):
                logging.info(f"[skip] {category}/{apk_name} (marker exists)")
                continue

            logging.info(f"[run]  {category}/{apk_name} (index {global_index})")

            # Run exactly this APK only
            try:
                _run_one_index(
                    processor=processor,
                    data_dir=args.data_dir,
                    category=category,
                    sonify=args.sonify,
                    output_dir=args.output_dir,
                    verbose=args.verbose,
                    index=global_index,
                )
            except TypeError as te:
                # Do NOT fall back here: falling back would process everything -> recompute.
                raise SystemExit(
                    "ERROR: Your processor.run() does not accept start/end, so per-APK skip cannot work.\n"
                    "Update StaticProcessor/DynamicProcessor to accept (start, end) and slice its sorted APK list.\n"
                    f"Original error: {te}"
                )

            # Mark as done AFTER successful run
            try:
                with open(marker, "w") as f:
                    f.write("done\n")
            except Exception as ex:
                logging.warning(f"Could not write marker {marker}: {ex}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
