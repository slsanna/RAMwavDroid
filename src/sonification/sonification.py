import os
import argparse
import logging
from tqdm import tqdm

from extractors.file_extractor import ApkExtractor
from extractors.dataset_ram_extractor import DatasetRAMExtractor
from processors.file_processor import FileProcessor
from sonifiers.wav_sonifier_16bit import WavSonifier16Bit
from sonifiers.wav_sonifier_8bit import WavSonifier8Bit
from sonifiers.mp3_sonifier import Mp3Sonifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Sonifier:
    SONIFIER_CONFIG = {
        'wav8': (WavSonifier8Bit, "Wav", ".wav"),
        'wav16': (WavSonifier16Bit, "Wav16bit", ".wav"),
        'mp3': (Mp3Sonifier, "Mp3", ".mp3"),
    }

    LEGACY_OUTPUT_ROOT = "/home/ssanna/Desktop/malware_ram/Android/audio/malware-sonification-signal-processing/apk"

    CATEGORY_MAP_TO_GROUP = {
        "Benign": "benign",
        "Adware": "malware",
        "Banking": "malware",
        "SMS": "malware",
        "Riskware": "malware",
        "benign_dumps": "benign",
        "dumps_dataset": "malware",
    }

    def __init__(self, sonificationType='wav8', sampleRate=32768, workers=8, datasetType='legacy'):
        if sonificationType not in self.SONIFIER_CONFIG:
            raise ValueError(f"Unknown sonification type: {sonificationType}")

        self.datasetType = datasetType
        self.extractor = ApkExtractor() if datasetType == 'legacy' else DatasetRAMExtractor()
        self.sampleRate = sampleRate
        self.sonificationType = sonificationType
        self.workers = workers

        sonifierClass, self.outputSuffix, self.fileExtension = self.SONIFIER_CONFIG[sonificationType]
        self.sonifier = sonifierClass(sampleRate)
        self.processor = FileProcessor(self.extractor, self.sonifier, max_workers=workers)

        # existing
        self.apk_limit = None

        # ✅ NEW: index range control (applies to datasetRAM folder iteration)
        self.start_index = None  # inclusive
        self.end_index = None    # exclusive

    def processCategory(self, category, baseDir="dataset"):
        if self.datasetType == 'legacy':
            self._processLegacyCategory(category, baseDir)
        else:
            self._processDatasetRamCategory(category, baseDir)

    def _processLegacyCategory(self, category, baseDir):
        sourceDir = os.path.join(baseDir, category)

        if not os.path.exists(sourceDir) or not os.listdir(sourceDir):
            logging.warning(f"Folder '{sourceDir}' does not exist or is empty. Skipping...")
            return

        logging.info(f"\n{'=' * 50}")
        logging.info(f"[LEGACY] Processing category: {category}")
        logging.info(f"{'=' * 50}")
        logging.info(f"Sonification method: {self.sonificationType}, Sample rate: {self.sampleRate}")

        group = self.CATEGORY_MAP_TO_GROUP.get(category, category.lower())
        if group not in ("benign", "malware"):
            logging.warning(f"Unknown legacy category '{category}', defaulting to 'malware' output.")
            group = "malware"

        outputDir = os.path.join(self.LEGACY_OUTPUT_ROOT, group)
        os.makedirs(outputDir, exist_ok=True)

        logging.info(f"[LEGACY] Output dir: {outputDir}")
        self.processor.process_folder(sourceDir, outputDir, file_extension=self.fileExtension)

    def _processDatasetRamCategory(self, category, baseDir):
        sourceDir = os.path.join(baseDir, category)

        if not os.path.exists(sourceDir) or not os.listdir(sourceDir):
            logging.warning(f"Folder '{sourceDir}' does not exist or is empty. Skipping...")
            return

        logging.info(f"\n{'=' * 50}")
        logging.info(f"[datasetRAM] Processing category: {category}")
        logging.info(f"{'=' * 50}")
        logging.info(f"Sonification method: {self.sonificationType}, Sample rate: {self.sampleRate}")

        apkFolders = [f for f in os.listdir(sourceDir) if os.path.isdir(os.path.join(sourceDir, f))]
        apkFolders.sort()

        # existing limit (top-N)
        if self.apk_limit:
            apkFolders = apkFolders[:self.apk_limit]

        # ✅ NEW: slice by index range (after sort, after limit)
        start = 0 if self.start_index is None else max(0, self.start_index)
        end = len(apkFolders) if self.end_index is None else min(len(apkFolders), self.end_index)

        if start >= end:
            logging.warning(
                f"No APKs to process for {category}: start={start}, end={end}, total={len(apkFolders)}"
            )
            return

        apkFolders = apkFolders[start:end]
        logging.info(f"[datasetRAM] Processing index range [{start}:{end}) => {len(apkFolders)} APK folder(s).")

        for apkFolder in tqdm(apkFolders, desc=f"Processing {category}"):
            apkPath = os.path.join(sourceDir, apkFolder)
            logging.info(f"Processing APK: {apkFolder}")

            relevantAddressesDir = os.path.join(apkPath, 'r--')

            if not os.path.exists(relevantAddressesDir):
                logging.warning(f"Missing r-- folder in {apkFolder}")
                continue

            dumpFiles = [f for f in os.listdir(relevantAddressesDir) if f.endswith("_dump.data")]

            if not dumpFiles:
                logging.warning(f"No .data files found in r-- for {apkFolder}")
                continue

            apkOutputDir = os.path.join(apkPath, "audio", self.sonificationType)

            if os.path.exists(apkOutputDir):
                expectedFiles = {f.replace("_dump.data", self.fileExtension) for f in dumpFiles}
                existingFiles = {f for f in os.listdir(apkOutputDir) if f.endswith(self.fileExtension)}
                missingFiles = expectedFiles - existingFiles
                zeroByteFiles = {f for f in existingFiles if os.path.getsize(os.path.join(apkOutputDir, f)) == 0}

                if not missingFiles:
                    logging.info(f"Skipping {apkFolder} — already fully sonified ({len(expectedFiles)} files).")
                    continue
                elif zeroByteFiles:
                    logging.info(f"Resuming {apkFolder} — {len(missingFiles)} files missing.")

            os.makedirs(apkOutputDir, exist_ok=True)

            processedCount = 0

            for dumpFilename in dumpFiles:
                address = dumpFilename.split("_")[0]
                dumpPath = os.path.join(relevantAddressesDir, dumpFilename)
                outputFilename = f"{address}{self.fileExtension}"
                outputPath = os.path.join(apkOutputDir, outputFilename)

                try:
                    binaryData, extracted = self.extractor.extract(dumpPath)
                    if extracted and binaryData:
                        success = self.sonifier.sonify(binaryData, outputPath)
                        if success:
                            processedCount += 1
                        else:
                            logging.warning(f"Failed to sonify {dumpFilename}")
                except Exception as e:
                    logging.error(f"Processing {dumpFilename}: {e}")

            logging.info(f"Processed {processedCount} memory regions for {apkFolder}")

    def processAllCategories(self, categories, baseDir="dataset"):
        os.makedirs(baseDir, exist_ok=True)
        os.makedirs(os.path.join(baseDir, "features"), exist_ok=True)

        for category in categories:
            os.makedirs(os.path.join(baseDir, category), exist_ok=True)
            self.processCategory(category, baseDir)


def parseArgs():
    parser = argparse.ArgumentParser(description="APK sonification tool")
    parser.add_argument("-d", "--data-dir", default="dataset", help="Base data directory")
    parser.add_argument("-w", "--workers", type=int, default=8, help="Worker threads")
    parser.add_argument("-r", "--sample-rate", type=int, default=32768, help="Sample rate")
    parser.add_argument(
        "-c", "--categories", nargs="+",
        help="Categories to process (default: legacy=['Adware','Banking','SMS','Riskware','Benign'], datasetRAM=['benign','malicious'])"
    )
    parser.add_argument(
        "-s", "--sonify", nargs="+",
        choices=["wav8", "wav16", "mp3"],
        default=["wav8"], help="Sonification methods"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--dataset-type", choices=["legacy", "datasetRAM"], default="legacy",
        help="Dataset format: legacy (APK files) or datasetRAM (memory dumps)"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of APKs to process per category")

    parser.add_argument("--start", type=int, default=None, help="Start index (inclusive) after sorting")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive) after sorting")

    return parser.parse_args()


def main():
    args = parseArgs()

    if args.categories is None:
        if args.dataset_type == "legacy":
            args.categories = ["Adware", "Banking", "SMS", "Riskware", "Benign"]
        else:
            args.categories = ["benign", "malicious"]

    if args.verbose:
        logging.info(f"Processing: {args.data_dir} | Categories: {', '.join(args.categories)}")
        logging.info(f"Methods: {', '.join(args.sonify)} | Rate: {args.sample_rate} | Workers: {args.workers}")
        logging.info(f"Dataset type: {args.dataset_type}")
        logging.info(f"Index range: start={args.start}, end={args.end}, limit={args.limit}")

    for sonifyType in args.sonify:
        logging.info(f"Processing {sonifyType} sonification...")
        sonifier = Sonifier(sonifyType, args.sample_rate, args.workers, args.dataset_type)
        sonifier.apk_limit = args.limit

        # ✅ NEW: pass index range
        sonifier.start_index = args.start
        sonifier.end_index = args.end

        sonifier.processAllCategories(args.categories, args.data_dir)


if __name__ == "__main__":
    main()
