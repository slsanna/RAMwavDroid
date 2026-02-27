import os
import pandas as pd
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StaticProcessor:
    """
    Static audio feature pipeline compatible with the staging tree:
      <data_dir>/<category>/<apk_id>/audio/<sonify>/<audio files>
    """

    def __init__(self, featureExtractor, maxWorkers: int = 4):
        self.featureExtractor = featureExtractor
        self.maxWorkers = int(maxWorkers)

    # ---------- Public entry point expected by extract_features.py ----------
    def run(self,
            data_dir: str,
            categories: List[str],
            sonify: str,
            *,
            output_dir: str = "features",
            verbose: bool = False,
            create_combined: bool = False,
            **_ignored: Any) -> None:
        """
        Orchestrates a full extraction pass.

        Parameters
        ----------
        data_dir : root of staged dataset
        categories : e.g. ["benign", "malware"]
        sonify : e.g. "wav8"
        output_dir : where to write CSVs
        verbose : toggles info logging
        create_combined : also writes all_<sonify>_features.csv
        """
        if verbose:
            logging.getLogger().setLevel(logging.INFO)

        os.makedirs(output_dir, exist_ok=True)
        self.processDataset(baseDir=data_dir,
                            categories=categories,
                            sonifyTypes=[sonify],
                            featuresDir=output_dir)

        if create_combined:
            self.createCombinedDatasets(featuresDir=output_dir,
                                        categories=categories,
                                        sonifyTypes=[sonify])

    # ---------- Workers ----------
    def _extractFeaturesWorker(self, args):
        audioFilePath, categoryName = args
        try:
            features: Dict[str, Any] = self.featureExtractor.extractFeatures(audioFilePath)
        except Exception as e:
            logging.warning(f"Feature extraction failed for {audioFilePath}: {e}")
            return None

        if features:
            features['filename'] = os.path.basename(audioFilePath)
            features['category'] = categoryName
            return features
        return None

    # ---------- Category-level processing ----------
    def processCategoryFolder(self, categoryFolder: str, categoryName: str, sonify: str):
        """
        Scans <categoryFolder>/<apk_id>/audio/<sonify>/* and extracts features in parallel.
        """
        if not os.path.isdir(categoryFolder):
            logging.warning(f"Category folder doesn't exist: {categoryFolder}")
            return []

        extensions = ('.wav', '.mp3', '.opus', '.flac', '.mid')
        audioFiles: List[str] = []

        # Only look inside each APK's audio/<sonify> subdir
        for apk_id in os.listdir(categoryFolder):
            apk_path = os.path.join(categoryFolder, apk_id)
            if not os.path.isdir(apk_path):
                continue
            candidate_dir = os.path.join(apk_path, "audio", sonify)
            if not os.path.isdir(candidate_dir):
                continue
            for filename in os.listdir(candidate_dir):
                if filename.lower().endswith(extensions):
                    audioFiles.append(os.path.join(candidate_dir, filename))

        if not audioFiles:
            logging.warning(f"No audio files found for {categoryName} (looked in */audio/{sonify})")
            return []

        logging.info(f"Processing {len(audioFiles)} files from {categoryName} [{sonify}]")

        tasks = [(path, categoryName) for path in audioFiles]
        results = []

        # Use processes (CPU-bound feature extraction)
        with ProcessPoolExecutor(max_workers=self.maxWorkers) as executor:
            for result in tqdm(executor.map(self._extractFeaturesWorker, tasks),
                               total=len(tasks), desc=f"Processing {categoryName}:{sonify}"):
                if result:
                    results.append(result)

        return results

    # ---------- Dataset-level orchestration ----------
    def processDataset(self, baseDir: str, categories: List[str], sonifyTypes: List[str], featuresDir: str):
        """
        Walks the staged dataset and writes one CSV per (category, sonify).
        """
        os.makedirs(featuresDir, exist_ok=True)

        for category in categories:
            category_root = os.path.join(baseDir, category)
            if not os.path.isdir(category_root):
                logging.warning(f"Skipping {category_root} - doesn't exist")
                continue

            for sonify in sonifyTypes:
                logging.info(f"Processing category='{category}' sonify='{sonify}'")
                results = self.processCategoryFolder(category_root, category, sonify)

                outputCsv = os.path.join(featuresDir, f"{category}_{sonify}_features.csv")
                if results:
                    try:
                        df = pd.DataFrame(results)
                        df.to_csv(outputCsv, index=False)
                        logging.info(f"Saved {len(results)} rows -> {outputCsv}")
                    except Exception as e:
                        logging.error(f"Failed writing {outputCsv}: {e}")
                else:
                    logging.info(f"No features extracted for {category} ({sonify}); not writing {outputCsv}")

    # ---------- Optional combined CSVs ----------
    def createCombinedDatasets(self, featuresDir: str, categories: List[str], sonifyTypes: List[str]):
        logging.info("Creating combined datasets")
        for sonify in sonifyTypes:
            logging.info(f"Combining {sonify} features...")
            dfs = []
            for category in categories:
                csvPath = os.path.join(featuresDir, f"{category}_{sonify}_features.csv")
                if os.path.isfile(csvPath):
                    try:
                        df = pd.read_csv(csvPath)
                        dfs.append(df)
                        logging.info(f"Added {category} ({len(df)} samples)")
                    except Exception as e:
                        logging.error(f"Reading {csvPath}: {e}")

            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                outputPath = os.path.join(featuresDir, f"all_{sonify}_features.csv")
                combined.to_csv(outputPath, index=False)
                logging.info(f"Created {outputPath} with {len(combined)} samples")
            else:
                logging.info(f"No files found for {sonify}")
