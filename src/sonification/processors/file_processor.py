import os
import concurrent.futures
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class FileProcessor:
    def __init__(self, extractor, sonifier, max_workers=8):
        self.extractor = extractor
        self.sonifier = sonifier
        self.max_workers = max_workers
    
    def process_single_file(self, args):
        source_path, output_path = args
        try:
            binary_data, extracted = self.extractor.extract(source_path)
            if not extracted:
                return False, source_path
                
            success = self.sonifier.sonify(binary_data, output_path)
            return success, source_path
        except Exception as e:
            logger.error(f"Unhandled error on '{os.path.basename(source_path)}': {str(e)}")
            return False, source_path
    
    def process_folder(self, source_dir, output_dir, skip_existing=True, file_extension=None):
        os.makedirs(output_dir, exist_ok=True)
        
        files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
        total_files = len(files)
        
        if total_files == 0:
            logger.warning(f"No files found in {source_dir}")
            return
        
        logger.info(f"Processing {total_files} files: {source_dir} -> {output_dir}")
        
        if file_extension is None:
            file_extension = self.sonifier.getFileExtension()

        processed_count = 0
        skipped_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {}
            for filename in files:
                source_path = os.path.join(source_dir, filename)
                
                output_filename = filename
                if not output_filename.lower().endswith(file_extension):
                    output_filename = f"{output_filename}{file_extension}"
                output_path = os.path.join(output_dir, output_filename)

                if skip_existing and os.path.exists(output_path):
                    skipped_count += 1
                    continue

                future = executor.submit(self.process_single_file, (source_path, output_path))
                future_to_file[future] = filename

            success_count = 0
            failure_count = 0

            for future in tqdm(concurrent.futures.as_completed(future_to_file),
                              total=len(future_to_file),
                              desc=f"Processing files"):
                filename = future_to_file[future]
                try:
                    success, file_path = future.result()
                    if success:
                        success_count += 1
                    else:
                        failure_count += 1
                        logger.warning(f"Failed: {os.path.basename(file_path)}")
                except Exception as e:
                    failure_count += 1
                    logger.error(f"{filename}: {e}")

        processed_count = success_count + failure_count
        logger.info(f"Complete: {success_count}/{total_files} success, {skipped_count} skipped, {failure_count} failed")
