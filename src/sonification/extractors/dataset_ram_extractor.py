import os
import logging

logger = logging.getLogger(__name__)

class DatasetRAMExtractor:
    @staticmethod
    def extract(dump_path):
        try:
            with open(dump_path, 'rb') as f:
                data = f.read()
            return data, True
        except Exception as e:
            logger.error(f"Failed to read {dump_path}: {e}")
            return None, False