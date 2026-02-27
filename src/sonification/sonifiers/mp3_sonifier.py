import logging
from .base_sonify import BaseSonifier

logger = logging.getLogger(__name__)

class Mp3Sonifier(BaseSonifier):
    def __init__(self, sample_rate=32768, bitrate="192k"):
        super().__init__(sample_rate)
        self.bitrate = bitrate

    def sonify(self, binary_data, output_path):
        try:
            seg = self._bytesToAudiosegment(binary_data, self.sample_rate)
            seg.export(output_path, format="mp3", bitrate=self.bitrate)
            return True
        except Exception as exc:
            logger.error(f"MP3: {exc}")
            return False

    def getFileExtension(self):
        return ".mp3"

