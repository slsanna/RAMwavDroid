import wave
import logging
from .base_sonify import BaseSonifier

logger = logging.getLogger(__name__)

class WavSonifier8Bit(BaseSonifier):
    def sonify(self, binary_data, output_path):
        try:
            with wave.open(output_path, 'wb') as wave_file:
                wave_file.setnchannels(1)
                wave_file.setsampwidth(1)
                wave_file.setframerate(self.sample_rate)
                wave_file.writeframes(binary_data)
            return True
        except Exception as error:
            logger.error(f"WAV 8-bit: {error}")
            return False

    def getFileExtension(self):
        return ".wav"

