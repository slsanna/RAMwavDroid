import wave
import logging
from .base_sonify import BaseSonifier

logger = logging.getLogger(__name__)

class WavSonifier16Bit(BaseSonifier):
    def sonify(self, binary_data, output_path):
        try:
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(binary_data)
            return True
        except Exception as error:
            logger.error(f"WAV 16-bit: {error}")
            return False

    def getFileExtension(self):
        return ".wav"

