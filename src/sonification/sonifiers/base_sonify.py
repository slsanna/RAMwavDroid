from abc import ABC, abstractmethod
from pydub import AudioSegment


class BaseSonifier(ABC):

    def __init__(self, sample_rate=32768):
        self.sample_rate = sample_rate

    @staticmethod
    def _bytesToAudiosegment(data, sample_rate, sample_width=1):
        return AudioSegment(
            data=data,
            sample_width=sample_width,
            frame_rate=sample_rate,
            channels=1,
        )

    @abstractmethod
    def sonify(self, binary_data, output_path):
        pass

    @abstractmethod
    def getFileExtension(self):
        pass
