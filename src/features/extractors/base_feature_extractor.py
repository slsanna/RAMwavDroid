import os
import librosa
import logging
from abc import ABC, abstractmethod
from typing import Dict, Union, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseFeatureExtractor(ABC):
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        logger.info(f"Initializing {self.__class__.__name__}")
    
    @abstractmethod
    def _extractFeaturesFromAudio(self, audio_signal, sample_rate: int) -> Dict[str, Union[float, int]]:
        pass
    
    def extractFeatures(self, audio_file_path: str) -> Optional[Dict[str, Union[float, int]]]:
        try:
            audio_signal, sr = librosa.load(audio_file_path, sr=self.sample_rate)
            
            if len(audio_signal) == 0:
                logger.warning(f"Empty audio file: {audio_file_path}")
                return None
            
            features = self._extractFeaturesFromAudio(audio_signal, sr)
            
            features.update({
                'filename': os.path.basename(audio_file_path),
                'sample_rate': sr,
                'duration_sec': len(audio_signal) / sr,
                'num_samples': len(audio_signal)
            })
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing {audio_file_path}: {e}")
            return None
    
    def getFeatureNames(self) -> list:
        if hasattr(self, 'feature_names'):
            return self.feature_names
        else:
            return []
    
    def getFeatureDimension(self) -> int:
        return len(self.getFeatureNames())