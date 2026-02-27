import torch
import numpy as np
import logging
from .base_feature_extractor import BaseFeatureExtractor
from abc import abstractmethod
from typing import Dict, Union

logger = logging.getLogger(__name__)


class BaseDLFeatureExtractor(BaseFeatureExtractor):
    
    def __init__(self, model_name: str, device: str = None, sample_rate: int = 16000, 
                 extract_embeddings_from: str = 'last_hidden_state'):
        super().__init__(sample_rate)
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.extract_embeddings_from = extract_embeddings_from
        self.model = None
        self.feature_extractor = None
        logger.info(f"Initializing {self.__class__.__name__} on {self.device}")
        self._loadModel()

    @abstractmethod
    def _loadModel(self):
        pass

    @abstractmethod
    def _preprocessAudio(self, audio_signal: np.ndarray) -> torch.Tensor:
        pass

    @abstractmethod
    def _extractEmbeddings(self, inputs: torch.Tensor) -> np.ndarray:
        pass

    def _extractFeaturesFromAudio(self, audio_signal, sample_rate: int) -> Dict[str, Union[float, int]]:
        try:
            inputs = self._preprocessAudio(audio_signal)
            with torch.no_grad():
                embeddings = self._extractEmbeddings(inputs)
            
            if len(embeddings.shape) == 3:
                embeddings_mean = np.mean(embeddings, axis=1).squeeze()
                embeddings_std = np.std(embeddings, axis=1).squeeze()
                embeddings_max = np.max(embeddings, axis=1).squeeze()
            else:
                embeddings_mean = embeddings.squeeze()
                embeddings_std = np.zeros_like(embeddings_mean)
                embeddings_max = embeddings_mean
            
            features = {}
            for i, value in enumerate(embeddings_mean.flatten()):
                features[f'{self.model_name.split("/")[-1]}_mean_{i+1}'] = float(value)
            
            for i, value in enumerate(embeddings_std.flatten()):
                features[f'{self.model_name.split("/")[-1]}_std_{i+1}'] = float(value)
                
            for i, value in enumerate(embeddings_max.flatten()):
                features[f'{self.model_name.split("/")[-1]}_max_{i+1}'] = float(value)
            
            features['dl_embedding_dim'] = embeddings_mean.shape[-1] if len(embeddings_mean.shape) > 0 else 1
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting deep learning features: {e}")
            return {}

    def getFeatureDimension(self):
        return 768 * 3