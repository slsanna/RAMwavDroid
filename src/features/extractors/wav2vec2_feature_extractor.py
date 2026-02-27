import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from .base_dl_feature_extractor import BaseDLFeatureExtractor
import logging

logger = logging.getLogger(__name__)


class Wav2Vec2FeatureExtractor(BaseDLFeatureExtractor):
    """Wav2Vec2 feature extractor"""

    def _loadModel(self):
        self.model = Wav2Vec2Model.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name, torch_dtype=torch.float16,
                                                                          attn_implementation="flash_attention_2")
        logger.info(f"Loaded Wav2Vec2 model: {self.model_name}")

    def _preprocessAudio(self, audio_signal):
        inputs = self.feature_extractor(
            audio_signal,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        return inputs.input_values.to(self.device)

    def _extractEmbeddings(self, inputs):
        outputs = self.model(inputs)
        embeddings = outputs.last_hidden_state
        return embeddings.detach().cpu().numpy()