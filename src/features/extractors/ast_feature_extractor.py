import numpy as np
import torch
from transformers import AutoModel, AutoFeatureExtractor
from .base_dl_feature_extractor import BaseDLFeatureExtractor
import logging

logger = logging.getLogger(__name__)


class ASTFeatureExtractor(BaseDLFeatureExtractor):
    def _loadModel(self):
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        logger.info(f"Loaded AST model: {self.model_name}")

    def _preprocessAudio(self, audio_signal):
        inputs = self.feature_extractor(
            audio_signal,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        return inputs.input_values.to(self.device)

    def _extractEmbeddings(self, inputs):
        outputs = self.model(inputs)
        if self.extract_embeddings_from == 'pooler_output' and hasattr(outputs, 'pooler_output'):
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state
        return embeddings.cpu().numpy()
