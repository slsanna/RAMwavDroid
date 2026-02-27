import numpy as np
from .wav2vec2_feature_extractor import Wav2Vec2FeatureExtractor
from .ast_feature_extractor import ASTFeatureExtractor

class DLFeatureExtractor:
    def __init__(self, feature_type='wav2vec2', device=None, sample_rate=16000):
        self.feature_type = feature_type
        
        if feature_type == 'wav2vec2':
            model_name = "facebook/wav2vec2-base-960h"
            self.extractor = Wav2Vec2FeatureExtractor(
                model_name=model_name,
                device=device,
                sample_rate=sample_rate
            )
        elif feature_type == 'ast':
            model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
            self.extractor = ASTFeatureExtractor(
                model_name=model_name,
                device=device,
                sample_rate=sample_rate
            )
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def extractFeatures(self, audio_file_path):
        return self.extractor.extractFeatures(audio_file_path)
    
    def getFeatureDimension(self):
        return self.extractor.getFeatureDimension()
    
    def getFeatureNames(self):
        return self.extractor.getFeatureNames()