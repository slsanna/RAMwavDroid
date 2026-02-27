import numpy as np
import librosa
import logging
import warnings
from .base_feature_extractor import BaseFeatureExtractor
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHROMA_FEATURES_COUNT = 12
MFCC_FEATURES_COUNT = 20
POLY_FEATURES_COUNT = 2
TONNETZ_FEATURES_COUNT = 6
DEFAULT_SAMPLE_RATE = 32768

CHROMA_FEATURES_COUNT = 12
MFCC_FEATURES_COUNT = 20
POLY_FEATURES_COUNT = 2
TONNETZ_FEATURES_COUNT = 6

DEFAULT_SAMPLE_RATE = 32768


class ManualFeatureExtractor(BaseFeatureExtractor):
    
    def __init__(self, sampleRate: int = DEFAULT_SAMPLE_RATE):
        super().__init__(sampleRate)
        self.featureNames = self._getFeatureNames()
    
    def _getFeatureNames(self):
        featureList = ['sample_rate', 'num_samples', 'duration_sec', 'rms_mean', 'rms_std',
                      'spec_centroid_mean', 'spec_centroid_std', 'spec_bw_mean', 'spec_bw_std',
                      'rolloff_mean', 'rolloff_std', 'zcr_mean', 'zcr_std']
        
        for i in range(CHROMA_FEATURES_COUNT):
            featureList.extend([f'chroma{i + 1}_mean', f'chroma{i + 1}_std'])
        
        for i in range(MFCC_FEATURES_COUNT):
            featureList.extend([f'mfcc{i + 1}_mean', f'mfcc{i + 1}_std'])
        
        for i in range(POLY_FEATURES_COUNT):
            featureList.extend([f'poly{i + 1}_mean', f'poly{i + 1}_std'])
        
        for i in range(TONNETZ_FEATURES_COUNT):
            featureList.extend([f'tonnetz{i + 1}_mean', f'tonnetz{i + 1}_std'])
        
        return featureList
    
    def _extractFeaturesFromAudio(self, audioSignal, sampleRate: int):
        extractedFeatures = {}

        rmsEnergy = librosa.feature.rms(y=audioSignal)
        extractedFeatures['rms_mean'] = float(np.mean(rmsEnergy))
        extractedFeatures['rms_std'] = float(np.std(rmsEnergy))

        spectralCentroid = librosa.feature.spectral_centroid(y=audioSignal, sr=sampleRate)
        extractedFeatures['spec_centroid_mean'] = float(np.mean(spectralCentroid))
        extractedFeatures['spec_centroid_std'] = float(np.std(spectralCentroid))

        spectralBandwidth = librosa.feature.spectral_bandwidth(y=audioSignal, sr=sampleRate)
        extractedFeatures['spec_bw_mean'] = float(np.mean(spectralBandwidth))
        extractedFeatures['spec_bw_std'] = float(np.std(spectralBandwidth))

        spectralRolloff = librosa.feature.spectral_rolloff(y=audioSignal, sr=sampleRate)
        extractedFeatures['rolloff_mean'] = float(np.mean(spectralRolloff))
        extractedFeatures['rolloff_std'] = float(np.std(spectralRolloff))

        zeroCrossingRate = librosa.feature.zero_crossing_rate(audioSignal)
        extractedFeatures['zcr_mean'] = float(np.mean(zeroCrossingRate))
        extractedFeatures['zcr_std'] = float(np.std(zeroCrossingRate))

        chromaFeatures = librosa.feature.chroma_stft(y=audioSignal, sr=sampleRate)
        for i in range(chromaFeatures.shape[0]):
            extractedFeatures[f'chroma{i + 1}_mean'] = float(np.mean(chromaFeatures[i]))
            extractedFeatures[f'chroma{i + 1}_std'] = float(np.std(chromaFeatures[i]))

        mfccFeatures = librosa.feature.mfcc(y=audioSignal, sr=sampleRate, n_mfcc=MFCC_FEATURES_COUNT)
        for i in range(mfccFeatures.shape[0]):
            extractedFeatures[f'mfcc{i + 1}_mean'] = float(np.mean(mfccFeatures[i]))
            extractedFeatures[f'mfcc{i + 1}_std'] = float(np.std(mfccFeatures[i]))

        polynomialFeatures = librosa.feature.poly_features(y=audioSignal, sr=sampleRate)
        for i in range(polynomialFeatures.shape[0]):
            extractedFeatures[f'poly{i + 1}_mean'] = float(np.mean(polynomialFeatures[i]))
            extractedFeatures[f'poly{i + 1}_std'] = float(np.std(polynomialFeatures[i]))

        harmonicComponent = librosa.effects.harmonic(audioSignal)
        tonnetzFeatures = librosa.feature.tonnetz(y=harmonicComponent, sr=sampleRate)
        for i in range(tonnetzFeatures.shape[0]):
            extractedFeatures[f'tonnetz{i + 1}_mean'] = float(np.mean(tonnetzFeatures[i]))
            extractedFeatures[f'tonnetz{i + 1}_std'] = float(np.std(tonnetzFeatures[i]))
        
        return extractedFeatures
    
    def getFeatureDimension(self):
        return len(self.featureNames)