import os
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from .classifier_base import BaseTrainer

class RandomForestTrainer(BaseTrainer):
    def __init__(self, numEstimators=100, maxDepth=None, randomState=42):
        super().__init__()
        self.numEstimators = numEstimators
        self.maxDepth = maxDepth
        self.randomState = randomState
        self.buildModel()

    def buildModel(self):
        logging.info(f"Building RandomForest model with {self.numEstimators} estimators, maxDepth={self.maxDepth}")
        self.model = RandomForestClassifier(
            n_estimators=self.numEstimators,
            max_depth=self.maxDepth,
            random_state=self.randomState
        )

    def train(self, trainFeatures, trainLabels):
        logging.info(f"Training RandomForest on dataset: {trainFeatures.shape[0]} samples, {trainFeatures.shape[1]} features")
        self.model.fit(trainFeatures, trainLabels)
        logging.info("Training completed")

    def evaluate(self, testFeatures, testLabels):
        logging.info(f"Evaluating RandomForest model on {testFeatures.shape[0]} test samples")
        modelPredictions = self.model.predict(testFeatures)
        logging.info("Confusion Matrix:")
        confusionMatrix = confusion_matrix(testLabels, modelPredictions)
        print(confusionMatrix)
        logging.info("Classification Report:")
        classificationReport = classification_report(testLabels, modelPredictions, digits=4)
        print(classificationReport)
        accuracy = (modelPredictions == testLabels).mean()
        logging.info(f"Evaluation accuracy: {accuracy:.4f}")

    def saveModel(self, modelPath):
        os.makedirs(os.path.dirname(modelPath), exist_ok=True)
        joblib.dump(self.model, modelPath)
        logging.info(f"Model saved to: {modelPath}")

    def loadModel(self, modelPath):
        if os.path.isfile(modelPath):
            self.model = joblib.load(modelPath)
            logging.info(f"Model loaded from: {modelPath}")
        else:
            logging.error(f"Model file not found: {modelPath}")
            raise FileNotFoundError(f"Model file not found: {modelPath}")
