from abc import ABC, abstractmethod

class BaseTrainer(ABC):

    def __init__(self):
        self.model = None

    @abstractmethod
    def buildModel(self):
        pass

    @abstractmethod
    def train(self, trainFeatures, trainLabels):
        pass

    @abstractmethod
    def evaluate(self, testFeatures, testLabels):
        pass

    @abstractmethod
    def saveModel(self, modelPath):
        pass

    @abstractmethod
    def loadModel(self, modelPath):
        pass
