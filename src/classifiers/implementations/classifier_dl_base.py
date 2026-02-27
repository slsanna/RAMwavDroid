from abc import ABC, abstractmethod

class BaseDlTrainer(ABC):

    def __init__(self):
        self.model = None
        self.optimizer = None
        self.criterion = None

    @abstractmethod
    def build_model(self, input_dimensions, num_classes):
        pass

    @abstractmethod
    def train(self, train_dataloader, num_epochs, device_name):
        pass

    @abstractmethod
    def evaluate(self, test_dataloader, device_name):
        pass

    @abstractmethod
    def save_model(self, model_path):
        pass

    @abstractmethod
    def load_model(self, model_path, device_name):
        pass

