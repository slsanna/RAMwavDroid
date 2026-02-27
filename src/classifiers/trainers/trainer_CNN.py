import os
import torch
import torch.nn as nn
import logging
from sklearn.metrics import classification_report, confusion_matrix

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..implementations.classifier_dl_base import BaseDlTrainer
from ..implementations.simple_CNN import SampleCNN


class CnnTrainer(BaseDlTrainer):
    def build_model(self, input_shape, num_classes):
        logging.info(f"Building CNN model with {num_classes} classes")
        self.model = SampleCNN(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader, epochs, device):
        logging.info(f"Training model on {device} for {epochs} epochs")
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()
        for epoch in range(epochs):
            totalLoss = 0
            correctPredictions = 0
            totalSamples = 0
            for audioFeatures, labels in train_loader:
                audioFeatures, labels = audioFeatures.to(device), labels.to(device)
                if audioFeatures.ndim == 2:
                    audioFeatures = audioFeatures.unsqueeze(1)
                self.optimizer.zero_grad()
                modelOutput = self.model(audioFeatures)
                loss = self.criterion(modelOutput, labels)
                loss.backward()
                self.optimizer.step()
                totalLoss += loss.item() * audioFeatures.size(0)
                predictions = torch.argmax(modelOutput, dim=1)
                correctPredictions += (predictions == labels).sum().item()
                totalSamples += labels.size(0)
            accuracy = correctPredictions / totalSamples
            avgLoss = totalLoss / totalSamples
            logging.info(f"Epoch {epoch + 1}/{epochs}: Loss={avgLoss:.4f}, Accuracy={accuracy:.4f}")
            print(f"Epoch {epoch + 1}/{epochs}: Loss={avgLoss:.4f}, Accuracy={accuracy:.4f}")

    def evaluate(self, test_loader, device):
        logging.info(f"Evaluating model on {device}")
        self.model.to(device)
        self.model.eval()
        allTrueLabels = []
        allPredictions = []
        with torch.no_grad():
            for audioFeatures, labels in test_loader:
                audioFeatures, labels = audioFeatures.to(device), labels.to(device)
                if audioFeatures.ndim == 2:
                    audioFeatures = audioFeatures.unsqueeze(1)
                modelOutput = self.model(audioFeatures)
                predictions = torch.argmax(modelOutput, dim=1)
                allTrueLabels.append(labels)
                allPredictions.append(predictions)

        trueLabels = torch.cat(allTrueLabels).cpu().numpy()
        predictedLabels = torch.cat(allPredictions).cpu().numpy()

        logging.info("Confusion Matrix:")
        confusionMatrix = confusion_matrix(trueLabels, predictedLabels)
        print(confusionMatrix)
        logging.info("Classification Report:")
        classificationReport = classification_report(trueLabels, predictedLabels, digits=4)
        print(classificationReport)

        accuracy = (predictedLabels == trueLabels).mean()
        logging.info(f"Evaluation accuracy: {accuracy:.4f}")

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logging.info(f"Model saved to {path}")

    def load_model(self, path, device):
        logging.info(f"Loading model from {path}")
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.to(device)
        logging.info(f"Model loaded successfully and moved to {device}")
