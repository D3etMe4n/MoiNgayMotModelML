# trainers/softmax_trainer.py

import torch
import logging
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model, train_loader, test_loader, input_dim, lr=0.001, weight_decay=0):
        self.model = model.to(self._get_device())
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.input_dim = input_dim
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.training_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []

    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_model(self):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.view(-1, self.input_dim).to(self._get_device())
                labels = labels.to(self._get_device())
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy

    def train(self, epochs):
        logger.info("Starting training of Softmax Regression model...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = correct = total = 0
            for images, labels in self.train_loader:
                images = images.view(-1, self.input_dim).to(self._get_device())
                labels = labels.to(self._get_device())

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            avg_loss = total_loss / len(self.train_loader)
            train_acc = correct / total
            val_acc = self.evaluate_model()

            self.training_losses.append(avg_loss)
            self.training_accuracies.append(train_acc)
            self.validation_accuracies.append(val_acc)

            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, "
                        f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")