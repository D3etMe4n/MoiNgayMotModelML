# trainers/xgboost_trainer.py

import logging
import numpy as np
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

class XGBoostTrainer:
    def __init__(self, model, train_loader, test_loader, input_dim):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.input_dim = input_dim
        self.train_features = []
        self.train_labels = []
        self.val_features = []
        self.val_labels = []

    def _load_all_data(self):
        logger.info("Loading training data for XGBoost...")
        self.train_features = []
        self.train_labels = []
        for images, labels in self.train_loader:
            images = images.view(-1, self.input_dim).numpy()
            labels = labels.numpy()
            self.train_features.append(images)
            self.train_labels.append(labels)

        self.train_features = np.vstack(self.train_features)
        self.train_labels = np.hstack(self.train_labels)

        logger.info("Loading validation data for XGBoost...")
        self.val_features = []
        self.val_labels = []
        for images, labels in self.test_loader:
            images = images.view(-1, self.input_dim).numpy()
            labels = labels.numpy()
            self.val_features.append(images)
            self.val_labels.append(labels)

        self.val_features = np.vstack(self.val_features)
        self.val_labels = np.hstack(self.val_labels)

    def train(self):
        self._load_all_data()
        logger.info("Training XGBoost model...")
        self.model.fit(self.train_features, self.train_labels)
        logger.info("Training complete.")

    def evaluate(self):
        preds = self.model.predict(self.val_features)
        acc = accuracy_score(self.val_labels, preds)
        logger.info(f"XGBoost Validation Accuracy: {acc:.4f}")
        return acc