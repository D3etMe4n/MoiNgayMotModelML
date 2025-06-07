# models/xgboost.py

import numpy as np
import torch
from xgboost import XGBClassifier

class XGBoostWrapper:
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=num_classes,
            eval_metric='mlogloss',
            tree_method='hist',  # Tăng tốc độ train
            use_label_encoder=False,
            verbosity=1
        )

    def fit(self, X, y):
        """Huấn luyện mô hình XGBoost"""
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        self.model.fit(X, y)

    def predict(self, X):
        """Dự đoán trên dữ liệu numpy array"""
        return self.model.predict(X)

    def predict_tensor(self, X_tensor):
        """Dự đoán từ tensor của PyTorch"""
        X_flat = X_tensor.view(-1, self.input_dim).cpu().numpy()
        pred = self.model.predict(X_flat)
        return torch.tensor(pred, dtype=torch.int64)