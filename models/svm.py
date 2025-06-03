# models/svm.py

from sklearn.svm import LinearSVC, SVC
import numpy as np
import torch

class SVMModelWrapper:
    def __init__(self,
                 input_dim,
                 num_classes,
                 kernel="linear",      # thêm tham số kernel
                 C=1e-3,
                 gamma="scale",        # dùng cho rbf
                 max_iter=1000):
        self.input_dim = input_dim
        self.num_classes = num_classes
        if kernel == "linear":
            self.model = LinearSVC(max_iter=max_iter, C=C, verbose=False)
        else:
            # dùng SVC cho các kernel phi tuyến
            self.model = SVC(kernel=kernel, C=C, gamma=gamma, verbose=False, max_iter=max_iter)

    def fit(self, X, y):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_tensor(self, X_tensor):
        X_flat = X_tensor.view(-1, self.input_dim).cpu().numpy()
        pred = self.model.predict(X_flat)
        return torch.tensor(pred, dtype=torch.int64)