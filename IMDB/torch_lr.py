import numpy as np
from sklearn.base import BaseEstimator
import torch


class TorchLinearRegression(BaseEstimator):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        super().__init__()

    def fit(self, X, y, sample_weight=None):
        with torch.no_grad():
            X = torch.from_numpy(X).to(torch.float32)
            y = torch.from_numpy(y).to(torch.float32)

            if self.fit_intercept:
                X = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)], dim=1)

            if sample_weight is not None:
                W = torch.from_numpy(sample_weight).to(torch.float32)
            else:
                W = torch.ones_like(y)
            W = W.squeeze().unsqueeze(0)

            X = X.cpu()
            y = y.cpu()
            W = W.cpu()
            W2 = torch.sqrt(W).squeeze().unsqueeze(1)
            if len(y.shape) == 1:
                y = y.unsqueeze(1)

            self.coef_ = torch.lstsq(y * W2, X * W2)[0][:X.shape[1]].squeeze()

            if self.fit_intercept:
                self.intercept_ = self.coef_[-1].detach().cpu()
                self.coef_ = self.coef_[:-1]
            self.coef_ = self.coef_.t().detach().cpu()

        return self

    def predict(self, X):
        if self.fit_intercept:
            return np.matmul(X, np.transpose(self.coef_)) + self.intercept_
        else:
            return np.matmul(X, np.transpose(self.coef_))


import torch
from torch import nn
import torch.nn.functional as F


class TorchRidge:
    def __init__(self, alpha=0, fit_intercept=True, ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X: torch.tensor, y: torch.tensor, sample_weight=None) -> None:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).to(torch.float32)
        if isinstance(sample_weight, np.ndarray):
            sample_weight = torch.from_numpy(sample_weight).to(torch.float32)

        X = X.rename(None)
        y = y.rename(None).squeeze().unsqueeze(1)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
        assert (len(y.shape) == 2)
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1), X], dim=1)

        if sample_weight is not None:
            W2 = torch.sqrt(sample_weight).squeeze().unsqueeze(1)
            y = y * W2
            X = X * W2

        # Solving X*w = y with Normal equations:
        # X^{T}*X*w = X^{T}*y
        lhs = X.T @ X
        rhs = X.T @ y
        if self.alpha == 0:
            self.w, _ = torch.lstsq(rhs, lhs)
        else:
            ridge = self.alpha * torch.eye(lhs.shape[0])
            self.w, _ = torch.lstsq(rhs, lhs + ridge)

        if self.fit_intercept:
            self.intercept_ = self.w[0].detach().cpu()
            self.coef_ = self.w[1:]
        else:
            self.coef_ = self.w
        self.coef_ = self.coef_.t().detach().cpu()

    def predict(self, X: torch.tensor) -> None:
        X = X.rename(None)
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1), X], dim=1)
        return X @ self.w


if __name__ == "__main__":
    X = np.random.randn(10000, 100)
    w = np.random.randn(100, 1)
    y = np.matmul(X, w) + 1.0

    model = TorchLinearRegression().fit(X, y)
    predictions = model.predict(X)
    print((predictions - y).mean())
