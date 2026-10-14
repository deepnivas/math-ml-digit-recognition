import numpy as np


class PCAFromScratch:

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):

        X = np.asarray(X)

        self.mean_ = np.mean(X, axis=0)

        X_centered = X - self.mean_

        covariance_matrix = (
            X_centered.T @ X_centered
        ) / (X_centered.shape[0] - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance_matrix
        )

        sorted_indices = np.argsort(
            eigenvalues
        )[::-1]

        eigenvalues = eigenvalues[sorted_indices]

        eigenvectors = eigenvectors[:, sorted_indices]

        self.explained_variance_ = eigenvalues[
            :self.n_components
        ]

        self.components_ = eigenvectors[
            :, :self.n_components
        ]

        total_variance = np.sum(eigenvalues)

        self.explained_variance_ratio_ = (
            self.explained_variance_
            / total_variance
        )

        return self

    def transform(self, X):

        X = np.asarray(X)

        X_centered = X - self.mean_

        return X_centered @ self.components_

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)
