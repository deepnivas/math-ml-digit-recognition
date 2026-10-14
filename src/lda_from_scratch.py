import numpy as np
from sklearn.datasets import load_digits


class LDAFromScratch:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.means_ = None
        self.classes_ = None

    def fit(self, X, y):

        self.classes_ = np.unique(y)

        n_features = X.shape[1]

        # Overall mean
        overall_mean = np.mean(X, axis=0)

        # Within-class scatter matrix
        S_W = np.zeros(
            (n_features, n_features)
        )

        # Between-class scatter matrix
        S_B = np.zeros(
            (n_features, n_features)
        )

        self.means_ = {}

        # Calculate scatter matrices
        for cls in self.classes_:

            X_class = X[y == cls]

            class_mean = np.mean(
                X_class,
                axis=0
            )

            self.means_[cls] = class_mean

            # Within-class scatter
            X_centered = X_class - class_mean

            S_W += X_centered.T @ X_centered

            # Between-class scatter
            n_class = X_class.shape[0]

            mean_difference = (
                class_mean - overall_mean
            ).reshape(-1, 1)

            S_B += (
                n_class
                * (mean_difference @ mean_difference.T)
            )

        # Regularization
        regularization = 1e-6 * np.eye(
            n_features
        )

        S_W_regularized = (
            S_W + regularization
        )

        # Solve:
        #
        # S_B w = lambda S_W w
        #
        # by calculating inv(S_W) S_B

        matrix = (
            np.linalg.pinv(S_W_regularized)
            @ S_B
        )

        eigenvalues, eigenvectors = np.linalg.eig(
            matrix
        )

        # Remove tiny numerical imaginary parts
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        # Sort by largest eigenvalue
        indices = np.argsort(
            eigenvalues
        )[::-1]

        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]

        # Maximum possible LDA dimensions
        max_components = min(
            len(self.classes_) - 1,
            n_features
        )

        n_components = min(
            self.n_components,
            max_components
        )

        self.components_ = eigenvectors[
            :, :n_components
        ]

        print("\nLDA eigenvalues:")

        for i, value in enumerate(
            eigenvalues[:n_components],
            start=1
        ):
            print(
                f"LD{i}: {value:.6f}"
            )

        return self

    def transform(self, X):

        return X @ self.components_

    def fit_transform(self, X, y):

        self.fit(X, y)

        return self.transform(X)
