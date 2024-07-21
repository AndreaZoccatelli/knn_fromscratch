import numpy as np


class KNNClassifier:
    def __init__(self, k: int) -> None:
        self.k = k

    def fit(self, X_train: np.array, y_train: np.array):
        self.X_train = X_train
        self.y_train = y_train

    def _get_majority_class(self, arr: np.array):
        unique_values, unique_values_counts = np.unique(arr, return_counts=True)
        return unique_values[np.argmax(unique_values_counts)]

    def predict(self, X: np.array):
        # Compute the distance matrix using broadcasting
        dist_matrix = np.linalg.norm(X[:, np.newaxis] - self.X_train, axis=2)

        # Get the indices of the k nearest neighbors
        nearest_k = np.argsort(dist_matrix, axis=1)[:, : self.k]

        # Get the classes of the nearest neighbors
        nearest_k_classes = self.y_train[nearest_k]

        # Determine the majority class for each row
        classes = np.apply_along_axis(
            self._get_majority_class, axis=1, arr=nearest_k_classes
        )
        return classes
