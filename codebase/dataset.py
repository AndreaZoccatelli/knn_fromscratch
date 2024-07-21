import numpy as np


class ClassificationDataset:
    @staticmethod
    def generate_dataset(n_classes):
        np.random.seed(6)
        X = np.empty((0, 2))
        y = np.array([])
        for c in range(n_classes):
            class_X = np.random.multivariate_normal(
                mean=[c * 1.5, 3], cov=[[0.3, 0], [0, 0.3]], size=1000
            )
            class_y = np.array([c] * 1000)
            X = np.concatenate([X, class_X], axis=0)
            y = np.concatenate([y, class_y])

        return X, y

    @staticmethod
    def train_test_split(X, y, train_size: float):
        np.random.seed(6)
        X_indexes = range(len(X))
        train_size = round(train_size * len(X))
        train_indexes = np.random.choice(X_indexes, size=train_size, replace=False)
        test_indexes = np.setdiff1d(X_indexes, train_indexes)
        X_train = X[train_indexes, :]
        X_test = X[test_indexes, :]
        y_train = y[train_indexes]
        y_test = y[test_indexes]

        return X_train, X_test, y_train, y_test

    def __init__(self, n_classes: int, train_size: float):
        self.X, self.y = self.generate_dataset(n_classes)
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(
            self.X, self.y, train_size
        )
