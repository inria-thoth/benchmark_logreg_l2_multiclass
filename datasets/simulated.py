import numpy as np

from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        'n_samples, n_features': [
            (200, 600),
            (1000, 10),
            (100_000, 400),
        ]
    }

    def __init__(self, n_samples=10, n_features=50, random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)

        X = rng.randn(self.n_samples, self.n_features)
        y = rng.randint(5, size=self.n_samples)

        X_test = rng.randn(self.n_samples, self.n_features)
        y_test = rng.randint(5, size=self.n_samples)

        data = dict(X=X, y=y, name=self.name, X_test=X_test, y_test=y_test)

        return data
