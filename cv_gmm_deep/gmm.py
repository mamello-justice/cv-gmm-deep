import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, K: int, I: int):
        """Initialize Gaussian mixture model

        Args:
            K (int): Number of gaussian models
            I (int): Number of features
        """
        self.K = K
        self.I = I

        self.compiled = False

    def compile(self):
        self.lambda_m = np.ones(self.K) / self.K

        self.mu_m = np.random.random((self.K, self.I))

        sigma_m = np.random.random((self.K, self.I, self.I))
        for k in range(self.K):
            sigma_m[k] *= sigma_m[k].T
            sigma_m[k] += self.I * np.eye(self.I)
        self.sigma_m = sigma_m

        self.compiled = True

    def _require_compile(self):
        if not self.compiled:
            raise Exception(
                "Cannot fit before compiling model. Run model.compile()")

    def __call__(self, x):
        self._require_compile()

        values = np.empty((0, *x.shape[:-1]))

        for k in range(self.K):
            y = [self.lambda_m[k] *
                 multivariate_normal.pdf(x, self.mu_m[k], self.sigma_m[k])]
            values = np.append(values, y, axis=0)

        return np.sum(values, axis=0), values

    def _expectation_step(self, x):
        pass

    def _maximisation_step(self, x):
        pass

    def _train_step(self):
        pass

    def fit(self, x, y, batch_size=1, epochs=1, validation_data=None):
        self._require_compile()
        pass

    def save(self, path):
        pass

    @staticmethod
    def load_model(path):
        pass
