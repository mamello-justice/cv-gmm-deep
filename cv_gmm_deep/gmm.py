import numpy as np
from tqdm.auto import trange
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, K: int, D: int, I: int):
        """Initialize Gaussian mixture model

        Args:
            K (int): Number of gaussian models
            D (int): Number of dimensions/features
            I (int): Number of pixels
        """
        self.K = K
        self.D = D
        self.I = I

        self.compiled = False

    def compile(self):
        self.lambda_m = np.ones(self.K) / self.K

        self.mu_m = np.random.random((self.K, self.D))

        sigma_m = np.random.random((self.K, self.D, self.D))
        for k in range(self.K):
            sigma_m[k] *= sigma_m[k].T
            sigma_m[k] += self.D * np.eye(self.D)
        self.sigma_m = sigma_m

        self.compiled = True

    def _require_compile(self):
        if not self.compiled:
            raise Exception(
                "Cannot fit before compiling model. Run model.compile()")

    def __call__(self, x):
        """Predict values usings current weights

        Args:
            x (I, D): Input data by features/dimensions

        Returns:
            tuple: (sum, values)
                sum (I,): summed values along K
                values (K, I): predicted pixel values by K 
        """
        self._require_compile()

        values = np.empty((0, *x.shape[:-1]))

        for lambda_, mu, sigma in zip(self.lambda_m, self.mu_m, self.sigma_m):
            y = lambda_ * multivariate_normal.pdf(x, mu, sigma)
            values = np.append(values, [y], axis=0)

        return np.sum(values, axis=0), values

    def _e_step(self, x):
        """Expectation Step (Calculates responsibilities)

        Args:
            x (I, D): Input data by features/dimensions

        Returns:
            (K, I): Responsibilities
        """
        V_sum, V = self(x)
        return V / V_sum

    def _m_step(self, x, r_ki):
        """Maximization Step (Updates parameters using responsibilities)

        Args:
            x (I, D): Input data by features/dimensions
            r (K, I): Responsibilities
        """
        # Update cluster spread
        r_k = np.sum(r_ki, axis=-1)
        r = np.sum(r_k)
        self.lambda_m = r_k / r

        # Make responsibilities able to be broadcasted with input
        r_ki_ = np.expand_dims(r_ki, axis=-1)

        # Update mean
        self.mu_m = np.sum(r_ki_ * x, axis=1) / np.expand_dims(r_k, axis=-1)

        # Update variance
        d_scores = x - np.expand_dims(self.mu_m, axis=1)
        sigma_num = np.zeros_like(self.sigma_m)

        for k in range(self.K):
            for i in range(self.I):
                d_score = d_scores[k, i]
                sigma_num += r_ki_[k, i] * d_score.T @ d_score

        self.sigma_m = sigma_num / np.expand_dims(r_k, axis=(-2, -1))

    def _train_step(self, x):
        ll = self._e_step(x)
        self._m_step(x, ll)

    def train(self, x, epochs=1):
        self._require_compile()

        for ep in range(epochs):
            for i in trange(0, len(x), desc='epoch %d' % ep):
                self._train_step(x[i])

        return False

    def save(self, path):
        pass

    @staticmethod
    def load_model(path):
        pass
