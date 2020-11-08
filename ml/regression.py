import numpy as np


class LinearRegression:
    DEFAULT_ALPHA = 0.0005
    DEFAULT_EPOCHS_COUNT = 1000

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.y = y
        self.x = self._construct_x(x)
        self.theta = np.ones(2).reshape(2, 1)

    def precompute(self, alpha=DEFAULT_ALPHA, epochs_count=DEFAULT_EPOCHS_COUNT) -> np.ndarray:
        """
        Prepare model.

        Use Gradient Dissent to prepare model for future use.

        :param alpha: Learning rate for the model.
        :param epochs_count: Iterations count for Gradient Dissent.
        :return: Theta data (np.ndarray[b, multiplier]).
        """
        for _ in range(int(np.sqrt(epochs_count))):
            self._gd(alpha, epochs_count)
        return self.theta

    def _gd(self, alpha: float, epochs_count: int) -> np.ndarray:
        hx = np.matmul(self.x, self.theta)
        cost_1 = self.compute_cost(self.theta)
        for _ in range(epochs_count):
            temp = (alpha / self.y.size) * np.matmul(self.x.T, (hx - self.y))
            cost_2 = self.compute_cost(self.theta - temp)

            if cost_1 > cost_2:
                cost_1 = cost_2
                self.theta -= temp

        return self.theta

    def compute_cost(self, theta=None) -> float:
        """
        Compute sum of squared error.

        :param theta: Theta data (np.ndarray[b, multiplier]).
        :return: Calculated sum of squared error.
        """
        if theta is None:
            theta = self.theta
        hx = np.matmul(self.x, theta)
        return np.sum(np.power(np.subtract(hx, self.y), 2)) / (2 * self.y.size)

    def predict(self, one_x: float) -> float:
        b, m = self.theta[0][0], self.theta[1][0]
        return one_x * m + b

    @staticmethod
    def _construct_x(x: np.ndarray) -> np.ndarray:
        return np.hstack((np.ones(x.size).reshape(x.size, 1), x))

    def r2_score(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the r2 score of the model.

        :param x: Input dataset.
        :param y: Target dataset.
        :return Calculated r2 score of the model.
        """
        predicted = self._construct_x(x) @ self.theta
        return 1 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2)
