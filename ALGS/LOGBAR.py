import numpy as np
from ALGS.OSMD import OSMD

class LOGBAR(OSMD):
    def __init__(self, time_horizon, num_arms, num_action):
        super().__init__(time_horizon, num_arms, num_action)
        self.EPS = 1e-12
    def _regular_func(self, x):
        return np.sum(-np.log(x+self.EPS))
    def _regular_grad_func(self, x):
        return -1 / (x+self.EPS)
    def _regular_grad_inv_func(self, y):
        y_clipped = np.clip(y, None, -self.EPS)
        return -1 / y_clipped
    def _get_learning_rate(self, t):
        return 0.5 * np.sqrt(np.log(1.0 + t) / t)