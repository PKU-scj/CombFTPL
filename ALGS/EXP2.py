import numpy as np
from ALGS.OSMD import OSMD

class EXP2(OSMD):
    def __init__(self, time_horizon, num_arms, num_action):
        super().__init__(time_horizon, num_arms, num_action)
        self.EPS = 1e-12
    def _regular_func(self, x):
        return np.sum(x * np.log(x + self.EPS))
    def _regular_grad_func(self, x):
        return np.log(x+self.EPS) + 1
    def _regular_grad_inv_func(self, y):
        return np.exp(y-1)
    def _get_learning_rate(self, t):
        return 4 / np.sqrt(t)