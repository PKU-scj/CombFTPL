import numpy as np
from ALGS.OSMD import OSMD

class HYBRID(OSMD):
    def __init__(self, time_horizon, num_arms, num_action):
        super().__init__(time_horizon, num_arms, num_action)
        self.EPS = 1e-12
        self.gamma = 1
    def _regular_func(self, x):
        return -np.sum(np.sqrt(x)) + self.gamma*np.sum((1-x) * np.log(1-x+self.EPS))
    def _regular_grad_func(self, x):
        return -0.5 / np.sqrt(x+self.EPS) + self.gamma*(-np.log(1-x+self.EPS) - 1)
    def _regular_grad_inv_func(self, y, max_iter=100):
        def f(x):
            return -0.5 / np.sqrt(x+self.EPS) + self.gamma*(-np.log(1-x+self.EPS) - 1) - y
        def df(x):
            return 0.25 / (x**1.5+self.EPS) + self.gamma* 1 / (1-x+self.EPS)
        x0 = np.ones(self.num_arms) * self.num_action / self.num_arms
        tol = 1e-5
        iteration = 0
        while True:
            iteration += 1
            delta_x = f(x0) / df(x0)
            x0 = np.clip(x0-delta_x, x0/2, (x0+1)/2)
            if np.max(np.abs(delta_x))<tol or iteration>=max_iter:
                break
        return x0
    def _get_learning_rate(self, t):
        return 0.5 / np.sqrt(t)