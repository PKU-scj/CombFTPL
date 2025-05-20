import numpy as np
from tqdm.auto import tqdm
from ALGS.BANDIT import BANDIT

class TompsonSampling(BANDIT):
    def __init__(self, time_horizon, num_arms, num_action):
        super().__init__(time_horizon, num_arms, num_action)
        self.alpha = np.ones(num_arms)+1e-12
        self.beta = np.ones(num_arms)+1e-12
    
    def _choose_action(self, t=None):
        theta = np.random.beta(self.alpha, self.beta)
        Action = np.argsort(theta)[:self.num_action]
        return Action
    
    def _update(self, Action, losses):
        probs = np.array(losses)
        Y = np.random.binomial(1, probs)
        self.alpha[Action] += Y
        self.beta[Action] += (1-Y)
        return
 