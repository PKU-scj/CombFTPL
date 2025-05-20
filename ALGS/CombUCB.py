import numpy as np
from ALGS.BANDIT import BANDIT
from tqdm.auto import tqdm

class CombUCB(BANDIT):
    def __init__(self, time_horizon, num_arms, num_action):
        super().__init__(time_horizon, num_arms, num_action)
        self.warm = (self.num_arms+1) // self.num_action
        self.mu_hat = np.zeros(num_arms)
        self.count = np.zeros(num_arms)
    
    def _choose_action(self, t):
        if t<=self.warm:
            Action = [((t-1)*self.num_action+j)%self.num_arms for j in range(self.num_action)]
        else:
            lcb = self.mu_hat - np.sqrt(3 * np.log(t) / (2 * self.count))
            Action = np.argsort(lcb)[:self.num_action]
        return Action

    def _update(self, Action, losses):
        losses = np.array(losses)
        self.mu_hat[Action] = (self.mu_hat[Action]*self.count[Action] + losses) / (self.count[Action] + 1)
        self.count[Action] += 1
        return
 