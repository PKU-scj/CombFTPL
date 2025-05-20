import numpy as np
from tqdm.auto import tqdm
from ALGS.BANDIT import BANDIT

class FTPL(BANDIT):
    def __init__(self, time_horizon, num_arms, num_action):
        super().__init__(time_horizon, num_arms, num_action)
        self.Loss_Vector = np.zeros(num_arms)
    
    def _choose_action(self, t=None):
        r = self._F2_sample(self.num_arms)
        pertubed_loss = self.Loss_Vector - r / self.eta
        Action = np.argsort(pertubed_loss)[:self.num_action]
        return Action
    
    def _update(self, Action, losses):
        obs = np.zeros(self.num_arms)
        obs[Action] = losses
        K = np.zeros(self.num_arms)
        for act in Action:
            _Action = []
            while act not in _Action:
                K[act]+=1
                _Action = self._choose_action()
        self.Loss_Vector += obs * K
        return
    
    def _get_learning_rate(self, t):
        return 1 / np.sqrt(t)

    @staticmethod
    def _F2_sample(N):
        u = np.random.uniform(0, 1, N)
        samples = 1/ np.sqrt(-np.log(u))
        return samples
 