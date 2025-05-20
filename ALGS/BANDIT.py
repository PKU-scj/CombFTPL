import numpy as np
from tqdm.auto import tqdm

class BANDIT:
    def __init__(self, time_horizon, num_arms, num_action):
        self.time_horizon = time_horizon
        self.num_arms = num_arms
        self.num_action = num_action
        self.eta = None
    
    def _choose_action(self, t):
        raise NotImplementedError("This method should be overridden by subclasses")

    def _update(self, Action, losses):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def _get_learning_rate(self, t):
        return None

    def run(self, env):
        action_set = []
        loops = tqdm(range(1,self.time_horizon+1))
        for t in loops:
            self.eta = self._get_learning_rate(t)
            Action = self._choose_action(t)
            losses = env.step(Action)
            action_set.append(self._one_hot(Action))
            self._update(Action, losses)
        action_set = np.array(action_set)
        return action_set
    
    def _one_hot(self, Action):
        Action_one_hot = np.zeros(self.num_arms)
        Action_one_hot[Action] = 1
        return Action_one_hot