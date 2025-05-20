import numpy as np
class Env:
    def __init__(self, time_horizon = 10**7, num_arms = 10,
                 num_action = 5, Delta = 0.1, way="stochastic"):
        self.time_horizon = time_horizon
        self.num_arms = num_arms
        self.num_action = num_action
        self.Delta = Delta
        self.way = way
        self.now = 0
        self.arms = self._setup(random_seed=None)

    def reset(self, random_seed = None):
        self.now = 0
        self.arms = self._setup(random_seed=random_seed)

    def _setup(self, random_seed = None):
        if self.way == "stochastic":
            self.rng = np.random.default_rng(seed=None)
            self.arm_means = np.zeros(self.num_arms)
            self.arm_means[:self.num_action] = 0.5-self.Delta
            self.arm_means[self.num_action:] = 0.5+self.Delta
            arm_losses = np.zeros((self.time_horizon,self.num_arms))
            arm_losses[:,:self.num_action] = self._get_loss(0.5 - self.Delta, size = (self.time_horizon, self.num_action))
            arm_losses[:,self.num_action:] = self._get_loss(0.5 + self.Delta, size = (self.time_horizon, self.num_arms-self.num_action))
        if self.way == "adversarial":
            self.rng = np.random.default_rng(seed=random_seed)
            arm_losses = np.zeros((self.time_horizon,self.num_arms))
            now_T = 0
            now_phase = 1
            while (now_T < self.time_horizon):
                now_phase_length = int(1.6**now_phase)
                if now_T + now_phase_length > self.time_horizon:
                    now_phase_length = self.time_horizon - now_T
                if now_phase % 2 == 1:
                    arm_losses[now_T:now_T+now_phase_length, :self.num_action] = self._get_loss(1-self.Delta, size = (now_phase_length, self.num_action))
                    arm_losses[now_T:now_T+now_phase_length, self.num_action:] = self._get_loss(1, size = (now_phase_length, self.num_action))
                else:
                    arm_losses[now_T:now_T+now_phase_length, :self.num_action] = self._get_loss(0, size = (now_phase_length, self.num_action))
                    arm_losses[now_T:now_T+now_phase_length, self.num_action:] = self._get_loss(self.Delta, size = (now_phase_length, self.num_action))
                now_phase += 1
                now_T += now_phase_length
        return arm_losses
    
    def _get_loss(self, mean, size=1):
        return self.rng.binomial(1, mean, size=size)
    
    def benchmark(self):
        if self.way == "stochastic":
            losses = np.broadcast_to(self.arm_means, (self.time_horizon, self.num_arms))
        else:
            losses = self.arms
        cum_losses = np.cumsum(losses, axis=0)
        best_losses = np.sum(np.sort(cum_losses, axis = 1)[:,:self.num_action], axis=1)
        return best_losses

    def step(self, actions):
        if len(actions) != self.num_action:
            raise ValueError("The length of actions should be equal to num_action.")
        losses = self.arms[self.now, actions].copy()
        self.now += 1
        return losses
    
