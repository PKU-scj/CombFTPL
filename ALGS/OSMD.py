import numpy as np
import random
from ALGS.BANDIT import BANDIT
from tqdm.auto import tqdm
class OSMD(BANDIT):
    def __init__(self, time_horizon, num_arms, num_action):
        super().__init__(time_horizon, num_arms, num_action)
        self.weights = np.ones(num_arms) * self.num_action / num_arms

    def _choose_action(self, t=None):
        x = self.weights
        order_x = np.argsort(-x)

        split_distribution = self._split_sample(x[order_x])
        weights = [w[0] for w in split_distribution]

        sample_idx = np.random.choice(len(split_distribution), p=weights)
        _, left, right = split_distribution[sample_idx]
        candidates = [i for i in range(left, right)]
        random.shuffle(candidates)
        sample = [i for i in range(left)] + candidates[:(self.num_action - left)]

        Action = [order_x[i] for i in sample]
        return Action

    def _split_sample(self, x):
        left, right = 0, self.num_arms

        to_fill = np.copy(x)
        complement_to_fill = 1 - to_fill

        total_prob = 1
        splits = []

        while left <= self.num_action and right >= self.num_action and left < right:
            pos_fill = (self.num_action - left) / (right - left)
            neg_fill = 1 - pos_fill
            now = None
            if pos_fill == 0:
                weight = complement_to_fill[left]
                now = 2
            elif neg_fill == 0:
                weight = to_fill[right - 1]
                now = 1
            else:
                weight1 = complement_to_fill[left]/neg_fill
                weight2 = to_fill[right-1]/pos_fill
                if weight1 < weight2:
                    weight = weight1
                    now = 1
                else:
                    weight = weight2
                    now = 2

            weight = max(weight,0)
            splits.append((weight, left, right))
            
            total_prob -= weight
            to_fill -= weight * pos_fill
            complement_to_fill -= weight * neg_fill

            if now == 1:
                left += 1
            else:
                right -= 1
        if total_prob > 0:
            splits.append((total_prob, self.num_action, self.num_action+1))
        return splits
    
    def _update(self, Action, losses):
        loss_estimate = np.zeros(self.num_arms)
        losses = np.array(losses)
        loss_estimate[Action] = losses / self.weights[Action] 
        x_1 = self._regular_grad_func(self.weights) - self.eta * loss_estimate
        self.weights = self._bisect_search(x_1)

    def _bisect_search(self, now_x, max_iter=100):
        iteration = 0
        TOL = 1e-3
        lower = None
        upper = None
        mid = 0
        step_size = 1
        while True:
            iteration += 1
            x_2 = np.clip(self._regular_grad_inv_func(now_x + mid), 0, 1)
            if np.sum(x_2) > self.num_action:
                upper = mid
            else:
                lower = mid
            
            if abs(np.sum(x_2) - self.num_action) < TOL or iteration>max_iter:
                break

            if lower is None:
                mid = upper - step_size
                step_size *= 2
            elif upper is None:
                mid = lower + step_size
                step_size *= 2
            else:
                mid = (lower + upper) / 2

        return x_2
    
    def _target_func(self, x, now_x):
        return self._regular_func(x) - np.sum(x*self._regular_grad_func(now_x))

    def _target_grad_func(self, x, now_x):
        return self._regular_grad_func(x) - self._regular_grad_func(now_x)

    def _regular_func(self, x):
        raise NotImplementedError("regular_func should be implemented in subclasses")

    def _regular_grad_func(self, x):
        raise NotImplementedError("regular_grad_func should be implemented in subclasses")

    def _regular_grad_inv_func(self, x):
        raise NotImplementedError("regular_grad_inv_func should be implemented in subclasses")
