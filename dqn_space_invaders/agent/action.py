import numpy as np


class ArgmaxActionSelector():
    def __init__(self, action_size):
        self.action_size = action_size

    def action(self, score):
        return np.argmax(score)


class EpsilonGreedyActionSelector():
    def __init__(self, action_size, epsilon_start=0.99, epsilon_end=0.01, explore_steps=50000):
        self.action_size = action_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.explore_steps = explore_steps
        self.step = 0
        self.epsilon = self.cal_epsilon()

    def cal_epsilon(self):
        epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.step / self.explore_steps
        epsilon = max(self.epsilon_end, epsilon)
        return epsilon

    def action(self, score):
        self.step += 1
        self.epsilon = self.cal_epsilon()

        if np.random.uniform() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(score.detach().cpu())
