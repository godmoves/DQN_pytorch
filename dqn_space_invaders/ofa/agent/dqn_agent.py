import numpy as np

import torch
import torch.nn as nn

from ofa.agent.monitor import Monitor
from ofa.agent.memory import BasicMemory
from ofa.agent.network import CNN
from ofa.agent.action import EpsilonGreedyActionSelector
from ofa.agent.utils import *


class DQNAgent():
    def __init__(self, action_size, state_size, learning_rate, max_memory, gamma, batch_size):
        self.device = set_device()

        self.net = CNN(action_size).to(self.device)
        self.target_net = CNN(action_size).to(self.device)
        self.memory = BasicMemory(max_memory)
        self.monitor = Monitor()

        self.action_size = action_size
        self.state_size = state_size
        self.batch_size = batch_size

        self.selector = EpsilonGreedyActionSelector(self.action_size)
        self.gamma = gamma

        self.criterion = nn.MSELoss(size_average=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        init_net(self.net, self.target_net)

    def act(self, frame_index=None):
        if frame_index is None:
            frame_index = self.memory.previous_frame_id

        state = self.memory.get(frame_index).reshape(-1, *self.state_size)
        state = torch.Tensor(state).to(self.device)
        qs = self.net(state)
        action = self.selector.action(qs)
        return action

    def observe(self):
        return np.random.randint(self.action_size)

    def update(self, step, episode):
        save_net(self.net)
        update_target(self.target_net)
        self.monitor.show_stats(step, episode, self.selector.epsilon)

    def memorize(self, data):
        self.memory.memorize(data)
        if isinstance(data, tuple):
            self.monitor.add_reward(data[2])

    def get_all_qs(self, frame_index):
        state = self.memory.get(frame_index).reshape(-1, *self.state_size)
        state = torch.Tensor(state).to(self.device)
        qs = self.net(state)
        return qs.detach().cpu().numpy()

    def train(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        dones = np.array(dones).astype(int)

        states = torch.Tensor(states).to(self.device)
        preds = self.net(states)

        actions = torch.LongTensor(actions).view(-1, 1)
        one_hot_action = torch.Tensor(self.batch_size, self.action_size).zero_()
        one_hot_action.scatter_(1, actions, 1)

        preds = torch.sum(preds.mul(one_hot_action.to(self.device)), dim=1).view(-1, 1)

        next_states = torch.Tensor(next_states).to(self.device)
        # use the prediction of the target net rather than current net to
        # keep the training process stable.
        next_preds = self.target_net(next_states)

        rewards = torch.Tensor(rewards).to(self.device)
        dones = torch.Tensor(dones).to(self.device)

        targets = rewards + (1 - dones) * self.gamma * next_preds.max(1)[0]
        targets = targets.detach()

        loss = self.criterion(preds, targets.view(-1, 1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
