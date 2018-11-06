import os
import random
import numpy as np

import torch
import torch.nn as nn

from agent.memory import Memory
from agent.network import CNN
from agent.action import EpsilonGreedyActionSelector


class DQNAgent():
    def __init__(self, action_size, state_size, learning_rate, max_memory, gamma, batch_size):
        self.device = self.set_device()

        self.net = CNN(action_size).to(self.device)
        self.target_net = CNN(action_size).to(self.device)
        self.memory = Memory(max_memory)

        self.action_size = action_size
        self.state_size = state_size

        self.selector = EpsilonGreedyActionSelector(self.action_size)
        self.gamma = gamma
        self.batch_size = batch_size

        self.criterion = nn.MSELoss(size_average=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        self.init_net()
        self.update_target()

    def init_net(self):
        if os.path.exists('params.pkl'):
            print('Parameters exists.\nInit agent from exists parameters...')
            # initialize our agent using the exist model if we have one.
            self.net.load_state_dict(torch.load('params.pkl'))
        else:
            print('Create new parameters...')
            torch.save(self.net.state_dict(), 'params.pkl')

    def save_net(self):
        torch.save(self.net.state_dict(), 'params.pkl')

    def update_target(self):
        self.target_net.load_state_dict(torch.load('params.pkl'))

    def set_device(self):
        if torch.cuda.is_available():
            print('CUDA backend enabled.')
            device = torch.device('cuda')
        else:
            print('CPU backend enabled.')
            device = torch.device('cpu')
        return device

    def get_action(self, frame_index):
        if np.random.uniform() > self.selector.epsilon:
            state = self.memory.get(frame_index).reshape(-1, *self.state_size)
            state = torch.Tensor(state).to(self.device)
            score = self.net(state)
            action = self.selector.action(score)
            return action
        else:
            return np.random.randint(self.action_size)

    def get_all_actions(self, frame_index):
        state = self.memory.get(frame_index).reshape(-1, *self.state_size)
        state = torch.Tensor(state).to(self.device)
        y = self.net(state)
        return y.detach().cpu().numpy()

    def train(self):
        minibatch = random.sample(self.memory.sarsd_buffer, self.batch_size)

        states = [self.memory.get(d[0]) for d in minibatch]
        actions = [d[1] for d in minibatch]
        rewards = [d[2] for d in minibatch]
        next_states = [self.memory.get(d[3]) for d in minibatch]
        dones = [d[4] for d in minibatch]

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
