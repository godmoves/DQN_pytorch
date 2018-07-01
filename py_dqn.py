from collections import deque
import os
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BATCH_SIZE = 32
LEARNING_RATE = 1e-5
GAMMA = 0.99

INITIAL_EPSILON = 0.99
FINAL_EPSILON = 0.01
EXPLORE_STEPS = 500000
EPSILON_DECAY = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STEPS

OB_STEPS = 100000
MAX_MEMORY = 100000


class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv_1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear_1 = nn.Linear(2304, 784)
        self.linear_2 = nn.Linear(784, action_size)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = F.relu(self.linear_1(x.view(x.size(0), -1)))
        x = self.linear_2(x)
        return x


class DQNAgent():
    def __init__(self, action_size):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            print('CUDA enabled')
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.net = DQN(action_size).to(self.device)
        self.target_net = DQN(action_size).to(self.device)

        self.memory = Memory(MAX_MEMORY)
        self.action_size = action_size
        self.epsilon = INITIAL_EPSILON

        self.criterion = nn.MSELoss(size_average=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

        self.init_net()
        self.update_target()

    def init_net(self):
        if os.path.exists('params.pkl'):
            print('Parameters exists. Init agent from exists parameters...')
            # if we already have a model, then init our agent using the exist one.
            self.net.load_state_dict(torch.load('params.pkl'))
        else:
            print('Create new parameters...')
            torch.save(self.net.state_dict(), 'params.pkl')

    def save_net(self):
        torch.save(self.net.state_dict(), 'params.pkl')

    def update_target(self):
        self.target_net.load_state_dict(torch.load('params.pkl'))

    def get_action(self, frame_index):
        state = self.memory.get(frame_index).reshape(-1, 4, 80, 80)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.Tensor(state).to(self.device)
            y = self.net(state)
            action = torch.max(y, 1)[1]
            return int(action)

    def train(self):
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= EPSILON_DECAY

        minibatch = random.sample(self.memory.index_buffer, BATCH_SIZE)

        states = [self.memory.get(d[0]) for d in minibatch]
        actions = [d[1] for d in minibatch]
        rewards = [d[2] for d in minibatch]
        next_states = [self.memory.get(d[3]) for d in minibatch]
        dones = [d[4] for d in minibatch]

        dones = np.array(dones).astype(int)

        states = torch.Tensor(states).to(self.device)
        preds = self.net(states)

        actions = torch.LongTensor(actions).view(-1, 1)
        one_hot_action = torch.Tensor(BATCH_SIZE, self.action_size).zero_()
        one_hot_action.scatter_(1, actions, 1)

        preds = torch.sum(preds.mul(one_hot_action.to(self.device)), dim=1).view(-1, 1)

        next_states = torch.Tensor(next_states).to(self.device)
        next_preds = self.target_net(next_states)

        rewards = torch.Tensor(rewards).to(self.device)
        dones = torch.Tensor(dones).to(self.device)

        targets = rewards + (1 - dones) * GAMMA * next_preds.max(1)[0]
        targets = targets.detach()

        loss = self.criterion(preds, targets.view(-1, 1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Memory():
    def __init__(self, max_memory):
        self.sarsd_buffer = deque(maxlen=max_memory)
        self.memory_dict = {}
        self.max_memory = max_memory
        self.index = 0
        self.full = False

    def append(self, sarsd):
        self.sarsd_buffer.append(sarsd)

    def register(self, frame):
        id_now = self.index
        self.memory_dict[self.index] = frame
        self.index += 1
        if self.index >= self.max_memory:
            self.index = 0
            self.full = True
        return id_now

    def get(self, index):
        index_list = self.get_index_list(index)
        # print(index_list)
        frame0 = self.memory_dict[index_list[0]]
        frame1 = self.memory_dict[index_list[1]]
        frame2 = self.memory_dict[index_list[2]]
        frame3 = self.memory_dict[index_list[3]]
        state = np.stack((frame0, frame1, frame2, frame3), axis=0)
        return state

    def get_index_list(self, index):
        i_list = []
        for i in range(4):
            if index - i >= 0:
                i_list.append(index - i)
            elif self.full:
                i_list.append(self.max_memory + index - i)
            else:
                i_list.append(0)
        return i_list


class Env():
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        # self.state = None

    def prepare(self, ob):
        ob = ob[35:195]
        ob = ob[::2, ::2, 0]
        ob[ob == 144] = 0
        ob[ob == 109] = 0
        ob[ob != 0] = 1
        return ob.astype(np.float)

    def reset(self):
        ob = self.env.reset()
        ob = self.prepare(ob)
        return ob

    def render(self):
        self.env.render()

    def step(self, action):
        next_ob, reward, done, _ = self.env.step(action)
        next_ob = self.prepare(next_ob)
        return next_ob, reward, done


class Info():
    def __init__(self):
        self.running_reward = None
        self.reward_sum = 0

    def show(self, step, episode, epsilon):
        if self.running_reward is None:
            self.running_reward = self.reward_sum
        else:
            self.running_reward = self.running_reward * 0.99 + self.reward_sum * 0.01
        print('Step {:d} Episode {:d} Epsilon {:5.3f} Reward {:5.1f} Running mean {:7.3f}'.format(
            step, episode, epsilon, self.reward_sum, self.running_reward))
        self.reward_sum = 0


def main():
    env = Env('Pong-v0')
    agent = DQNAgent(env.action_size)
    info = Info()

    frame = env.reset()
    frame_index = agent.memory.register(frame)

    episode = 0
    step = 0
    while True:
        action = agent.get_action(frame_index)

        # env.render()
        next_frame, reward, done = env.step(action)
        next_frame_index = agent.memory.register(next_frame)

        info.reward_sum += reward

        agent.memory.append((frame_index, action, reward, next_frame_index, done))

        frame_index = next_frame_index

        step += 1

        if step > OB_STEPS:
            agent.train()

        if done:
            episode += 1
            agent.save_net()
            agent.update_target()

            info.show(step, episode, agent.epsilon)

            frame = env.reset()
            frame_index = agent.memory.register(frame)


if __name__ == '__main__':
    main()
