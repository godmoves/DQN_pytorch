import argparse
import os
import random
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter


BATCH_SIZE = 32
LEARNING_RATE = 1e-5
GAMMA = 0.99

INITIAL_EPSILON = 0.99
FINAL_EPSILON = 0.01
EXPLORE_STEPS = 500000
EPSILON_DECAY = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STEPS

OB_STEPS = 100000
MAX_MEMORY = 100000

MAX_TEST_EPISODE = 100


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
        # not add softmax at the last layer.
        x = self.linear_2(x)
        return x


class DQNAgent():
    def __init__(self, action_size):
        if torch.cuda.is_available():
            print('CUDA backend enabled.')
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.net = DQN(action_size).to(self.device)
        self.target_net = DQN(action_size).to(self.device)

        self.memory = Memory(MAX_MEMORY)
        self.action_size = action_size
        self.epsilon = INITIAL_EPSILON

        self.criterion = nn.MSELoss(size_average=True)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=LEARNING_RATE)

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

    def get_action(self, frame_index):
        state = self.memory.get(frame_index).reshape(-1, 4, 80, 80)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.Tensor(state).to(self.device)
            y = self.net(state)
            action = torch.max(y, 1)[1]
            return int(action)

    def get_all_actions(self, frame_index):
        state = self.memory.get(frame_index).reshape(-1, 4, 80, 80)
        state = torch.Tensor(state).to(self.device)
        y = self.net(state)
        return y.detach().cpu().numpy()

    def train(self):
        # recude the randomness of action.
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= EPSILON_DECAY

        minibatch = random.sample(self.memory.sarsd_buffer, BATCH_SIZE)

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
        # use the prediction of the target net rather than current net to
        # keep the training process stable.
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
    '''A lot of memory will be used if we save the whole frame due to the same
    frame is actually be saved 4 times, so we choose to just save the index
    of each frame and construct the state when we need it.'''

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

    def prepare(self, ob):
        # the parameters are chosen for pong-v0 only.
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
        self.reward_sum = 0
        self.running_reward = None
        self.win_episode_count = 0
        self.writer = SummaryWriter('log')

    def show(self, step, episode, epsilon, arg_test):
        if self.running_reward is None:
            self.running_reward = self.reward_sum
        else:
            self.running_reward = self.running_reward * 0.99 + self.reward_sum * 0.01

        # record the training info.
        self.writer.add_scalars('rewards', {'current_r': self.reward_sum,
                                            'running_r': self.running_reward}, step)
        if arg_test:
            our_score = 21 if self.reward_sum > 0 else int(21 + self.reward_sum)
            opp_score = int(21 - self.reward_sum) if self.reward_sum > 0 else 21

            if self.reward_sum > 0:
                # increase winning count if we win.
                self.win_episode_count += 1

            print('Step {:d}  Episode {:d}/{:d} (win/total)  Result {:d}:{:d} (opp:our)'.format(
                step, self.win_episode_count, episode, opp_score, our_score))

            if episode >= MAX_TEST_EPISODE:
                exit()
        else:
            print('Step {:d} Episode {:d} Epsilon {:5.3f} Reward {:5.1f} Running mean {:7.3f}'.format(
                step, episode, epsilon, self.reward_sum, self.running_reward))

        # zero sum for next episode
        self.reward_sum = 0


def main(args):
    env = Env('Pong-v0')
    agent = DQNAgent(env.action_size)
    info = Info()

    frame_index = agent.memory.register(env.reset())

    if args.test:
        print('Running in test mode.')
        # choose actions greedily when testing.
        agent.epsilon = 0

        if args.figure:
            fig = plt.figure()
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
            plt.ion()
            max_q_list = deque(maxlen=300)

    episode = 0
    step = 0
    while True:
        action = agent.get_action(frame_index)

        if args.test and args.figure:
            all_q = agent.get_all_actions(frame_index)[0]
            ax1.bar(np.arange(env.action_size), all_q)

            max_q = max(all_q)
            max_q_list.append(max_q)
            ax2.plot(max_q_list)

            plt.pause(0.00001)
            ax1.cla()
            ax2.cla()

            env.render()

        next_frame, reward, done = env.step(action)
        next_frame_index = agent.memory.register(next_frame)

        info.reward_sum += reward

        agent.memory.append((frame_index, action, reward, next_frame_index, done))

        frame_index = next_frame_index

        step += 1

        if step > OB_STEPS and not args.test:
            agent.train()

        if done:
            episode += 1

            if not args.test:
                agent.save_net()
                agent.update_target()

            info.show(step, episode, agent.epsilon, args.test)

            frame_index = agent.memory.register(env.reset())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False,
                        help='test the performance of current net')
    parser.add_argument('--figure', action='store_true', default=False,
                        help='show q value figure while testing')
    args = parser.parse_args()

    main(args)
