import argparse
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from ofa.agent import DQNAgent
from ofa.agent import Monitor
from ofa.env import BasicRetroEnv


BATCH_SIZE = 32
LEARNING_RATE = 1e-5
GAMMA = 0.99

INITIAL_EPSILON = 0.99
FINAL_EPSILON = 0.01
EXPLORE_STEPS = 500000
EPSILON_DECAY = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STEPS

OB_STEPS = 10000
MAX_MEMORY = 50000

MAX_TEST_EPISODE = 100


def test_agent():
    env = BasicRetroEnv(game='SpaceInvaders-Atari2600')
    agent = DQNAgent(env.action_size, env.state_size, LEARNING_RATE, MAX_MEMORY, GAMMA, BATCH_SIZE)
    monitor = Monitor()

    frame_index = agent.memory.register(env.reset())

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
        action = agent.act(frame_index)

        all_q = agent.get_all_qs(frame_index)[0]
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

        monitor.add(reward)

        agent.memory.append((frame_index, action, reward, next_frame_index, done))

        frame_index = next_frame_index

        step += 1

        if done:
            episode += 1

            monitor.show(step, episode, agent.selector.epsilon, args.test)

            frame_index = agent.memory.register(env.reset())


def main(args):
    env = BasicRetroEnv(game='SpaceInvaders-Atari2600')
    agent = DQNAgent(env.action_size, env.state_size, LEARNING_RATE, MAX_MEMORY, GAMMA, BATCH_SIZE)

    frame_index = agent.memory.register(env.reset())

    episode = 0
    step = 0
    while True:
        if step > OB_STEPS:
            action = agent.act(frame_index)
        else:
            action = agent.observe()

        next_frame, reward, done = env.step(action)

        next_frame_index = agent.memory.register(next_frame)
        agent.monitor.add_reward(reward)
        agent.memory.append((frame_index, action, reward, next_frame_index, done))

        frame_index = next_frame_index

        step += 1

        if step > OB_STEPS:
            agent.train()

        if done:
            episode += 1

            agent.update()
            agent.monitor.show_stats(step, episode, agent.selector.epsilon)

            frame_index = agent.memory.register(env.reset())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False,
                        help='test the performance of current net')
    parser.add_argument('--figure', action='store_true', default=False,
                        help='show q value figure while testing')
    args = parser.parse_args()

    main(args)
