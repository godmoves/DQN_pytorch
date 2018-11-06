import argparse
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from agent import DQNAgent
from agent import Env
from agent import Monitor

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


def main(args):
    env = Env(game='SpaceInvaders-Atari2600')
    agent = DQNAgent(env.action_size, env.state_size, LEARNING_RATE, MAX_MEMORY, GAMMA, BATCH_SIZE)
    monitor = Monitor()

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
        if step > OB_STEPS:
            action = agent.get_action(frame_index)
        else:
            action = np.random.randint(env.action_size)

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

        monitor.add(reward)

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

            monitor.show(step, episode, agent.selector.epsilon, args.test)

            frame_index = agent.memory.register(env.reset())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False,
                        help='test the performance of current net')
    parser.add_argument('--figure', action='store_true', default=False,
                        help='show q value figure while testing')
    args = parser.parse_args()

    main(args)
