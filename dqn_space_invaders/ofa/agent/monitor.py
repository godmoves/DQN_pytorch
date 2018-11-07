import time
from tensorboardX import SummaryWriter


class Monitor():
    def __init__(self, log_path='log'):
        self.frame_count = 0
        self.start_time = time.time()

        self.episode_reward = 0
        self.mean_reward = None

        self.writer = SummaryWriter(log_path)

    def add_reward(self, reward):
        self.frame_count += 1
        self.episode_reward += reward

    def show_stats(self, step, episode, epsilon, test_mode=False):
        if self.mean_reward is None:
            self.mean_reward = self.episode_reward
        else:
            self.mean_reward = self.mean_reward * 0.99 + self.episode_reward * 0.01

        speed = self.frame_count / (time.time() - self.start_time)
        self.start_time = time.time()
        self.frame_count = 0

        # record the training info.
        self.writer.add_scalars('rewards', {'current_reward': self.episode_reward,
                                            'mean_reward': self.mean_reward}, step)
        self.writer.add_scalar('speed', speed, step)

        print('Step {:d} Episode {:d} Epsilon {:5.4f} Reward {:5.1f} Mean_reward {:7.3f} Speed {:.3f} f/s'.format(
            step, episode, epsilon, self.episode_reward, self.mean_reward, speed))

        # zero sum for next episode
        self.episode_reward = 0
