import retro
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize


class BasicRetroEnv():
    def __init__(self, game, state=retro.State.DEFAULT, stack_size=4):
        self.env = retro.make(game=game, state=state)
        self.action_size = self.env.action_space.n
        self.state_size = [stack_size, 110, 84]
        self.onehot_action = np.array(np.identity(self.action_size).tolist())

    def prepare(self, ob):
        ob = rgb2gray(ob)
        ob = ob[8:-12, 4:-14]
        ob = ob / 255.0
        ob = resize(ob, [110, 84])
        return ob

    def reset(self):
        ob = self.env.reset()
        ob = self.prepare(ob)
        return ob

    def render(self):
        self.env.render()

    def step(self, action):
        action = self.onehot_action[action]
        next_ob, reward, done, _ = self.env.step(action)
        next_ob = self.prepare(next_ob)
        return next_ob, reward, done
