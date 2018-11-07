import numpy as np
import random

from collections import deque


class BasicMemory():
    '''A lot of memory will be used if we save the whole frame due to the same
    frame is actually saved 4 times, so we choose to just save the index of
    each frame and construct the state when we need it.
    '''

    def __init__(self, max_memory, stack_size=4):
        self.experience = deque(maxlen=max_memory)
        self.memory_dict = {}

        self.max_memory = max_memory
        self.index = 0
        self.full = False
        self.stack_size = stack_size
        self.previous_frame_id = 0

    def memorize(self, data):
        if isinstance(data, tuple):
            next_frame_id = self.register(data[3])
            # data[1] = self.previous_frame_id
            # data[3] = next_frame_id
            self.append((self.previous_frame_id, data[1], data[2], next_frame_id, data[4]))
            self.previous_frame_id = next_frame_id
        else:
            self.previous_frame_id = self.register(data)

    def append(self, sarsd):
        self.experience.append(sarsd)

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
        frames = []
        for i in range(self.stack_size):
            frames.append(self.memory_dict[index_list[i]])
        state = np.stack(frames, axis=0)
        return state

    def get_index_list(self, index):
        i_list = []
        for i in range(self.stack_size):
            if index - i >= 0:
                i_list.append(index - i)
            elif self.full:
                i_list.append(self.max_memory + index - i)
            else:
                i_list.append(0)
        return i_list

    def sample(self, batch_size):
        minibatch = random.sample(self.experience, batch_size)

        states = [self.get(d[0]) for d in minibatch]
        actions = [d[1] for d in minibatch]
        rewards = [d[2] for d in minibatch]
        next_states = [self.get(d[3]) for d in minibatch]
        dones = [d[4] for d in minibatch]

        return states, actions, rewards, next_states, dones
