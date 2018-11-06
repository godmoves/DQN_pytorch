import numpy as np

from collections import deque


class Memory():
    '''A lot of memory will be used if we save the whole frame due to the same
    frame is actually saved 4 times, so we choose to just save the index of
    each frame and construct the state when we need it.
    '''

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
