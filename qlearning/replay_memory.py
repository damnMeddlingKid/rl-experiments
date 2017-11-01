from __future__ import unicode_literals
import numpy as np


class ReplayMemory(object):

    def _init_memory(self, size, observation_shape, dtype):
        return np.zeros((size,) + observation_shape, dtype=dtype)

    def __init__(self, size, observation_shape, dtype=np.uint8):
        super(ReplayMemory, self).__init__()
        self._memory = self._init_memory(size, observation_shape, dtype)
        self._size = size
        self._head = 0

    def add(self, observation):
        self._memory[self._head] = np.copy(observation)
        self._head = (self._head + 1) % self._size

    def sample(self, batch_size):
        indices = np.random.randint(0, self._size, size=batch_size)
        return self._memory[indices]