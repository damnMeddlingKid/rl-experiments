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
        self._observation_shape = observation_shape

    def add(self, observation):
        assert observation.shape == self._observation_shape, "Observation has incorrect shape."
        self._memory[self._head] = np.copy(observation)
        self._head = (self._head + 1) % self._size

    def sample(self, batch_size):
        assert batch_size <= self._head, "Sampling more elements than exist in memory."
        indices = np.random.randint(0, self._head, size=batch_size)
        return self._memory[indices]

    def get(self, idx):
        return self._memory[idx]