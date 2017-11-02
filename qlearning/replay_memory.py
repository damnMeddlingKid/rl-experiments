from __future__ import unicode_literals
import numpy as np


class ReplayMemory(object):

    @staticmethod
    def _init_memory(size, observation_shape, dtype):
        return np.zeros((size,) + observation_shape, dtype=dtype)

    def _batch_to_dict(self, indices):
        return {
            'current_state': self._current_state_memory[indices],
            'future_state': self._future_state_memory[indices],
            'action': self._action_memory[indices],
            'reward': self._reward_memory[indices],
            'terminal': self._terminal_memory[indices],
        }

    def __len__(self):
        return self._size

    def __init__(self, size, observation_shape):
        super(ReplayMemory, self).__init__()
        self._current_state_memory = self._init_memory(size, observation_shape, np.uint8)
        self._future_state_memory = self._init_memory(size, observation_shape, np.uint8)
        self._action_memory = self._init_memory(size, (), np.uint8)
        self._reward_memory = self._init_memory(size, (), np.float32)
        self._terminal_memory = self._init_memory(size, (), np.uint8)
        self._max_size = size
        self._head = 0
        self._observation_shape = observation_shape
        self._size = 0

    def add(self, current_state, action, reward, terminal, future_state):
        assert current_state.shape == self._observation_shape and future_state.shape == self._observation_shape, "Observation has incorrect shape."
        self._current_state_memory[self._head] = np.copy(current_state)
        self._future_state_memory[self._head] = np.copy(future_state)
        self._action_memory[self._head] = action
        self._reward_memory[self._head] = reward
        self._terminal_memory[self._head] = terminal
        self._head = (self._head + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def sample(self, batch_size):
        assert batch_size <= self._size, "Sampling more elements than exist in memory."
        indices = np.random.randint(0, self._size, size=batch_size)
        return self._batch_to_dict(indices)

    def get(self, idx):
        return self._batch_to_dict(idx)
