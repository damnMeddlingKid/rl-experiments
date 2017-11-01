from __future__ import unicode_literals
import pytest
import numpy as np
from qlearning.replay_memory import ReplayMemory


@pytest.fixture
def replay_memory():
    return ReplayMemory(4, (2,))


def test_adding_wrong_shape_fails(replay_memory):
    with pytest.raises(Exception) as e:
        replay_memory.add(np.array([1]))
