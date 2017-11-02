from __future__ import unicode_literals
import pytest
import numpy as np
from qlearning.replay_memory import ReplayMemory


@pytest.fixture
def replay_memory():
    return ReplayMemory(4, (1,), dtype=np.float32)


def test_adding_wrong_shape_fails(replay_memory):
    with pytest.raises(AssertionError) as e:
        replay_memory.add(np.array([[6]]))

    assert "Observation has incorrect shape." in unicode(e.value)


def test_get_index(replay_memory):
    actual = [np.random.rand(1) for i in xrange(4)]
    [replay_memory.add(observation) for observation in actual]
    expected = [np.allclose(replay_memory.get(idx), array) for idx, array in enumerate(actual)]

    assert all(expected)


def test_adding_more_values_rolls_over(replay_memory):
    actual = [np.array([i]) for i in xrange(6)]
    [replay_memory.add(observation) for observation in actual]

    expected = [
        np.array([4]),
        np.array([5]),
        np.array([2]),
        np.array([3]),
    ]

    assert all([np.allclose(replay_memory.get(idx), expected[idx]) for idx in xrange(4)])


def test_copy_stored(replay_memory):
    observation = np.array([9])

    replay_memory.add(observation)

    observation[0] = 32

    assert replay_memory.get(0) == np.array([9])


def test_sampling_too_much_fails(replay_memory):
    replay_memory.add(np.array([1]))

    with pytest.raises(AssertionError) as e:
        replay_memory.sample(2)

    assert "Sampling more elements than exist in memory." in unicode(e.value)


def test_replay_memory_size(replay_memory):
    replay_memory.add(np.array([1]))
    replay_memory.add(np.array([1]))

    assert len(replay_memory) == 2

    replay_memory.add(np.array([1]))
    replay_memory.add(np.array([1]))
    replay_memory.add(np.array([1]))

    assert len(replay_memory) == 4


@pytest.mark.parametrize('dtype', [np.float32, np.uint8, np.int32])
def test_memory_casts_correctly(dtype):
    replay_memory = ReplayMemory(4, (1,), dtype=dtype)
    replay_memory.add(np.array([2], dtype=np.float32))

    assert replay_memory.get(0).dtype == dtype


def test_memory_default_type():
    replay_memory = ReplayMemory(4, (1,))
    replay_memory.add(np.array([2], dtype=np.float32))

    assert replay_memory.get(0).dtype == np.uint8
