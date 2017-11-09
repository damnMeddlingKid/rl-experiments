from __future__ import unicode_literals
import gym
from history import History


class Game(object):

    @property
    def current_state(self):
        return self._current_state

    def __init__(self, game_name, observation_shape, action_repeat, life_ends=True):
        self._env = gym.make(game_name)
        self._observation_shape = observation_shape
        self._action_repeat = action_repeat
        self._life_ends = life_ends
        self._current_state = self._reset_state()
        self._no_op_count = 0

    def _reset_state(self):
        state = History.init_observation(self._observation_shape)
        state = History.new_observation(state, self._env.reset())
        return state

    def reset_game(self):
        self._no_op_count = 0
        self._current_state = self._reset_state()

    def play_random(self):
        action = self._env.action_space.sample()
        state_data = self.play(action)
        return state_data

    def play(self, action):
        total_reward = 0
        terminal = False
        raw_state, reward, game_over, info = None, 0, False, None

        for repeat in xrange(self._action_repeat):
            raw_state, reward, game_over, info = self._env.step(action)
            total_reward += reward
            if game_over:
                terminal = True

        before_state = self._current_state
        after_state = History.new_observation(self._current_state, raw_state)

        if not game_over:
            self._current_state = after_state
        else:
            self.reset_game()

        return {
            "current_state": before_state,
            "reward": total_reward,
            "terminal": 1.0 if terminal else 0.0,
            "future_state": after_state,
            "action": action,
        }