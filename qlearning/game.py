from __future__ import unicode_literals
import gym
import numpy as np
from history import History
from PIL import Image
from gym.envs.classic_control import rendering


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
        self._viewer = rendering.SimpleImageViewer()

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
        lives = self._env.unwrapped.ale.lives()

        for repeat in xrange(self._action_repeat):
            raw_state, reward, game_over, info = self._env.step(action)
            rgb = self._env.render('rgb_array')
            upscaled = self.repeat_upsample(rgb, 4, 4)
            self._viewer.imshow(upscaled)
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

    def repeat_upsample(self, rgb_array, k=1, l=1, err=[]):
        # repeat kinda crashes if k/l are zero
        if k <= 0 or l <= 0:
            if not err:
                print "Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l)
                err.append('logged')
            return rgb_array

        # repeat the pixels k times along the y axis and l times along the x axis
        # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

        return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

    def play_frames(self, action):
        total_reward = 0
        terminal = False
        raw_state, reward, game_over, info = None, 0, False, None
        #lives = self._env.unwrapped.ale.lives()
        frames = []

        for repeat in xrange(self._action_repeat):
            raw_state, reward, game_over, info = self._env.step(action)
            frames.append(Image.fromarray(raw_state))
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
            "frames": frames
        }