from __future__ import unicode_literals
import numpy as np


class History():

    @staticmethod
    def normalized_image(observation):
        temp = np.zeros((observation.shape[0], observation.shape[1]), dtype=np.float32)  # Blank array for new image

        # Luminosity grayscale
        temp[:, :] += (.2126 * observation[:, :, 0]).astype(np.float32)
        temp[:, :] += (.7156 * observation[:, :, 1]).astype(np.float32)
        temp[:, :] += (.0722 * observation[:, :, 2]).astype(np.float32)

        # Downsample
        temp = temp[::2, ::2]

        # Crop

        return temp[17:-8, :]

    @classmethod
    def new_observation(cls, state, raw_frame):
        new_state = np.copy(state)
        new_state[:, :, :-1] = new_state[:, :, 1:]
        new_state[:, :, -1] = cls.normalized_image(raw_frame)
        return new_state

    @classmethod
    def init_observation(cls, shape):
        return np.zeros(shape)
