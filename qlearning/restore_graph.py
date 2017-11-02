from __future__ import unicode_literals
import tensorflow as tf
import gym
from scipy.misc import imresize
import numpy as np

STATE_FRAMES = 4
STATE_FRAME_WIDTH = 80
STATE_FRAME_HEIGHT = 80
DISCOUNT = 0.99
BATCH_SIZE = 32
EPSILON = 1

frame_count = 0
finished_episode = False
import matplotlib.pyplot as plt
plt.ion()

env = gym.make('BreakoutNoFrameskip-v0')
env.reset()


def grey_scale(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def normalized_image(observation):
    temp = np.zeros((observation.shape[0], observation.shape[1]), dtype=np.uint8)  # Blank array for new image

    # Luminosity grayscale
    temp[:, :] += (.2126 * observation[:, :, 0]).astype(np.uint8)
    temp[:, :] += (.7156 * observation[:, :, 1]).astype(np.uint8)
    temp[:, :] += (.0722 * observation[:, :, 2]).astype(np.uint8)

    # Downsample
    temp = temp[::2, ::2]

    # Crop
    return temp[17:-8, :]


def update_state_vector(state_vector, new_frame):
    new_state_vector = np.array(state_vector, copy=True)
    new_state_vector[:, :, :-1] = new_state_vector[:, :, 1:]
    new_state_vector[:, :, -1] = normalized_image(new_frame)
    return new_state_vector

sess=tf.Session()
saver = tf.train.import_meta_graph('/Users/franklyndsouza/dev/rl-experiments/models/model.ckpt.meta')
saver.restore(sess,  '/Users/franklyndsouza/dev/rl-experiments/models/model.ckpt')
graph = tf.get_default_graph()
print [n.name for n in tf.get_default_graph().as_graph_def().node]

state_input = graph.get_tensor_by_name("dqn_agent/input_state:0")
q_function = graph.get_tensor_by_name("dqn_agent/q_value/BiasAdd:0")


state_0 = normalized_image(env.reset())
state_1 = normalized_image(env.step(0)[0])
state_2 = normalized_image(env.step(0)[0])
state_3 = normalized_image(env.step(0)[0])

state_vector = np.stack([state_0, state_1, state_2, state_3], axis=2)

for frame in xrange(1000000):
    if finished_episode:
        env.reset()

    current_state = {state_input: [state_vector]}
    current_q = sess.run(q_function, current_state)[0]

    current_action = np.argmax(current_q)
    state_frame, reward, finished_episode, info = env.step(current_action)

    next_state = update_state_vector(state_vector, state_frame)

    state_vector = next_state
    env.render()

