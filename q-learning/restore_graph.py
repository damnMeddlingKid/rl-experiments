from __future__ import unicode_literals
import tensorflow as tf
import gym
from scipy.misc import imresize
import numpy as np

STATE_FRAMES = 4
STATE_FRAME_WIDTH = 84
STATE_FRAME_HEIGHT = 84
DISCOUNT = 0.99
BATCH_SIZE = 32
EPSILON = 1

frame_count = 0
finished_episode = False
import matplotlib.pyplot as plt
plt.ion()

env = gym.make('Breakout-v4')
env.reset()


def grey_scale(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def normalized_image(input_image):
    output_image = grey_scale(input_image)
    output_image = imresize(output_image, (STATE_FRAME_WIDTH, STATE_FRAME_HEIGHT))
    return output_image


def update_state_vector(state_vector, new_frame):
    new_state_vector = np.array(state_vector, copy=True)
    new_state_vector[:, :, :-1] = new_state_vector[:, :, 1:]
    new_state_vector[:, :, -1] = normalized_image(new_frame)
    return new_state_vector

sess=tf.Session()
saver = tf.train.import_meta_graph('/Users/franklyndsouza/dev/rl-experiments/models/tmp/model.ckpt.meta')
saver.restore(sess,  '/Users/franklyndsouza/dev/rl-experiments/models/tmp/model.ckpt')
graph = tf.get_default_graph()
print [n.name for n in tf.get_default_graph().as_graph_def().node]

state_input = graph.get_tensor_by_name("Placeholder:0")
q_function = graph.get_tensor_by_name("add_4:0")


state_0 = grey_scale(env.reset())
state_0 = np.stack([state_0] * STATE_FRAMES, axis=2)
state_vector = imresize(state_0, (STATE_FRAME_WIDTH, STATE_FRAME_HEIGHT))

env.step(1)

for frame in xrange(1000000):
    if finished_episode:
        env.reset()

    current_state = {state_input: [state_vector]}
    current_q = sess.run(q_function, current_state)[0]
    print current_q
    current_action = np.argmax(current_q)

    state_frame, reward, finished_episode, _ = env.step(current_action)

    next_state = update_state_vector(state_vector, state_frame)

    state_vector = next_state
    env.render()

