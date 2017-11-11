from __future__ import unicode_literals
import tensorflow as tf
import gym
from scipy.misc import imresize
import numpy as np
from game import Game

STATE_FRAMES = 4
STATE_FRAME_WIDTH = 80
STATE_FRAME_HEIGHT = 80
DISCOUNT = 0.99
BATCH_SIZE = 32
EPSILON = 1

frame_count = 0
finished_episode = False

# env = gym.make('BreakoutNoFrameskip-v0')
# env.reset()


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

checkpoint = 4900000

sess=tf.Session()
graph = tf.get_default_graph()
saver = tf.train.import_meta_graph('/Users/franklyndsouza/dev/rl-experiments/models/model_done.ckpt.meta'.format(checkpoint))
saver.restore(sess,  '/Users/franklyndsouza/dev/rl-experiments/models/model_done.ckpt'.format(checkpoint))


state_input = graph.get_tensor_by_name("dqn_agent/input_state:0")
q_function = graph.get_tensor_by_name("dqn_agent/q_value/BiasAdd:0")

# start = normalized_image(env.reset())
# state_0 = np.zeros(start.shape)
# state_1 = np.zeros(start.shape)
# state_2 = np.zeros(start.shape)
# state_3 = start
#
# state_vector = np.stack([state_0, state_1, state_2, state_3], axis=2)
# # import matplotlib.pyplot as plt
# # plt.ion()
# state_frame = None
# frames = []
#
# for frame in xrange(1000):
#     if finished_episode:
#         env.reset()
#
#     current_state = {state_input: [state_vector]}
#     current_q = sess.run(q_function, current_state)[0]
#
#     current_action = np.argmax(current_q)
#     # if np.random.rand(1)[0] < 0.05:
#     #     current_action = env.action_space.sample() - 1
#     state_frame, reward, finished_episode, info = env.step(current_action+1)
#     env.render()
#     frames.append(env.render(mode = 'rgb_array'))
#
#     state_frame, reward, finished_episode, info = env.step(current_action+1)
#     env.render()
#     frames.append(env.render(mode = 'rgb_array'))
#
#     state_frame, reward, finished_episode, info = env.step(current_action+1)
#     env.render()
#     frames.append(env.render(mode = 'rgb_array'))
#
#     state_frame, reward, finished_episode, info = env.step(current_action+1)
#     env.render()
#     frames.append(env.render(mode = 'rgb_array'))
#
#     next_state = update_state_vector(state_vector, state_frame)
#
#     state_vector = next_state
#
#     #plt.imshow(normalized_image(state_frame))
#     #plt.pause(0.05)
observation_shape = (80, 80, 4)
game = Game('BreakoutNoFrameskip-v0', observation_shape, 4)


def evaluate(session):
    total_rewards = 0
    game.reset_game()

    for i in xrange(10):
        terminal = 0
        while terminal == 0:
            # if np.random.rand(1)[0] < 0.05:
            #     state = game.play_random()
            # else:
            q_values = session.run(q_function, {state_input: [game.current_state]})[0]
            action = np.argmax(q_values)
            state = game.play(action)
            total_rewards += state['reward']
            terminal = state['terminal']
        game.reset_game()

    game.reset_game()
    print "1000 Frame reward: ", (total_rewards/10)

evaluate(sess)