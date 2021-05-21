from __future__ import unicode_literals
import tensorflow as tf
import gym
from scipy.misc import imresize
import numpy as np
from game import Game
from agent import DQNAgent

STATE_FRAMES = 4
STATE_FRAME_WIDTH = 80
STATE_FRAME_HEIGHT = 80
DISCOUNT = 0.99
BATCH_SIZE = 32
EPSILON = 1

frame_count = 0
finished_episode = False


def normalized_image(observation):
    temp = np.zeros((observation.shape[0], observation.shape[1]), dtype=np.uint8)  # Blank array for new image

    # Luminosity grayscale
    temp[:, :] += (.2126 * observation[:, :, 0]).astype(np.uint8)
    temp[:, :] += (.7156 * observation[:, :, 1]).astype(np.uint8)
    temp[:, :] += (.0722 * observation[:, :, 2]).astype(np.uint8)

    # Downsample
    temp = temp[::2, ::2]

    # Crop
    return temp

checkpoint = 4900000

observation_shape = (105, 80, 4)
#target_agent = DQNAgent("target_agent", observation_shape, 4, 0.99)
#main_agent = DQNAgent("dqn_agent", observation_shape, 4, 0.99, target_agent)
sess=tf.Session()
graph = tf.get_default_graph()

# saver = tf.train.import_meta_graph('../models/space_invaders/model_done.ckpt.meta')
# saver.restore(sess,  '../models/space_invaders/model_done.ckpt')
# game = Game('SpaceInvaders-v0', observation_shape, 4)


saver = tf.train.import_meta_graph('../models/even_more/model.ckpt-1100000.meta')
saver.restore(sess,  '../models/even_more/model.ckpt-1100000')
game = Game('BreakoutNoFrameskip-v0', observation_shape, 4)


state_input = graph.get_tensor_by_name("dqn_agent/input_state:0")
q_function = graph.get_tensor_by_name("dqn_agent/q_value/BiasAdd:0")


def evaluate(session):
    total_rewards = 0
    game.reset_game()

    for i in xrange(10):
        terminal = 0.0
        while terminal != 1.0:
            # if np.random.rand(1)[0] < 0.05:
            #     state = game.play_random()
            # else:
            current_state = {state_input: [game.current_state]}
            current_q = session.run(q_function, current_state)[0]
            action = np.argmax(current_q)
            state = game.play(action)
            total_rewards += state['reward']
            terminal = state['terminal']
        game.reset_game()

    game.reset_game()
    print "1000 Frame reward: ", (total_rewards / 10.0)

evaluate(sess)