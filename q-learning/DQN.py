import gym
import numpy as np
import tensorflow as tf

import random
from collections import deque, namedtuple
from scipy.misc import imresize
env = gym.make('Breakout-v4')

STATE_FRAMES = 4
STATE_FRAME_WIDTH = 84
STATE_FRAME_HEIGHT = 84
ACTION_SPACE = env.action_space.n
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 1000000
REPLAY_MEMORY = deque(maxlen=REPLAY_MEMORY_SIZE)
REPLAY_START_SIZE = 50000
BATCH_SIZE = 32
EPSILON = 1


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


def filter_weights(width, height, input_channels, output_channels):
    stddev = 1 / float(width * height * input_channels)
    return tf.Variable(tf.random_normal([width, height, input_channels, output_channels], stddev=stddev))


def fully_connected(input_width, output_width):
    stddev = 2 / float(input_width + output_width)
    return tf.Variable(tf.random_normal([input_width, output_width], stddev=stddev))

state_0 = grey_scale(env.reset())
state_0 = np.stack([state_0] * STATE_FRAMES, axis=2)
state_vector = imresize(state_0, (STATE_FRAME_WIDTH, STATE_FRAME_HEIGHT))


def Q(state_vector):
    conv1 = tf.nn.conv2d(state_vector, filter_weights(8, 8, 4, 32), [1, 4, 4, 1], padding='VALID')
    biases1 = tf.Variable(tf.zeros([32]))
    relu1 = tf.nn.relu(conv1 + biases1)

    conv2 = tf.nn.conv2d(relu1, filter_weights(4, 4, 32, 64), [1, 2, 2, 1], padding='VALID')
    biases2 = tf.Variable(tf.zeros([64]))
    relu2 = tf.nn.relu(conv2 + biases2)

    conv3 = tf.nn.conv2d(relu2, filter_weights(3, 3, 64, 64), [1, 1, 1, 1], padding='VALID')
    biases3 = tf.Variable(tf.zeros([64]))
    relu3 = tf.nn.relu(conv3 + biases3)

    total_pixels = relu3.get_shape()
    total_pixels = total_pixels[1].value * total_pixels[2].value * total_pixels[3].value

    flattened = tf.reshape(relu3, [-1, total_pixels])
    fc_weights = fully_connected(total_pixels, 512)
    fc_biases = tf.Variable(tf.zeros([512]))
    fc_output = tf.nn.relu(tf.matmul(flattened, fc_weights) + fc_biases)

    fc2_weights = fully_connected(512, ACTION_SPACE)
    fc2_biases = tf.Variable(tf.zeros([ACTION_SPACE]))
    fc2_output = tf.matmul(fc_output, fc2_weights) + fc2_biases

    return fc2_output


def huber_loss(x):
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

state_input = tf.placeholder(tf.float32, [None, STATE_FRAME_WIDTH, STATE_FRAME_HEIGHT, STATE_FRAMES])
q_function = Q(state_input)
feed = {state_input: [state_vector]}

q_target = tf.placeholder(shape=[None, 1, 4], dtype=tf.float32)
loss = tf.reduce_mean(huber_loss(q_function - q_target))

trainer = tf.train.RMSPropOptimizer(
    learning_rate=0.00025,
    momentum=0.95,

)
model_update = trainer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
env.step(1)

Observation = namedtuple('Observation', ['state', 'action', 'reward', 'next_state', 'terminal'])

frame_count = 0
finished_episode = False

for frame in xrange(1000000):
    if finished_episode:
        env.reset()

    for play in xrange(4):
        current_state = {state_input: [state_vector]}

        #epsilon greedy
        if np.random.rand(1)[0] < EPSILON or len(REPLAY_MEMORY) < REPLAY_START_SIZE:
            current_action = env.action_space.sample()
        else:
            current_q = q_function.eval(current_state, session=sess)[0]
            current_action = np.argmax(current_q)

        state_frame, reward, finished_episode, _ = env.step(current_action)
        next_state = update_state_vector(state_vector, state_frame)

        observation = Observation(state=state_vector, action=current_action, reward=reward, next_state=next_state, terminal=finished_episode)
        REPLAY_MEMORY.append(observation)

        state_vector = next_state
        EPSILON = (-9e-7 * frame_count + 1) if frame_count < 1e6 else 0.1

        if len(REPLAY_MEMORY) > REPLAY_START_SIZE:
            frame_count += 1

    # Train a batch
    if len(REPLAY_MEMORY) >= REPLAY_START_SIZE:
        states = []
        targets = []
        sample = random.sample(REPLAY_MEMORY, BATCH_SIZE)

        for observation in sample:
            q_values = q_function.eval({state_input: [observation.state, observation.next_state]}, session=sess)
            future_reward = max(q_values[1])
            q_current = q_values[0]

            target_update = observation.reward

            if not observation.terminal:
                target_update += DISCOUNT * future_reward

            q_current[observation.action] = target_update

            states.append(observation.state)
            targets.append([q_current])

        sess.run([model_update], feed_dict={state_input: states, q_target: targets})

    if frame % 1000 == 0:
        print "Frame: ", frame, "REPLAY MEMORY SIZE: ", len(REPLAY_MEMORY), "Epsilon: ", EPSILON


saver = tf.train.Saver()
save_path = saver.save(sess, "/tmp/model.ckpt")