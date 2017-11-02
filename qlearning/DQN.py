import gym
import numpy as np
import datetime
import tensorflow as tf

import random
from collections import deque, namedtuple
from replay_memory import ReplayMemory

env = gym.make('BreakoutNoFrameskip-v3')

NUM_REPEATS = 4
STATE_FRAMES = 4
STATE_FRAME_WIDTH = 80
STATE_FRAME_HEIGHT = 80
ACTION_SPACE = env.action_space.n
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 500000
REPLAY_MEMORY = deque(maxlen=REPLAY_MEMORY_SIZE)
REPLAY_START_SIZE = 50000
BATCH_SIZE = 32
EPSILON = 1
EPSILON_START = 1
EPSILON_END = 0.1
EPSILON_END_FRAME = 100000
NUM_EPOCHS = 100000
NUM_BATCHES = 3


def grey_scale(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# def normalized_image(input_image):
#     output_image = grey_scale(input_image)
#     output_image = imresize(output_image, (STATE_FRAME_WIDTH, STATE_FRAME_HEIGHT))
#     return output_image

# def plot_state(state_vector):
#     axarr[0,0].imshow(state_vector[:, :,0])
#     axarr[0,1].imshow(state_vector[:, :,1])
#     axarr[1,0].imshow(state_vector[:, :,2])
#     axarr[1,1].imshow(state_vector[:, :,3])
#     plt.pause(0.05)


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


def update_state_vector(state_vector, new_frame):
    new_state_vector = np.array(state_vector, copy=True, dtype=np.float32)
    new_state_vector[:, :, :-1] = new_state_vector[:, :, 1:]
    new_state_vector[:, :, -1] = normalized_image(new_frame)
    return new_state_vector


def random_arg_max(array):
    return np.random.choice(np.where(array==array.max())[0])


def Q(state_vector, name):
    conv1 = tf.layers.conv2d(
        inputs=state_vector,
        filters=32,
        kernel_size=[8, 8],
        strides=(4, 4),
        padding="valid",
        activation=tf.nn.relu,
        name="conv1_{}".format(name)
    )

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[4, 4],
        strides=(2, 2),
        padding="valid",
        activation=tf.nn.relu,
        name="conv2_{}".format(name)
    )

    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=64,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.relu,
        name="conv3_{}".format(name)
    )

    flattened = tf.contrib.layers.flatten(conv3)

    dense1 = tf.layers.dense(inputs=flattened, units=512, activation=tf.nn.relu, name="dense1_{}".format(name))
    dense2 = tf.layers.dense(inputs=dense1, units=ACTION_SPACE, name=name)

    return dense2

state_input = tf.placeholder(tf.float32, [None, STATE_FRAME_WIDTH, STATE_FRAME_HEIGHT, STATE_FRAMES], name='state_input')
target_state_input = tf.placeholder(tf.float32, [None, STATE_FRAME_WIDTH, STATE_FRAME_HEIGHT, STATE_FRAMES], name='target_state_input')

q_function = Q(state_input, 'q_function')
q_target_network = Q(target_state_input, 'q_target')

NUM_VARIABLES = len(tf.trainable_variables())/2


def update_target_network():
    variables = tf.trainable_variables()
    main_network = variables[:NUM_VARIABLES]
    target_network = variables[NUM_VARIABLES:]
    assert len(main_network) == len(target_network)
    return [target_network[i].assign(main_network[i]) for i in xrange(len(main_network))]


action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
terminal_mask = tf.placeholder(shape=[None], dtype=tf.float32)
reward_input = tf.placeholder(shape=[None], dtype=tf.float32)

action_one_hot = tf.one_hot(action_holder, env.action_space.n, 1.0, 0.0, name='action_one_hot')

q_acted = tf.reduce_sum(q_function * action_one_hot, axis=1, name='q_acted')
q_target = tf.reduce_max(q_target_network, axis=1)
q_selected_target = reward_input + (1.0 - terminal_mask) * tf.scalar_mul(DISCOUNT, q_target)

#loss = tf.losses.mean_squared_error(q_acted, q_selected_target)
loss = tf.losses.huber_loss(q_acted, q_selected_target, reduction='weighted_sum')

tf.summary.scalar('loss', loss)
tf.summary.scalar('trueQ', tf.reduce_mean(q_selected_target))
tf.summary.scalar('predictedQ', tf.reduce_mean(q_acted))

global_step = tf.Variable(0, name='global_step', trainable=False)

trainer = tf.train.RMSPropOptimizer(
    learning_rate=0.00025,
    momentum=0.95
)

model_update = trainer.minimize(loss, global_step=global_step, var_list=tf.trainable_variables()[:NUM_VARIABLES])

sess = tf.Session()
merged_summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter('/output/logs/{}'.format(str(datetime.datetime.now()).split('.')[0]), graph=sess.graph)
sess.run(tf.global_variables_initializer())
sess.run(update_target_network())

Observation = namedtuple('Observation', ['state', 'action', 'reward', 'next_state', 'terminal'])

loss_value = 0
reward = 0
frame_count = 0
finished_episode = True
state_frame = None
state_vector = None
total_episode_reward = 0
num_episodes = 0
ale_info = {'ale.lives': 5}

# plt.ion()
# f, axarr = plt.subplots(2,2)


def evaluate_model(q_function, num_games, env):
    total_rewards = 0
    print "Beginnning model evaluation"

    games_played = 0

    while games_played < num_games:
        state_0 = normalized_image(env.reset())
        state_0 = np.stack([state_0] * 4, axis=2)
        state_vector = state_0

        while True:
            current_state = {state_input: [state_vector]}
            current_q = sess.run(q_function, current_state)[0]

            current_action = np.argmax(current_q)
            state_frame, reward, finished_episode, info = env.step(current_action)

            next_state = update_state_vector(state_vector, state_frame)

            state_vector = next_state
            total_rewards += reward

            if info['ale.lives'] < 5 or finished_episode:
                games_played += 1
                break

    print "{} game average reward: ".format(num_games), total_rewards / float(num_games)


for epoch in xrange(NUM_EPOCHS):
    if finished_episode:

        if num_episodes % 100 == 0:
            evaluate_model(q_function, 10, env)

        obs = env.reset()
        num_nop = np.random.randint(4, 30, size=1)
        last_four = num_nop - 4
        starter_frames = []

        # Perform a null operation to make game stochastic
        for k in xrange(num_nop):
            obs, reward, is_terminal, info = env.step(0)
            if k >= last_four:
                starter_frames.append(obs)

        starter_frames = map(normalized_image, starter_frames)
        state_vector = np.stack(starter_frames, axis=2).astype(np.float32)

        if num_episodes % 100 == 0:
            print("Episode: ", num_episodes, "100 episode avg reward: ", total_episode_reward/100)
            total_episode_reward = 0
        num_episodes += 1

    for play in xrange(4):
        EPSILON = 0.5 #(((0.1 -1) / EPSILON_END_FRAME) * epoch + 1) if epoch < EPSILON_END_FRAME else 0.1
        current_state = {state_input: [state_vector]}

        if len(REPLAY_MEMORY) < REPLAY_START_SIZE or np.random.rand(1)[0] < EPSILON:
            next_action = env.action_space.sample()
        else:
            current_q = sess.run([q_function], current_state)[0][0]
            next_action = random_arg_max(current_q)

        total_skip_reward = 0

        lives = ale_info['ale.lives']
        terminal = False

        for repeat_action in xrange(NUM_REPEATS):
            state_frame, reward, finished_episode, ale_info = env.step(next_action)
            total_skip_reward += reward

            if finished_episode or ale_info['ale.lives'] < lives:
                terminal = True
                break

        next_state = update_state_vector(state_vector, state_frame)

        observation = Observation(state=state_vector, action=next_action, reward=total_skip_reward, next_state=next_state, terminal=terminal)
        REPLAY_MEMORY.append(observation)

        state_vector = next_state
        total_episode_reward += total_skip_reward

        if len(REPLAY_MEMORY) > REPLAY_START_SIZE:
            frame_count += 1

    # Train a batch
    if len(REPLAY_MEMORY) >= REPLAY_START_SIZE:
        for i in xrange(NUM_BATCHES):
            states = []
            actions = []
            next_states = []
            rewards = []
            terminal_masks = []

            sample = random.sample(REPLAY_MEMORY, BATCH_SIZE)

            for observation in sample:
                states.append(observation.state)
                actions.append(observation.action)
                next_states.append(observation.next_state)
                rewards.append(observation.reward)
                terminal_masks.append(1.0 if observation.terminal else 0.0)

            _, summary = sess.run([model_update, merged_summaries], feed_dict={
                state_input: states,
                action_holder: actions,
                target_state_input: next_states,
                reward_input: rewards,
                terminal_mask: terminal_masks
            })

            writer.add_summary(summary, (epoch * NUM_BATCHES) + i)
            writer.flush()

    if epoch % 1000 == 0:
        print("Frame: ", epoch, "REPLAY MEMORY SIZE: ", len(REPLAY_MEMORY), "EPSILON: ", EPSILON)

    if len(REPLAY_MEMORY) > REPLAY_START_SIZE and epoch % 10000 == 0:
        print("swapping network")
        sess.run(update_target_network())

saver = tf.train.Saver()
save_path = saver.save(sess, "/output/model.ckpt")