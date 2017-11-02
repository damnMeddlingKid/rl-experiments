from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
from datetime import  datetime
from replay_memory import ReplayMemory
from game import Game
from agent import DQNAgent


tf.app.flags.DEFINE_float("DISCOUNT_FACTOR", 0.99, "Amount to discount future rewards.")
tf.app.flags.DEFINE_integer("PLAY_FRAMES", 4, "Number of frames to play before training a batch.")
tf.app.flags.DEFINE_integer("ACTION_REPETITION", 4, "Number of frames to repeat the action while playing.")
tf.app.flags.DEFINE_integer("HISTORY_SIZE", 4, "Number of frames of past history to model game state.")
tf.app.flags.DEFINE_integer("HISTORY_WIDTH", 80, "Width of the input frame.")
tf.app.flags.DEFINE_integer("HISTORY_HEIGHT", 80, "Height of the input frame.")
tf.app.flags.DEFINE_integer("ACTION_SPACE", 4, "Number of possible output actions.")
tf.app.flags.DEFINE_integer("REPLAY_MEMORY_LENGTH", 500000, "Number of historical experiences to store.")
tf.app.flags.DEFINE_integer("MIN_REPLAY_MEMORY_LENGTH", 50000, "Minimum number of experiences to start training.")
tf.app.flags.DEFINE_integer("BATCH_SIZE", 32, "Size of mini-batch.")
tf.app.flags.DEFINE_integer("TARGET_NETWORK_UPDATE_FREQUENCY", 100, "Rate at which to update the target network.")

tf.app.flags.DEFINE_integer("EPOCHS", 100000, "Number of training episodes.")
tf.app.flags.DEFINE_integer("TRAINING_ITERATIONS", 20, "Number of iterations per training episode.")

tf.app.flags.DEFINE_float("EPSILON_START", 1, "Starting value for probability of greediness.")
tf.app.flags.DEFINE_float("EPSILON_END", 0.1, "Ending value for probability of greediness.")
tf.app.flags.DEFINE_float("EPSILON_END_EPOCH", 90000, "Ending epoch to anneal epsilon.")

FLAGS = tf.app.flags.FLAGS


observation_shape = (FLAGS.HISTORY_WIDTH, FLAGS.HISTORY_HEIGHT, FLAGS.HISTORY_SIZE)
replay_memory = ReplayMemory(FLAGS.REPLAY_MEMORY_LENGTH, observation_shape)
game = Game('BreakoutNoFrameskip-v0', observation_shape, FLAGS.ACTION_REPETITION)
target_agent = DQNAgent("target_agent", observation_shape, FLAGS.ACTION_SPACE, FLAGS.DISCOUNT_FACTOR)
main_agent = DQNAgent("dqn_agent", observation_shape, FLAGS.ACTION_SPACE, FLAGS.DISCOUNT_FACTOR, target_agent)


def epsilon(epoch):
    return (((FLAGS.EPSILON_END - FLAGS.EPSILON_START) / FLAGS.EPSILON_END_EPOCH) * epoch + 1) if epoch < FLAGS.EPSILON_END_EPOCH else FLAGS.EPSILON_END


def store_data(state_data):
    replay_memory.add(state_data['current_state'], state_data['action'], state_data['reward'], state_data['terminal'], state_data['future_state'])


def fill_replay_memory():
    while len(replay_memory) < FLAGS.MIN_REPLAY_MEMORY_LENGTH:
        state_data = game.play_random()
        store_data(state_data)
        if len(replay_memory) % 1000 == 0:
            print "Memory size: ", len(replay_memory)


def main(_):
    fill_replay_memory()
    session = tf.Session()
    writer = tf.summary.FileWriter('/output/logs/{}'.format(str(datetime.now()).split('.')[0]), graph=session.graph)
    session.run(tf.global_variables_initializer())
    main_agent.sync_target(session)

    for epoch in xrange(FLAGS.EPOCHS):
        if epoch % 1000 == 0:
            print "Epoch: ", epoch, "Epsilon: ", epsilon(epoch)

        for frame in xrange(FLAGS.PLAY_FRAMES):
            if False and np.random.rand(1)[0] < epsilon(epoch):
                state_data = game.play_random()
            else:
                q_values = main_agent.infer(session, [game.current_state])[0][0]
                action = np.argmax(q_values)
                state_data = game.play(action)

            store_data(state_data)

        for iteration in xrange(FLAGS.TRAINING_ITERATIONS):
            batch = replay_memory.sample(FLAGS.BATCH_SIZE)
            summary = main_agent.train(session, batch)
            writer.add_summary(summary, (epoch * FLAGS.TRAINING_ITERATIONS) + iteration)
            writer.flush()

        if epoch != 0 and (epoch % FLAGS.TARGET_NETWORK_UPDATE_FREQUENCY == 0):
            print "Swapping Network"
            main_agent.sync_target(session)

    saver = tf.train.Saver()
    saver.save(session, "/output/model.ckpt")

if __name__ == '__main__':
    tf.app.run()