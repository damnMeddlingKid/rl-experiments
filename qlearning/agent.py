from __future__ import unicode_literals
import tensorflow as tf


class DQNAgent(object):

    @property
    def input_state(self):
        return self._input

    @property
    def q_value(self):
        return self._output

    def __init__(self, name, observation_shape, action_space, discount, target_agent=None):
        super(DQNAgent, self).__init__()
        self._name = name
        self._input, self._output = self.build_model(name, observation_shape, action_space)
        self._discount = discount
        self._target_agent = target_agent

        with tf.variable_scope(name):
            self._action = tf.placeholder(tf.int32, shape=[None], name="action")
            self._reward = tf.placeholder(tf.float32, shape=[None], name="reward")
            self._terminal = tf.placeholder(tf.float32, shape=[None], name="terminal")

            if target_agent:
                self._loss = self._loss()
                self._update = self._optimizer(self._loss, name)

        self._merged_summaries = tf.summary.merge_all()

    @staticmethod
    def build_model(name, observation_shape, action_shape):
        with tf.variable_scope(name):
            observation = tf.placeholder(tf.float32, (None,) + observation_shape, 'input_state')

            conv1 = tf.layers.conv2d(
                inputs=observation,
                filters=32,
                kernel_size=[8, 8],
                strides=(4, 4),
                padding='valid',
                activation=tf.nn.relu,
                name="conv1"
            )

            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=64,
                kernel_size=[4, 4],
                strides=(2, 2),
                padding='valid',
                activation=tf.nn.relu,
                name="conv2"
            )

            conv3 = tf.layers.conv2d(
                inputs=conv2,
                filters=64,
                kernel_size=[3, 3],
                padding='valid',
                activation=tf.nn.relu,
                name="conv3"
            )

            flat = tf.contrib.layers.flatten(conv3)

            dense1 = tf.layers.dense(flat, 512, activation=tf.nn.relu, name="dense1")
            dense2 = tf.layers.dense(dense1, action_shape, name="q_value")

            return observation, dense2

    def _loss(self):
        assert self._target_agent is not None, "Attempting to train agent without target"
        one_hot = tf.one_hot(self._action, 4)
        prediction = tf.reduce_sum(one_hot * self.q_value, axis=1)
        target = self._reward + (1 - self._terminal) * self._discount * tf.reduce_max(self._target_agent.q_value, axis=1)
        loss = tf.losses.huber_loss(target, prediction, reduction='weighted_mean')
        tf.summary.scalar("PredictedQ", tf.reduce_mean(prediction))
        tf.summary.scalar("TrueQ", tf.reduce_mean(target))
        tf.summary.scalar("Loss", loss)
        return loss

    def _optimizer(self, loss, name):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=0.00025,
            momentum=0.95
        )
        return optimizer.minimize(loss, global_step, var_list=variables)

    def infer(self, session, state):
        return session.run([self.q_value], {
            self.input_state: state
        })

    def train(self, session, batch):
        _, summary = session.run([self._update, self._merged_summaries], {
            self.input_state: batch['current_state'],
            self._target_agent.input_state: batch['future_state'],
            self._action: batch['action'],
            self._reward: batch['reward'],
            self._terminal: batch['terminal'],
        })
        return summary

    def sync_target(self, session):
        active_variables = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name), key=lambda var: var.name)
        target_variables = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._target_agent._name), key=lambda var: var.name)
        session.run([tf.assign(target, active, validate_shape=True) for target, active in zip(target_variables, active_variables)])
