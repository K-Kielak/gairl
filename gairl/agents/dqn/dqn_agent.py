from functools import reduce
from random import random, randrange

import numpy as np
import tensorflow as tf

from gairl.agents.abstract_agent import AbstractAgent
from gairl.memory.replay_buffer import ReplayBuffer
from gairl.utils.neural_utils import DenseLayerParams
from gairl.utils.neural_utils import gen_rand_biases, gen_rand_weights


# TODO add logging, tensorboard, and model saving/loading
class DQNAgent(AbstractAgent):

    def __init__(self,
                 actions_num,
                 state_shape,
                 hidden_layers,
                 session,
                 dtype=tf.float64,
                 activation_fn=tf.nn.leaky_relu,
                 optimizer=tf.train.AdamOptimizer(
                     learning_rate=6.25e-5,
                     epsilon=1.5e-4
                 ),
                 gradient_clip=1,
                 discount_factor=0.99,
                 epsilon_start=0.9,
                 epsilon_warmup=1000,
                 epsilon_end=0.05,
                 epsilon_period=100000,
                 replay_buffer=ReplayBuffer(250000, 80000, 32),
                 update_freq=4,
                 target_update_freq=10000):
        """
        Initializes feed-forward version of DQN
        :param actions_num: int; describes number of actions the
            agent can choose from.
        :param state_shape: tuple of ints; describes shape of the
            state input.
        :param hidden_layers: tuple of ints; describes number of nodes
            in each hidden layer of the feedforwad network.
        :param session: tensorflow..Session; tensorflow session that
            will be used to run the model.
        :param dtype: tensorflow.DType; type of the state input.
        :param activation_fn: activation function for hidden layers of networks
        :param optimizer: tf.trainOptimizer; optimizer that calculates
            gradients and updates online network accordingly.
        :param gradient_clip: float > 0; gradient applied during the networks
            update will be clipped between [-gradient_clip, gradient_clip]
        :param discount_factor: float in [0, 1]; describes time preference of
            the algorithm, how much future rewards are discounted with time.
        :param epsilon_start: float in [0, 1]; starting probability of
            taking random exploratory action.
        :param epsilon_warmup: int: how many steps the agent will
            perform before starting epsilon decay.
        :param epsilon_end: float in [0, 1]; ending probability of
            taking random exploratory action.
        :param epsilon_period: int; how many steps epsilon will decay
            from epsilon_start to epsilon_end.
        :param replay_buffer: gairl..ReplayBuffer; memory object
            storing, and replaying during training, agent's experience.
        :param update_freq: int; how many updates the agent
            will perform in-between online network updates.
        :param target_update_freq: int; how many online network
            updates the agent will perform in-between target network updates.
        """
        super().__init__(actions_num, state_shape)

        self._sess = session
        self._optimizer = optimizer
        self._gradient_clip = gradient_clip
        self._discount_factor = discount_factor
        self._curr_epsilon = epsilon_start
        self._epsilon_warmup = epsilon_warmup
        self._epsilon_period = epsilon_period
        self._epsilon_decay = (epsilon_start - epsilon_end) / epsilon_period
        self._replay_buffer = replay_buffer
        self._update_freq = update_freq
        self._target_update_freq = target_update_freq

        self._prev_state = None
        self._prev_action = None
        self._steps_so_far = 0

        # Create network parameters
        online_params = create_network_params(state_shape, hidden_layers,
                                              actions_num, dtype)
        target_params = create_network_params(state_shape, hidden_layers,
                                              actions_num, dtype,
                                              trainable=False)

        # Create copy from online to target network tensorflow operations
        self._online_to_target_ops = []
        for i in range(len(online_params)):
            weights_copy_ops = target_params[i].weights\
                .assign(online_params[i].weights)
            biases_copy_ops = target_params[i].biases\
                .assign(online_params[i].biases)
            self._online_to_target_ops.append(weights_copy_ops)
            self._online_to_target_ops.append(biases_copy_ops)

        # Set up placeholders
        self._start_states = tf.placeholder(shape=(None, *state_shape),
                                            dtype=dtype)
        self._chosen_actions = tf.placeholder(shape=(None,), dtype=tf.int32)
        self._rewards = tf.placeholder(shape=(None,), dtype=dtype)
        self._next_states = tf.placeholder(shape=(None, *state_shape),
                                           dtype=dtype)

        # Model outputs
        online_start_qs = model_output(self._start_states,
                                       online_params, activation_fn)
        target_next_outputs = model_output(self._next_states,
                                           target_params, activation_fn)

        # Make decision on action
        self._best_online_actions = tf.argmax(online_start_qs, axis=1)

        # Update calculation
        chosen_online_qs = tf.gather(online_start_qs,
                                     self._chosen_actions, axis=1)
        best_target_next_qs = tf.reduce_max(target_next_outputs, axis=1)
        expected_q_values = self._rewards + tf.scalar_mul(self._discount_factor,
                                                          best_target_next_qs)
        td_errors = tf.square(chosen_online_qs - expected_q_values)
        loss = tf.reduce_mean(td_errors)
        grads = self._optimizer.compute_gradients(loss)
        grads = [(tf.clip_by_value(grad, -self._gradient_clip, self._gradient_clip),
                  var) for grad, var in grads]
        self._online_update = optimizer.apply_gradients(grads)

        # Initialize variables
        init_ops = tf.variables_initializer(
            reduce(lambda x, y: x.extend(list(y)) or list(x), online_params, []) +
            reduce(lambda x, y: x.extend(list(y)) or list(x), target_params, []) +
            self._optimizer.variables()
        )
        self._sess.run(init_ops)

    def step(self, state, reward=None):
        if reward:
            self._replay_buffer.add_experience(self._prev_state,
                                               self._prev_action,
                                               reward, state)

        if self._steps_so_far % self._update_freq == 0:
            self._update_networks()

        action = self._choose_action(state)
        self._update_epsilon()
        self._prev_state = state
        self._prev_action = action
        self._steps_so_far += 1
        return action

    def _update_networks(self):
        samples = self._replay_buffer.replay_experience()
        if samples is None:
            return

        if self._steps_so_far % self._target_update_freq < self._update_freq:
            self._copy_online_to_target()

        self._sess.run(self._online_update,
                       feed_dict={
                           self._start_states: np.vstack(samples[:, 0]),
                           self._chosen_actions: samples[:, 1],
                           self._rewards: samples[:, 2],
                           self._next_states: np.vstack(samples[:, 3])
                       })

    def _choose_action(self, state):
        if random() < self._curr_epsilon or \
           self._steps_so_far < self._epsilon_warmup:
            return randrange(self._actions_num)

        return self._sess.run(self._best_online_actions,
                              feed_dict={self._start_states: [state]})[0]

    def _update_epsilon(self):
        if self._epsilon_warmup < self._steps_so_far and \
           self._steps_so_far - self._epsilon_warmup < self._epsilon_period:
            self._curr_epsilon -= self._epsilon_decay

    def _copy_online_to_target(self):
        for op in self._online_to_target_ops:
            self._sess.run(op)


def create_network_params(input_shape, hidden_layers, outputs_num, dtype,
                          trainable=True):
    """
    Crates tensorflow variables for dense feedforward neural network.
    :param input_shape: tuple of ints; shape of the input to the network.
    :param hidden_layers: tuple of ints; number of nodes in each hidden
        layer of the network.
    :param outputs_num: int; number of outputs network should produce.
    :param dtype: numpy.Dtype; what type should the params have.
    :param trainable: bool; will these params be trainable, i.e.
        if they can be updated by tensorflow training algorithms.
    :return: list of gairl..DenseLayerParams; tensorflow variables
        representing network weights and biases.
    """
    if not hidden_layers:
        raise AttributeError('DQN has to have some hidden layers!')

    params = []
    layers = list(input_shape) + hidden_layers + [outputs_num]
    for i in range(1, len(layers)):
        weights = gen_rand_weights((layers[i - 1], layers[i]),
                                   dtype, trainable=trainable)
        biases = gen_rand_biases((layers[i],), dtype, trainable=trainable)
        params.append(DenseLayerParams(weights, biases))

    return params


def model_output(input, params, activation_fn):
    """
    :param input: tf.placeholder; placeholder for network input
    :param params: gairl..DenseLayerParams; tensorflow variables
        representing network weights and biases.
    :param activation_fn: function applied to result of each hidden
        layer of the network.
    :return: Final output of the network as a tensorflow tensor
    """
    layer_sum = tf.matmul(input, params[0].weights) + params[0].biases
    activation = activation_fn(layer_sum)

    # Up to len(params) - 1 because last layer doesn't use activation_fn
    for i in range(1, len(params) - 1):
        layer_sum = tf.matmul(activation, params[i].weights) + params[i].biases
        activation = activation_fn(layer_sum)

    return tf.matmul(activation, params[-1].weights) + params[-1].biases
