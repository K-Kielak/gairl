import logging
import os
from random import random, randrange

import numpy as np
import tensorflow as tf

from gairl.agents.abstract_agent import AbstractAgent
from gairl.memory.replay_buffer import ReplayBuffer
from gairl.neural_utils import DenseNetworkUtils as Dnu
from gairl.neural_utils import summarize_ndarray, summarize_vector
from gairl.neural_utils import create_copy_ops


# TODO add model saving/loading
class DQNAgent(AbstractAgent):

    def __init__(self,
                 actions_num,
                 state_size,
                 hidden_layers,
                 session,
                 output_directory,
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
                 target_update_freq=10000,
                 logging_freq=1000):
        """
        Initializes feed-forward version of DQN
        :param actions_num: int; describes number of actions the
            agent can choose from.
        :param state_size: int; describes size of the state vector.
        :param hidden_layers: tuple of ints; describes number of nodes
            in each hidden layer of the feedforward network.
        :param session: tensorflow..Session; tensorflow session that
            will be used to run the model.
        :param output_directory: directory to which all of the network
            outputs (logs, checkpoints) will be saved.
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
        :param logging_freq: int; frequency of progress logging and
            writing tensorflow summaries.
        """
        assert session, 'DQN agent requires tensorflow session!'
        assert gradient_clip > 0, 'gradient_clip needs to be higher than 0'
        assert 0. <= discount_factor <= 1, \
            'discount_factor needs to be in [0; 1] range'
        assert 0. <= epsilon_start <= 1, 'epsilon needs to be in [0; 1] range'
        assert 0. <= epsilon_end <= 1, 'epsilon needs to be in [0; 1] range'
        assert target_update_freq > update_freq, \
            'target_update_freq needs to be higher than update_freq'

        super().__init__(actions_num, state_size)

        # Set up important variables
        self._sess = session
        self._dtype = dtype
        self._activation_fn = activation_fn
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
        self._logging_freq = logging_freq

        self._prev_state = None
        self._prev_action = None
        self._was_terminal = True
        self._steps_so_far = 0
        self._episodes_so_far = 0
        self._episode_reward = 0.
        # Placeholders for tensorboard
        self._ep_reward_ph = tf.placeholder(dtype=dtype, shape=(),
                                            name='episode_reward')

        # Set up input placeholders
        self._start_states = tf.placeholder(shape=(None, state_size),
                                            dtype=dtype)
        self._chosen_actions = tf.placeholder(shape=(None,), dtype=tf.int32)
        self._rewards = tf.placeholder(shape=(None,), dtype=tf.float64)
        self._next_states = tf.placeholder(shape=(None, state_size),
                                           dtype=dtype)
        self._are_terminal = tf.placeholder(shape=(None,), dtype=bool)

        # Create network
        self._online_params = Dnu.create_network_params(state_size,
                                                        hidden_layers,
                                                        actions_num,
                                                        dtype,
                                                        name='online_params')
        self._target_params = Dnu.create_network_params(state_size,
                                                        hidden_layers,
                                                        actions_num,
                                                        dtype,
                                                        trainable=False,
                                                        name='target_params')
        self._online_to_target_ops = create_copy_ops(
            Dnu.unpack_params(self._online_params),
            Dnu.unpack_params(self._target_params)
        )
        self._create_outputs_and_update()

        self._set_up_outputs(output_directory)
        self._add_summaries()
        self._initialize_vars()

        tf.logging.info(
            f'\nCreating DQN Agent with:\n'
            f'Input size {state_size}\n'
            f'Hidden layers: {hidden_layers}\n'
            f'Outputs number: {actions_num}\n'
            f'Activation function: {activation_fn.__name__}\n'
            f'Optimizer: {optimizer.__class__.__name__}\n'
            f'Gradient clip: {gradient_clip}\n'
            f'Discount factor: {discount_factor}\n'
            f'Starting epsilon: {epsilon_start}\n'
            f'Epsilon warmup period: {epsilon_warmup}\n'
            f'Ending epsilon: {epsilon_end}\n'
            f'Epsilon decay period: {epsilon_period}\n'
            f'Replay buffer type: {replay_buffer.__class__.__name__}\n'
            f'Update frequency: {update_freq}\n'
            f'Target update frequency: {target_update_freq}'
        )

    def _create_outputs_and_update(self):
        self._online_start_qs = Dnu.model_output(self._start_states,
                                                 self._online_params,
                                                 self._activation_fn,
                                                 name='online_start_out')
        self._target_start_qs = Dnu.model_output(self._start_states,
                                                 self._target_params,
                                                 self._activation_fn,
                                                 name='target_start_out')
        self._target_next_qs = Dnu.model_output(self._next_states,
                                                self._target_params,
                                                self._activation_fn,
                                                name='target_next_out')
        self._best_online_actions = tf.argmax(self._online_start_qs, axis=1)

        # Set up online network update calculation
        chosen_online_qs = tf.gather(self._online_start_qs,
                                     self._chosen_actions, axis=1)
        best_target_next_qs = tf.reduce_max(self._target_next_qs, axis=1)
        # If non-terminal then take best next qs into account, oterwise 0
        zeros = tf.zeros_like(best_target_next_qs)
        self._real_next_qs = tf.where(self._are_terminal, zeros,
                                      best_target_next_qs)

        expected_q_values = self._rewards + tf.scalar_mul(self._discount_factor,
                                                          self._real_next_qs)
        td_errors = tf.square(chosen_online_qs - expected_q_values)
        self._loss = tf.reduce_mean(td_errors, name='loss')
        grads = self._optimizer.compute_gradients(self._loss)
        grads = [(tf.clip_by_value(grad, -self._gradient_clip, self._gradient_clip),
                 var) for grad, var in grads]
        self._online_update = self._optimizer.apply_gradients(grads)

    def _set_up_outputs(self, output_dir):
        os.mkdir(output_dir)
        logs_filepath = os.path.join(output_dir, 'logs.log')
        sumaries_dir = os.path.join(output_dir, 'tensorboard')

        tflogger = logging.getLogger('tensorflow')
        tflogger.addHandler(logging.FileHandler(logs_filepath))

        self._summary_writer = tf.summary.FileWriter(sumaries_dir,
                                                     self._sess.graph)

    def _add_summaries(self):
        training_summs = []
        with tf.name_scope(f'online-layer'):
            for i, layer in enumerate(self._online_params):
                with tf.name_scope(f'{i}/weights'):
                    training_summs.extend(summarize_ndarray(layer.weights))
                with tf.name_scope(f'{i}/biases'):
                    training_summs.extend(summarize_ndarray(layer.biases))

        with tf.name_scope('target-layer'):
            for i, layer in enumerate(self._target_params):
                with tf.name_scope(f'{i}/weights'):
                    training_summs.extend(summarize_ndarray(layer.weights))
                with tf.name_scope(f'{i}/biases'):
                    training_summs.extend(summarize_ndarray(layer.biases))

        with tf.name_scope('loss'):
            training_summs.append(tf.summary.scalar('value', self._loss))

        output_summs = []
        with tf.name_scope('online-qs'):
            output_summs.extend(summarize_vector(self._online_start_qs[0, :]))
        with tf.name_scope('target-qs'):
            output_summs.extend(summarize_vector(self._target_start_qs[0, :]))

        with tf.name_scope('episode-reward'):
            self._ep_reward_summary = tf.summary.scalar('value',
                                                        self._ep_reward_ph)
        self._training_summary = tf.summary.merge(training_summs)
        self._output_summary = tf.summary.merge(output_summs)

    def _initialize_vars(self):
        init_ops = tf.variables_initializer(
            Dnu.unpack_params(self._online_params) +
            Dnu.unpack_params(self._target_params) +
            self._optimizer.variables()
        )
        self._sess.run(init_ops)

    def step(self, state, reward=0, is_terminal=False):
        action = self._choose_action(state)
        if not self._was_terminal:
            self._episode_reward += reward
            self._replay_buffer.add_experience(self._prev_state,
                                               self._prev_action,
                                               reward, state, is_terminal)
            if self._steps_so_far % self._logging_freq == 0:
                self._log_step(state, reward, action, is_terminal)

        if is_terminal:
            reward_summary = \
                self._sess.run(self._ep_reward_summary, feed_dict={
                               self._ep_reward_ph: self._episode_reward})
            self._summary_writer.add_summary(reward_summary,
                                             self._episodes_so_far)
            self._episodes_so_far += 1
            self._episode_reward = 0
            self._prev_state = None
            self._prev_action = None
            self._was_terminal = True
            return None

        if self._steps_so_far % self._update_freq == 0:
            self._update_networks()

        self._update_epsilon()
        self._prev_state = state
        self._prev_action = action
        self._was_terminal = False
        self._steps_so_far += 1

        return action

    def _log_step(self, state, reward, action, is_terminal):
        out_summ, prev_online_qs, prev_target_qs, \
            curr_target_qs, curr_real_qs = \
            self._sess.run([self._output_summary, self._online_start_qs,
                            self._target_start_qs, self._target_next_qs,
                            self._real_next_qs], feed_dict={
                                self._start_states: [self._prev_state],
                                self._chosen_actions: [self._prev_action],
                                self._rewards: [reward],
                                self._next_states: [state],
                                self._are_terminal: [is_terminal]
                            })

        tf.logging.info(
            '\n--------------------------------------------------\n'
            f'Step {self._steps_so_far}\n'
            f'Received reward: {reward}\n'
            f'Current state: {state}\n'
            f'Chosen action: {action}\n'
            f'Current episode total reward: {self._episode_reward}\n'
            f'Is terminal: {is_terminal}\n\n'
            f'Prev step values:\n'
            f'Prev online qs: {prev_online_qs[0, :]}\n'
            f'Prev target qs: {prev_target_qs[0, :]}\n'
            f'Curr target qs: {curr_target_qs[0, :]}\n'
            f'Curr real qs: {curr_real_qs[0]}\n'
            f'Current epsilon: {self._curr_epsilon}\n'
        )
        self._summary_writer.add_summary(out_summ, self._steps_so_far)

    def _update_networks(self):
        samples = self._replay_buffer.replay_experience()
        if samples is None:
            return

        if self._steps_so_far % self._target_update_freq < self._update_freq:
            self._copy_online_to_target()

        _, loss, train_summ = \
            self._sess.run([self._online_update, self._loss,
                            self._training_summary],
                           feed_dict={
                               self._start_states: np.vstack(samples[:, 0]),
                               self._chosen_actions: samples[:, 1],
                               self._rewards: samples[:, 2],
                               self._next_states: np.vstack(samples[:, 3]),
                               self._are_terminal: samples[:, 4]
                            })

        if self._steps_so_far % self._logging_freq < self._update_freq:
            tf.logging.info(
                f'Training with loss: {loss}\n'
                f'----------------------------------------------------\n'
            )
            self._summary_writer.add_summary(train_summ, self._steps_so_far)

    def _copy_online_to_target(self):
        tf.logging.info(' Updating target network with online params')
        for op in self._online_to_target_ops:
            self._sess.run(op)

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