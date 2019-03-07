import logging
import os
import sys
from collections import deque
from random import random, randrange

import numpy as np
import tensorflow as tf

from gairl.agents.abstract_agent import AbstractAgent
from gairl.memory.replay_buffer import ReplayBuffer
from gairl.neural_utils import DenseNetworkUtils as Dnu
from gairl.neural_utils import summarize_ndarray, summarize_vector
from gairl.neural_utils import create_copy_ops


class DQNAgent(AbstractAgent):

    def __init__(self,
                 actions_num,
                 state_size,
                 hidden_layers,
                 session,
                 output_directory,
                 name='DQN',
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
                 logging_freq=1000,
                 logging_level=logging.INFO,
                 max_checkpoints=5,
                 save_freq=100000,
                 load_path=None):
        """
        Initializes feed-forward version of DQN
        :param actions_num: int; describes number of actions the
            agent can choose from.
        :param state_size: int; describes size of the state vector.
        :param hidden_layers: tuple of ints; describes number of nodes
            in each hidden layer of the feedforward network.
        :param session: tensorflow..Session; tensorflow session that
            will be used to run the model.
        :param output_directory: string; directory to which all of the
            network outputs (logs, checkpoints) will be saved.
        :param name: string; name of the network.
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
        :param logging_level: logging.LEVEL; level of the internal logger,
            None if it already reuses existing logger.
        :param max_checkpoints: int; number of checkpoints to keep.
        :param save_freq: int; how often the model will be saved.
        :param model_path: string; path of the model to load. If None
            then do not load any model and start from scratch.
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

        self._name = name
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
        self._save_freq = save_freq

        self._prev_state = None
        self._prev_action = None
        self._was_terminal = True
        self._steps_so_far = 0
        self._steps_per_episode = 0
        self._episodes_so_far = 0
        self._avg_episode_length = 0.
        self._100_episode_rewards = deque([0.])  # stores last 100 episodes
        # Placeholders for tensorboard
        self._avg_ep_length_ph = tf.placeholder(dtype=tf.float32, shape=(),
                                                name='avg_episode_length')
        self._ep_reward_ph = tf.placeholder(dtype=dtype, shape=(),
                                            name='episode_reward')
        self._avg_ep_reward_ph = tf.placeholder(dtype=dtype, shape=(),
                                                name='avg_episode_reward')

        # Set up input placeholders
        self._start_states = tf.placeholder(shape=(None, state_size),
                                            dtype=dtype, name='start_state')
        self._chosen_actions = tf.placeholder(shape=(None,), dtype=tf.int32,
                                              name='chosen_action')
        self._rewards = tf.placeholder(shape=(None,), dtype=dtype,
                                       name='reward')
        self._next_states = tf.placeholder(shape=(None, state_size),
                                           dtype=dtype, name='next_state')
        self._are_terminal = tf.placeholder(shape=(None,), dtype=bool,
                                            name='is_terminal')
        self._is_weights = tf.placeholder(shape=(None,), dtype=dtype,
                                          name='is_weights')

        # Create network
        self._create_network_params(state_size, hidden_layers, actions_num)
        self._create_outputs()
        self._create_update()

        self._set_up_outputs(output_directory, logging_level, max_checkpoints)
        self._add_summaries()

        # Initialize vars
        init_ops = tf.variables_initializer(self._unpacked_params)
        self._sess.run(init_ops)

        # Load network if load_path specified
        if load_path:
            self._logger.info(f'Loading the model from {load_path}')
            self._saver.restore(self._sess, load_path)

        self._logger.info(
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
            f'Replay buffer: {replay_buffer}\n'
            f'Update frequency: {update_freq}\n'
            f'Target update frequency: {target_update_freq}'
        )

    def _create_network_params(self, state_size, hidden_layers, actions_num):
        self._online_params = Dnu.create_network_params(state_size,
                                                        hidden_layers,
                                                        actions_num,
                                                        self._dtype,
                                                        name='online_params')
        self._target_params = Dnu.create_network_params(state_size,
                                                        hidden_layers,
                                                        actions_num,
                                                        self._dtype,
                                                        trainable=False,
                                                        name='target_params')
        self._online_to_target_ops = create_copy_ops(
            Dnu.unpack_params(self._online_params),
            Dnu.unpack_params(self._target_params)
        )
        self._unpacked_params = Dnu.unpack_params(self._online_params) + \
                                Dnu.unpack_params(self._target_params) + \
                                self._optimizer.variables()

    def _create_outputs(self):
        self._online_start_qs = Dnu.model_output(self._start_states,
                                                 self._online_params,
                                                 self._activation_fn,
                                                 name='online_start_qs')
        self._target_start_qs = Dnu.model_output(self._start_states,
                                                 self._target_params,
                                                 self._activation_fn,
                                                 name='target_start_qs')
        self._target_next_qs = Dnu.model_output(self._next_states,
                                                self._target_params,
                                                self._activation_fn,
                                                name='target_next_qs')
        self._best_online_actions = tf.argmax(self._online_start_qs, axis=1,
                                              name='best_actions')

    def _create_update(self):
        # Get online network qs for chosen actions
        action_indices = tf.range(tf.shape(self._chosen_actions)[0])
        action_indices = tf.stack([action_indices, self._chosen_actions], axis=1)
        chosen_online_qs = tf.gather_nd(self._online_start_qs, action_indices,
                                        name='chosen_online_qs')

        expected_qs = self._calc_expected_qs()
        self._td_errors = tf.square(chosen_online_qs - expected_qs,
                                    name='td_errors')
        weighted_td_errors = tf.multiply(self._is_weights, self._td_errors,
                                         name='weighted_td_errors')
        self._loss = tf.reduce_mean(weighted_td_errors, name='loss')
        grads = self._optimizer.compute_gradients(self._loss,
                                                  var_list=self._unpacked_params)
        grads = [(tf.clip_by_value(grad,
                                   -self._gradient_clip,
                                   self._gradient_clip), var)
                 for grad, var in grads]
        self._online_update = self._optimizer.apply_gradients(grads)

    def _calc_expected_qs(self):
        best_target_next_qs = tf.reduce_max(self._target_next_qs, axis=1,
                                            name='best_target_next_qs')
        # If non-terminal then take best next qs into account, oterwise 0
        zeros = tf.zeros_like(best_target_next_qs)
        self._real_next_qs = tf.where(self._are_terminal, zeros,
                                      best_target_next_qs, name='real_next_qs')
        discounted_next_qs = tf.scalar_mul(self._discount_factor,
                                           self._real_next_qs)
        return tf.add(self._rewards, discounted_next_qs, name='expected_qs')

    def _set_up_outputs(self, output_dir, logging_level, max_checkpoints):
        os.mkdir(output_dir)
        sumaries_dir = os.path.join(output_dir, 'tensorboard')
        checkpoints_dir = os.path.join(output_dir, 'checkpoints')
        os.mkdir(checkpoints_dir)
        self._ckpt_path = os.path.join(checkpoints_dir, 'ckpt')
        self._saver = tf.train.Saver(max_to_keep=max_checkpoints)

        self._logger = logging.getLogger(self._name)
        logs_filepath = os.path.join(output_dir, 'logs.log')
        formatter = logging.Formatter('%(asctime)s:%(name)s:'
                                      '%(levelname)s: %(message)s')
        file_handler = logging.FileHandler(logs_filepath)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        self._logger.setLevel(logging_level)

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

        ep_summs = []
        with tf.name_scope('episode'):
            ep_summs.append(tf.summary.scalar('reward', self._ep_reward_ph))
            ep_summs.append(tf.summary.scalar('avg_100_reward',
                                              self._avg_ep_reward_ph))
            ep_summs.append(tf.summary.scalar('avg_episode_length',
                                              self._avg_ep_length_ph))

        self._training_summary = tf.summary.merge(training_summs)
        self._output_summary = tf.summary.merge(output_summs)
        self._ep_summary = tf.summary.merge(ep_summs)

    def step(self, state, reward=0, is_terminal=False):
        action = self._choose_action(state)
        if not self._was_terminal:
            self._100_episode_rewards[-1] += reward
            self._replay_buffer.add_experience(self._prev_state,
                                               self._prev_action,
                                               reward, state, is_terminal)
            if self._steps_so_far % self._logging_freq == 0:
                self._log_step(state, reward, action, is_terminal)

        if is_terminal:
            self._avg_episode_length = (self._episodes_so_far *
                                        self._avg_episode_length +
                                        self._steps_per_episode) / \
                                       (self._episodes_so_far + 1)
            reward_summary = \
                self._sess.run(self._ep_summary, feed_dict={
                    self._ep_reward_ph: self._100_episode_rewards[-1],
                    self._avg_ep_reward_ph: np.mean(self._100_episode_rewards),
                    self._avg_ep_length_ph: self._avg_episode_length
                })
            self._summary_writer.add_summary(reward_summary,
                                             self._steps_so_far)
            self._steps_per_episode = 0
            self._episodes_so_far += 1
            self._100_episode_rewards.append(0.)
            if len(self._100_episode_rewards) > 100:
                self._100_episode_rewards.popleft()
            self._prev_state = None
            self._prev_action = None
            self._was_terminal = True
            return None

        if self._steps_so_far % self._update_freq == 0:
            self._update_networks()

        if self._steps_so_far % self._save_freq == 0:
            self._logger.info('Saving the model')
            self._saver.save(self._sess, self._ckpt_path,
                             global_step=self._steps_so_far)

        self._update_epsilon()
        self._prev_state = state
        self._prev_action = action
        self._was_terminal = False
        self._steps_so_far += 1
        self._steps_per_episode += 1
        return action

    def _choose_action(self, state):
        if random() < self._curr_epsilon or \
           self._steps_so_far < self._epsilon_warmup:
            return randrange(self._actions_num)

        return self._sess.run(self._best_online_actions,
                              feed_dict={self._start_states: [state]})[0]

    def _update_networks(self):
        samples = self._replay_buffer.replay_experience()
        if samples is None:
            return

        if self._replay_buffer.prioritized:
            samples, indices, is_weights = samples
        else:
            indices = None
            is_weights = [1]*len(samples)

        if self._steps_so_far % self._target_update_freq < self._update_freq:
            self._copy_online_to_target()

        _, loss, td_errors, train_summ = \
            self._sess.run([self._online_update, self._loss,
                            self._td_errors, self._training_summary],
                           feed_dict={
                               self._start_states: np.vstack(samples[:, 0]),
                               self._chosen_actions: samples[:, 1],
                               self._rewards: samples[:, 2],
                               self._next_states: np.vstack(samples[:, 3]),
                               self._are_terminal: samples[:, 4],
                               self._is_weights: is_weights
                            })

        if self._replay_buffer.prioritized:
            self._replay_buffer.update_priorities(indices, is_weights)

        if self._steps_so_far % self._logging_freq < self._update_freq:
            self._logger.info(
                f'Training with loss: {loss}\n'
                f'----------------------------------------------------\n'
            )
            self._summary_writer.add_summary(train_summ, self._steps_so_far)

    def _copy_online_to_target(self):
        if self._steps_so_far % self._logging_freq < self._target_update_freq:
            self._logger.info('Updating target network with online params')

        for op in self._online_to_target_ops:
            self._sess.run(op)

    def _update_epsilon(self):
        if self._epsilon_warmup < self._steps_so_far and \
           self._steps_so_far - self._epsilon_warmup < self._epsilon_period:
            self._curr_epsilon -= self._epsilon_decay

    def _log_step(self, state, reward, action, is_terminal):
        out_summ, curr_online_qs, curr_target_qs = \
            self._sess.run([self._output_summary, self._online_start_qs,
                            self._target_start_qs], feed_dict={
                                self._start_states: [state]
                            })

        prev_online_qs, real_curr_qs, sample_loss = \
            self._sess.run([self._online_start_qs, self._real_next_qs,
                            self._loss], feed_dict={
                                self._start_states: [self._prev_state],
                                self._chosen_actions: [self._prev_action],
                                self._rewards: [reward],
                                self._next_states: [state],
                                self._are_terminal: [is_terminal],
                                self._is_weights: [1]
                            })

        self._logger.info(
            '\n--------------------------------------------------\n'
            f'Previous step: {self._steps_so_far - 1}\n'
            f'Previous state: {self._prev_state}\n'
            f'Previous online qs: {prev_online_qs} (calculated now)\n'
            f'Previous action: {self._prev_action}\n'
            f'Received reward: {reward}\n'
            f'Episode cumulative reward: {self._100_episode_rewards[-1]}\n'
            f'***\n'
            f'Current step: {self._steps_so_far}\n'
            f'Is terminal: {is_terminal}\n'
            f'Current state: {state}\n'
            f'Current online qs: {curr_online_qs[0]}\n'
            f'Current target qs: {curr_target_qs[0]}\n'
            f'Current real qs: {real_curr_qs[0]}\n'
            f'Chosen action: {action}\n'
            f'Current sample loss: {sample_loss}\n'
            f'***\n'
            f'Episodes so far: {self._episodes_so_far}\n'
            f'Average episode length: {self._avg_episode_length}\n'
            f'Average reward over last 100 episodes: '
            f'{np.mean(self._100_episode_rewards)}\n'
            f'Current epsilon: {self._curr_epsilon}\n'
        )
        self._summary_writer.add_summary(out_summ, self._steps_so_far)
