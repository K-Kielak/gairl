import logging

import numpy as np
import tensorflow as tf

from gairl.agents.dqn.dqn_agent import DQNAgent
from gairl.memory.prioritized_replay_buffer import PrioritizedReplayBuffer
from gairl.neural_utils import create_copy_ops
from gairl.neural_utils import summarize_vector, summarize_ndarray
from gairl.neural_utils import DenseNetworkUtils as Dnu


class RainbowDQNAgent(DQNAgent):

    def __init__(self,
                 actions_num,
                 state_size,
                 hidden_layers,
                 session,
                 output_directory,
                 name='RainbowDQN',
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
                 replay_buffer=PrioritizedReplayBuffer(2**20, 5000, 32),
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
        :param hidden_layers: tuple 3 tuples of ints; describes number
            of nodes in each hidden layer of the network. 1st tuple - shared
            part, 2nd tuple - advantage part, 3rd tuple - value part.
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
        :param logging_level: logging.LEVEL; level of the internal logger.
        :param max_checkpoints: int; number of checkpoints to keep.
        :param save_freq: int; how often the model will be saved.
        :param model_path: string; path of the model to load. If None
            then do not load any model and start from scratch.
        """
        super().__init__(actions_num,
                         state_size,
                         hidden_layers,
                         session,
                         output_directory,
                         name=name,
                         dtype=dtype,
                         activation_fn=activation_fn,
                         optimizer=optimizer,
                         gradient_clip=gradient_clip,
                         discount_factor=discount_factor,
                         epsilon_start=epsilon_start,
                         epsilon_warmup=epsilon_warmup,
                         epsilon_end=epsilon_end,
                         epsilon_period=epsilon_period,
                         replay_buffer=replay_buffer,
                         update_freq=update_freq,
                         target_update_freq=target_update_freq,
                         logging_freq=logging_freq,
                         logging_level=logging_level,
                         max_checkpoints=max_checkpoints,
                         save_freq=save_freq,
                         load_path=load_path)

    def _create_network_params(self, state_size, hidden_layers, actions_num):
        self._onl_sh_params, self._onl_adv_params, self._onl_val_params = \
            self._create_duelling_params(state_size, hidden_layers, actions_num,
                                         trainable=True, name='online_params')
        self._tar_sh_params, self._tar_adv_params, self._tar_val_params = \
            self._create_duelling_params(state_size, hidden_layers, actions_num,
                                         trainable=False, name='target_params')

        self._online_params = self._onl_sh_params + \
                              self._onl_adv_params + self._onl_val_params
        self._target_params = self._tar_sh_params + \
                              self._tar_adv_params + self._tar_val_params

        self._online_to_target_ops = create_copy_ops(
            Dnu.unpack_params(self._online_params),
            Dnu.unpack_params(self._target_params)
        )

    def _create_duelling_params(self, state_size, hidden_layers, actions_num,
                                trainable=True, name=''):
        shared_params = Dnu.create_network_params(state_size,
                                                  hidden_layers[0][:-1],
                                                  hidden_layers[0][-1],
                                                  self._dtype,
                                                  trainable=trainable,
                                                  name=f'{name}_shared')
        advantage_params = Dnu.create_network_params(hidden_layers[0][-1],
                                                     hidden_layers[1],
                                                     actions_num,
                                                     self._dtype,
                                                     trainable=trainable,
                                                     name=f'{name}_advantage')
        value_params = Dnu.create_network_params(hidden_layers[0][-1],
                                                 hidden_layers[2],
                                                 1,  # V is independent
                                                 self._dtype,
                                                 trainable=trainable,
                                                 name=f'{name}_value')
        return shared_params, advantage_params, value_params

    def _create_outputs(self):
        self._online_start_qs, self._onl_start_adv, self._onl_start_val = \
            self._create_duelling_out(self._start_states,
                                      self._onl_sh_params,
                                      self._onl_adv_params,
                                      self._onl_val_params,
                                      name='online_start')
        self._target_start_qs, _, _ = \
            self._create_duelling_out(self._start_states,
                                      self._tar_sh_params,
                                      self._tar_adv_params,
                                      self._tar_val_params,
                                      name='target_start')
        self._online_next_qs, self._online_next_advantage, _ = \
            self._create_duelling_out(self._next_states,
                                      self._onl_sh_params,
                                      self._onl_adv_params,
                                      self._onl_val_params,
                                      name='online_next')
        self._target_next_qs, _, _ = \
            self._create_duelling_out(self._next_states,
                                      self._tar_sh_params,
                                      self._tar_adv_params,
                                      self._tar_val_params,
                                      name='target_next')

        self._best_online_actions = tf.argmax(self._onl_start_adv,
                                              axis=1, name='best_actions')

    def _create_duelling_out(self, start_states, shared_params,
                             adv_params, val_params, name=''):
        shared_out = Dnu.model_output(start_states, shared_params,
                                      self._activation_fn,
                                      name=f'{name}_shared')
        active_share_out = self._activation_fn(shared_out)
        advantage_out = Dnu.model_output(active_share_out, adv_params,
                                         self._activation_fn,
                                         name=f'{name}_advantages')
        value_out = Dnu.model_output(active_share_out, val_params,
                                     self._activation_fn,
                                     name=f'{name}_value')
        average_advantage = tf.reduce_mean(advantage_out, axis=1,
                                           name=f'{name}_mean_advantage')
        average_advantage = tf.expand_dims(average_advantage, axis=1)
        qs_out = tf.add(value_out,  (advantage_out - average_advantage),
                        name=f'{name}_qs')
        return qs_out, advantage_out, value_out

    def _calc_expected_qs(self):
        next_chosen_actions = tf.argmax(self._online_next_advantage,
                                        axis=1, output_type=tf.int32,
                                        name='next_chosen_actions')
        action_indices = tf.range(tf.shape(next_chosen_actions)[0])
        action_indices = tf.stack([action_indices, next_chosen_actions], axis=1)
        next_chosen_qs = tf.gather_nd(self._target_next_qs, action_indices,
                                      name='next_chosen_qs')

        # If non-terminal then take best next qs into account, oterwise 0
        zeros = tf.zeros_like(next_chosen_qs)
        self._real_next_qs = tf.where(self._are_terminal, zeros,
                                      next_chosen_qs, name='real_next_qs')
        discounted_next_qs = tf.scalar_mul(self._discount_factor,
                                           self._real_next_qs)
        return tf.add(self._rewards, discounted_next_qs, name='expected_qs')

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
        with tf.name_scope('online-advantages'):
            output_summs.extend(summarize_vector(self._onl_start_adv[0, :]))
        with tf.name_scope('online-values'):
            output_summs.extend(summarize_vector(self._onl_start_val[0, :]))
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

    def _log_step(self, state, reward, action, is_terminal):
        out_summ, curr_online_qs, curr_online_adv, \
        curr_online_val, curr_target_qs = \
            self._sess.run([self._output_summary, self._online_start_qs,
                            self._onl_start_adv, self._onl_start_val,
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
            f'Current online advantage: {curr_online_adv[0]}\n'
            f'Current online value: {curr_online_val[0]}\n'
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
