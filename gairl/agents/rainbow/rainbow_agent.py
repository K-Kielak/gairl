import logging

import tensorflow as tf

from gairl.agents.dqn.dqn_agent import DQNAgent
from gairl.memory.prioritized_replay_buffer import PrioritizedReplayBuffer
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

    def _create_outputs(self):
        self._online_start_qs = Dnu.model_output(self._start_states,
                                                 self._online_params,
                                                 self._activation_fn,
                                                 name='online_start_qs')
        self._target_start_qs = Dnu.model_output(self._start_states,
                                                 self._target_params,
                                                 self._activation_fn,
                                                 name='target_start_qs')
        self._online_next_qs = Dnu.model_output(self._next_states,
                                                self._online_params,
                                                self._activation_fn,
                                                name='online_next_qs')
        self._target_next_qs = Dnu.model_output(self._next_states,
                                                self._target_params,
                                                self._activation_fn,
                                                name='target_next_qs')
        self._best_online_actions = tf.argmax(self._online_start_qs, axis=1,
                                              name='best_actions')

    def _calc_expected_qs(self):
        next_chosen_actions = tf.argmax(self._online_next_qs, axis=1,
                                        output_type=tf.int32,
                                        name='next_chosen_actions')
        action_indices = tf.range(tf.shape(next_chosen_actions)[0])
        action_indices = tf.stack([action_indices, next_chosen_actions], axis=1)
        next_chosen_qs = tf.gather_nd(self._target_next_qs, action_indices,
                                      name='next_online_qs')

        # If non-terminal then take best next qs into account, oterwise 0
        zeros = tf.zeros_like(next_chosen_qs)
        self._real_next_qs = tf.where(self._are_terminal, zeros,
                                      next_chosen_qs, name='real_next_qs')
        discounted_next_qs = tf.scalar_mul(self._discount_factor,
                                           self._real_next_qs)
        return tf.add(self._rewards, discounted_next_qs, name='expected_qs')
