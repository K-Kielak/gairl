import logging
import os
import sys
from collections import deque
from random import random

import numpy as np
import tensorflow as tf

from gairl.agents.abstract_agent import AbstractAgent
from gairl.memory.replay_buffer import ReplayBuffer


class GAIRLAgent(AbstractAgent):

    def __init__(self,
                 actions_num,
                 state_size,
                 rl_agent,
                 generative_model,
                 session,
                 output_directory,
                 name='GAIRL',
                 dtype=tf.float64,
                 model_free_steps=1000,
                 model_training_steps=10000,
                 model_mem_size=100000,
                 model_batch_size=256,
                 model_test_size=0.2,
                 model_based_steps=10000,
                 logging_freq=5000,
                 logging_level=logging.INFO):
        """
        Initializes GAIRL agent
        :param actions_num: int; describes number of actions the
            agent can choose from.
        :param state_size: int; describes size of the state vector.
        :param rl_agent: gairl..agent; reinforcement learning agent
            used as a part of GAIRL.
        :param generative_model: garil..generator; generator used for
            state prediction as part of gairl.
        :param session: tensorflow..Session; tensorflow session that
            will be used to run the model.
        :param output_directory: string; directory to which all of the
            network outputs (logs, checkpoints) will be saved.
        :param name: string; name of the network.
        :param model_free_steps: int; how many model-free steps are
            performed over single GAIRL iteration.
        :param model_training_steps: int; how many model training
            iterations are performed over single GAIRL iteration.
        :param model_mem_size: int; size of the total GAIRL memory.
        :param model_batch_size: int; batch size used for training
            model generator.
        :param model_test_size: int in (0, 1); what part of total
            memory will be used as a test set for generator evaluation.
        :param model_based_steps: int; how many reinforcement
            learning steps will be performed on generator model
            over single GAIRL iteration.
        :param logging_freq: int; frequency of progress logging and
            writing tensorflow summaries.
        :param logging_level: logging.LEVEL; level of the internal logger,
            None if it already reuses existing logger.
        """
        super().__init__(actions_num, state_size)

        self._name = name
        self._sess = session
        self._dtype = dtype
        self._rl_agent = rl_agent
        self._generative_model = generative_model

        self._model_free_steps = model_free_steps
        self._model_training_steps = model_training_steps
        train_mem_size = model_mem_size * (1-model_test_size)
        self._training_memory = ReplayBuffer(train_mem_size,
                                             model_batch_size,
                                             model_batch_size)
        self._model_test_size = model_test_size
        test_mem_size = model_mem_size * model_test_size
        self._test_memory = ReplayBuffer(test_mem_size,
                                         min((model_batch_size, test_mem_size)),
                                         min((model_batch_size, test_mem_size)))
        self._model_batch_size = model_batch_size
        self._model_based_steps = model_based_steps
        self._logging_freq = logging_freq

        self._action_onehot_opts = np.eye(actions_num)

        self._real_steps_so_far = 0
        self._model_training_steps_so_far = 0
        self._prev_state = None
        self._prev_action = None
        self._was_terminal = True
        self._steps_per_episode = 0
        self._episodes_so_far = 0
        self._avg_episode_length = 0.
        self._100_episode_rewards = deque([0.])  # stores last 100 episodes

        self._set_up_outputs(output_directory, logging_level)
        self._add_summaries()

    def _set_up_outputs(self, output_dir, logging_level):
        # GAIRL is the only agent that's directory can be created beforehand
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        sumaries_dir = os.path.join(output_dir, 'tensorboard')

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
        # Reusing RL agent summary as tensorflow doesn't allow
        # to plot 2 variables under the same plot
        # Placeholders for tensorboard
        self._avg_ep_length_ph = self._rl_agent._avg_ep_length_ph
        self._ep_reward_ph = self._rl_agent._ep_reward_ph
        self._avg_ep_reward_ph = self._rl_agent._avg_ep_reward_ph
        self._rl_summary = self._rl_agent._ep_summary

        self._gen_train_loss_ph = tf.placeholder(dtype=self._dtype, shape=(),
                                                 name='gen_train_loss')
        self._gen_test_loss_ph = tf.placeholder(dtype=self._dtype, shape=(),
                                                name='gen_test_loss')
        gen_summs = []
        with tf.name_scope('losses/'):
            gen_summs.append(tf.summary.scalar('gen_train',
                                               self._gen_train_loss_ph))
            gen_summs.append(tf.summary.scalar('gen_test',
                                               self._gen_test_loss_ph))
        self._gen_summary = tf.summary.merge(gen_summs)

    def step(self, state, reward=0, is_terminal=False):
        action = self._rl_agent.step(state, reward=reward,
                                     is_terminal=is_terminal)

        if not self._was_terminal:
            self._100_episode_rewards[-1] += reward

            # Add experience with proportional probability
            if random() < self._model_test_size:
                self._test_memory.add_experience(self._prev_state,
                                                 self._prev_action,
                                                 reward, state, is_terminal)
            else:
                self._training_memory.add_experience(self._prev_state,
                                                     self._prev_action,
                                                     reward, state, is_terminal)

        if is_terminal:
            self._avg_episode_length = (self._episodes_so_far *
                                        self._avg_episode_length +
                                        self._steps_per_episode) / \
                                       (self._episodes_so_far + 1)
            rl_summary = self._sess.run(self._rl_summary, feed_dict={
                    self._ep_reward_ph: self._100_episode_rewards[-1],
                    self._avg_ep_reward_ph: np.mean(self._100_episode_rewards),
                    self._avg_ep_length_ph: self._avg_episode_length
                })
            self._summary_writer.add_summary(rl_summary,
                                             self._real_steps_so_far)
            self._steps_per_episode = 0
            self._episodes_so_far += 1
            self._100_episode_rewards.append(0.)
            if len(self._100_episode_rewards) > 100:
                self._100_episode_rewards.popleft()
            self._prev_state = None
            self._prev_action = None
            self._was_terminal = True
            return None

        self._prev_state = state
        self._prev_action = action
        self._was_terminal = is_terminal
        self._real_steps_so_far += 1
        self._steps_per_episode += 1

        if self._real_steps_so_far % self._model_free_steps == 0:
            # Train generative model based on aggregated data
            self._logger.info(
                f'\n******************************************************\n'
                f'************  Training generative model  *************\n'
                f'******************************************************\n'
            )
            for _ in range(self._model_training_steps):
                self._train_generative_model()

            self._logger.info(
                f'\n******************************************************\n\n'
                f'\n******************************************************\n'
                f'************  Back to real environment  **************\n'
                f'******************************************************\n'
            )

        return action

    def _train_generative_model(self):
        experience = self._training_memory.replay_experience()
        start_states = np.vstack(experience[:, 0])
        actions = np.stack(experience[:, 1])
        actions_onehot = self._action_onehot_opts[actions]
        conditional_in = np.concatenate((start_states, actions_onehot), axis=1)

        next_states = np.vstack(experience[:, 3])

        self._generative_model.train_step(next_states,
                                          condition=conditional_in)
        self._model_training_steps_so_far += 1
