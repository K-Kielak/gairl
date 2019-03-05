import logging
import os
import sys
import numpy as np

from gairl.agents.abstract_agent import AbstractAgent
from gairl.memory.replay_buffer import ReplayBuffer


class GAIRLAgent(AbstractAgent):

    def __init__(self,
                 actions_num,
                 state_size,
                 rl_agent,
                 generative_model,
                 output_directory,
                 name='GAIRL',
                 replay_buffer=ReplayBuffer(1000000, 256, 256),
                 model_free_steps=1000,
                 model_training_steps=10000,
                 model_based_steps=10000,
                 logging_level=logging.INFO):
        """
        Initializes GAIRL agent
        :param actions_num: int; describes number of actions the
            agent can choose from.
        :param state_size: int; describes size of the state vector.
        :param rl_agent: gairl..agent;
        :param generative_model:
        :param replay_buffer:
        :param model_free_steps:
        :param model_training_steps:
        :param model_based_steps:
        """
        super().__init__(actions_num, state_size)

        self._name = name
        self._rl_agent = rl_agent
        self._generative_model = generative_model
        self._replay_buffer = replay_buffer

        self._model_free_steps = model_free_steps
        self._model_training_steps = model_training_steps
        self._model_based_steps = model_based_steps

        self._model_batch_size = replay_buffer._replay_batch_size
        self._noise_size = self._generative_model._noise_size
        self._action_onehot_opts = np.eye(actions_num)

        self._real_steps_so_far = 0
        self._prev_state = None
        self._prev_action = None
        self._was_terminal = True

        self._set_up_outputs(output_directory, logging_level)

    def _set_up_outputs(self, output_dir, logging_level):
        # GAIRL is the only agent that's directory can be created beforehand
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        self._logger = logging.getLogger(self._name)
        if logging_level:  # Set up separate logger if not None
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

    def step(self, state, reward=0, is_terminal=False):
        if not self._was_terminal:
            self._replay_buffer.add_experience(self._prev_state,
                                               self._prev_action,
                                               reward, state, is_terminal)

        action = self._rl_agent.step(state, reward=reward,
                                     is_terminal=is_terminal)
        self._prev_state = state
        self._prev_action = action
        self._was_terminal = is_terminal
        self._real_steps_so_far += 1

        if self._real_steps_so_far % self._model_free_steps == 0:
            # Train generative model based on aggregated data
            self._logger.info(
                f'\n********************************************************\n'
                f'********************************************************\n'
                f'**************  Training generative model  *************\n'
                f'********************************************************\n'
                f'********************************************************\n'
            )
            for _ in range(self._model_training_steps):
                self._train_generative_model()

            self._logger.info(
                f'\n********************************************************\n'
                f'********************************************************\n\n'
                f'\n********************************************************\n'
                f'********************************************************\n'
                f'**************  Back to real environment  **************\n'
                f'********************************************************\n'
                f'********************************************************\n'
            )

        return action

    def _train_generative_model(self):
        experience = self._replay_buffer.replay_experience()
        start_states = np.vstack(experience[:, 0])
        actions = np.stack(experience[:, 1])
        actions_onehot = self._action_onehot_opts[actions]
        conditional_in = np.concatenate((start_states, actions_onehot), axis=1)

        next_states = np.vstack(experience[:, 3])
        are_terminal = np.vstack(experience[:, 4])
        to_generate = np.concatenate((next_states, are_terminal), axis=1)

        noise = np.random.normal(0, 1, (self._model_batch_size,
                                        self._noise_size))
        self._generative_model.train_step(to_generate, noise,
                                          g_condition=conditional_in)
