from random import random, randrange

import numpy as np
import tensorflow as tf

from gairl.agents.abstract_agent import AbstractAgent
from gairl.memory.replay_buffer import ReplayBuffer
from gairl.utils.neural_utils import create_copy_ops
from gairl.utils.neural_utils import DenseNetworkUtils as Dnu


# TODO add tensorboard, and model saving/loading
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
                 target_update_freq=10000,
                 logging_freq=1000):
        """
        Initializes feed-forward version of DQN
        :param actions_num: int; describes number of actions the
            agent can choose from.
        :param state_shape: tuple of ints; describes shape of the
            state input.
        :param hidden_layers: tuple of ints; describes number of nodes
            in each hidden layer of the feedforward network.
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

        tf.logging.info(
            f'\nCreating DQN Agent with:\n'
            f'Input shape {state_shape}\n'
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
            f'Target update frequency: {target_update_freq}\n'
        )

        super().__init__(actions_num, state_shape)

        # Set up important variables
        self._sess = session
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
        self._steps_so_far = 0
        self._episodes_so_far = 0
        self._episode_reward = 0

        # Set up input placeholders
        self._start_states = tf.placeholder(shape=(None, *state_shape),
                                            dtype=dtype)
        self._chosen_actions = tf.placeholder(shape=(None,), dtype=tf.int32)
        self._rewards = tf.placeholder(shape=(None,), dtype=dtype)
        self._next_states = tf.placeholder(shape=(None, *state_shape),
                                           dtype=dtype)

        # Create network
        self._online_params = Dnu.create_network_params(state_shape, hidden_layers,
                                                        actions_num, dtype)
        self._target_params = Dnu.create_network_params(state_shape, hidden_layers,
                                                        actions_num, dtype,
                                                        trainable=False)
        self._online_to_target_ops = create_copy_ops(
            Dnu.unpack_params(self._online_params),
            Dnu.unpack_params(self._target_params)
        )
        self._action, self._update_op = self._create_action_and_update_ops()

        self._initialize_vars()

    def _create_action_and_update_ops(self):
        online_start_qs = Dnu.model_output(self._start_states,
                                           self._online_params,
                                           self._activation_fn)
        target_next_qs = Dnu.model_output(self._next_states,
                                          self._target_params,
                                          self._activation_fn)
        best_online_actions = tf.argmax(online_start_qs, axis=1)

        # Set up online network update calculation
        chosen_online_qs = tf.gather(online_start_qs,
                                     self._chosen_actions, axis=1)
        best_target_next_qs = tf.reduce_max(target_next_qs, axis=1)
        expected_q_values = self._rewards + tf.scalar_mul(self._discount_factor,
                                                          best_target_next_qs)
        td_errors = tf.square(chosen_online_qs - expected_q_values)
        loss = tf.reduce_mean(td_errors)
        grads = self._optimizer.compute_gradients(loss)
        grads = [(tf.clip_by_value(grad, -self._gradient_clip, self._gradient_clip),
                 var) for grad, var in grads]
        online_update = self._optimizer.apply_gradients(grads)

        return best_online_actions, online_update

    def _initialize_vars(self):
        init_ops = tf.variables_initializer(
            Dnu.unpack_params(self._online_params) +
            Dnu.unpack_params(self._target_params) +
            self._optimizer.variables()
        )
        self._sess.run(init_ops)

    def step(self, state, reward=None):
        if reward:
            self._episode_reward += reward
            self._replay_buffer.add_experience(self._prev_state,
                                               self._prev_action,
                                               reward, state)
        else:
            self._episodes_so_far += 1
            self._episode_reward = 0

        if self._steps_so_far % self._update_freq == 0:
            self._update_networks()

        action = self._choose_action(state)
        self._update_epsilon()
        self._prev_state = state
        self._prev_action = action
        self._steps_so_far += 1

        tf.logging.log_every_n(
            tf.logging.INFO,
            '\n--------------------------------------------------\n'
            f'Step {self._steps_so_far-1}\n'
            f'Current state: {state}\n'
            f'Received reward: {reward}\n'
            f'Current episode total reward: {self._episode_reward}\n'
            f'Chosen action: {action}\n'
            f'Current epsilon: {self._curr_epsilon}\n'
            f'----------------------------------------------------\n',
            self._logging_freq
        )
        return action

    def _update_networks(self):
        samples = self._replay_buffer.replay_experience()
        if samples is None:
            return

        if self._steps_so_far % self._target_update_freq < self._update_freq:
            self._copy_online_to_target()

        self._sess.run(self._update_op,
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

        return self._sess.run(self._action,
                              feed_dict={self._start_states: [state]})[0]

    def _update_epsilon(self):
        if self._epsilon_warmup < self._steps_so_far and \
           self._steps_so_far - self._epsilon_warmup < self._epsilon_period:
            self._curr_epsilon -= self._epsilon_decay

    def _copy_online_to_target(self):
        tf.logging.info(' Updating target network with online params')
        for op in self._online_to_target_ops:
            self._sess.run(op)
