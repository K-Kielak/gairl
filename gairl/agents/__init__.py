from inspect import getfullargspec

import tensorflow as tf

from gairl.agents.dqn import dqn_config as dqn_conf
from gairl.agents.dqn.dqn_agent import DQNAgent
from gairl.agents.random_agent import RandomAgent


def create_agent(agent_name, actions_num, state_shape, session=None):
    if agent_name not in _STR_TO_AGENT.keys():
        raise AttributeError(f"There's no agent like {agent_name}. You "
                             f"can choose only from {_STR_TO_AGENT.keys()}")

    creation_method = _STR_TO_AGENT[agent_name]
    if 'session' in getfullargspec(creation_method).args:
        return creation_method(actions_num, state_shape, session=session)

    return creation_method(actions_num, state_shape)


def _create_random_agent(actions_num, state_shape):
    return RandomAgent(actions_num, state_shape)


def _create_dqn_agent(actions_num, state_shape, session):
    return DQNAgent(actions_num,
                    state_shape,
                    dqn_conf.HIDDEN_LAYERS,
                    session,
                    dtype=dqn_conf.DTYPE,
                    activation_fn=dqn_conf.ACTIVATION_FN,
                    optimizer=dqn_conf.OPTIMIZER,
                    gradient_clip=dqn_conf.GRADIENT_CLIP,
                    discount_factor=dqn_conf.DISCOUNT_FACTOR,
                    epsilon_start=dqn_conf.EPSILON_START,
                    epsilon_warmup=dqn_conf.EPSILON_WARMUP,
                    epsilon_end=dqn_conf.EPSILON_END,
                    epsilon_period=dqn_conf.EPSILON_PERIOD,
                    replay_buffer=dqn_conf.REPLAY_BUFFER,
                    update_freq=dqn_conf.UPDATE_FREQ,
                    target_update_freq=dqn_conf.TARGET_UPDATE_FREQ)


_STR_TO_AGENT = {
    'random': _create_random_agent,
    'dqn': _create_dqn_agent
}
