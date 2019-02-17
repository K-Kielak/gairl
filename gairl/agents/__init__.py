from inspect import getfullargspec

from gairl.agents.dqn import dqn_config as dqn_conf
from gairl.agents.dqn.dqn_agent import DQNAgent
from gairl.agents.rainbow import rainbow_config as rainbow_conf
from gairl.agents.rainbow.rainbow_agent import RainbowDQNAgent
from gairl.agents.random_agent import RandomAgent


def create_agent(agent_name, actions_num, state_size, session=None):
    if agent_name not in _STR_TO_AGENT.keys():
        raise AttributeError(f"There's no agent like {agent_name}. You "
                             f"can choose only from {_STR_TO_AGENT.keys()}")

    creation_method = _STR_TO_AGENT[agent_name]
    if 'session' in getfullargspec(creation_method).args:
        return creation_method(actions_num, state_size, session=session)

    return creation_method(actions_num, state_size)


def _create_random_agent(actions_num, state_size):
    return RandomAgent(actions_num, state_size)


def _create_dqn_agent(actions_num, state_size, session):
    return DQNAgent(actions_num,
                    state_size,
                    dqn_conf.HIDDEN_LAYERS,
                    session,
                    dqn_conf.OUTPUT_DIRECTORY,
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
                    target_update_freq=dqn_conf.TARGET_UPDATE_FREQ,
                    logging_freq=dqn_conf.LOGGING_FREQ,
                    logging_level=dqn_conf.LOGGING_LEVEL,
                    max_checkpoints=dqn_conf.MAX_CHECKPOINTS,
                    save_freq=dqn_conf.SAVE_FREQUENCY,
                    load_path=dqn_conf.MODEL_LOAD_PATH)


def _create_rainbowdqn_agent(actions_num, state_size, session):
    return RainbowDQNAgent(actions_num,
                           state_size,
                           rainbow_conf.HIDDEN_LAYERS,
                           session,
                           rainbow_conf.OUTPUT_DIRECTORY,
                           dtype=rainbow_conf.DTYPE,
                           activation_fn=rainbow_conf.ACTIVATION_FN,
                           optimizer=rainbow_conf.OPTIMIZER,
                           gradient_clip=rainbow_conf.GRADIENT_CLIP,
                           discount_factor=rainbow_conf.DISCOUNT_FACTOR,
                           epsilon_start=rainbow_conf.EPSILON_START,
                           epsilon_warmup=rainbow_conf.EPSILON_WARMUP,
                           epsilon_end=rainbow_conf.EPSILON_END,
                           epsilon_period=rainbow_conf.EPSILON_PERIOD,
                           replay_buffer=rainbow_conf.REPLAY_BUFFER,
                           update_freq=rainbow_conf.UPDATE_FREQ,
                           target_update_freq=rainbow_conf.TARGET_UPDATE_FREQ,
                           logging_freq=rainbow_conf.LOGGING_FREQ,
                           logging_level=rainbow_conf.LOGGING_LEVEL,
                           max_checkpoints=rainbow_conf.MAX_CHECKPOINTS,
                           save_freq=rainbow_conf.SAVE_FREQUENCY,
                           load_path=rainbow_conf.MODEL_LOAD_PATH)


_STR_TO_AGENT = {
    'random': _create_random_agent,
    'dqn': _create_dqn_agent,
    'rainbowdqn': _create_rainbowdqn_agent
}
