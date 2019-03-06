import os
from inspect import getfullargspec

from gairl.agents.dqn import dqn_config as dqn_conf
from gairl.agents.dqn.dqn_agent import DQNAgent
from gairl.agents.gairl import gairl_config as gairl_conf
from gairl.agents.gairl.gairl_agent import GAIRLAgent
from gairl.agents.rainbow import rainbow_config as rainbow_conf
from gairl.agents.rainbow.rainbow_agent import RainbowDQNAgent
from gairl.agents.random_agent import RandomAgent
from gairl.generators import create_generator


def create_agent(agent_name,
                 actions_num,
                 state_size,
                 session=None,
                 name=None,
                 output_dir=None,
                 separate_logging=True,
                 data_ranges=(-1, 1)):
    if agent_name not in _STR_TO_AGENT.keys():
        raise AttributeError(f"There's no agent like {agent_name}.")

    if agent_name == 'gairl':
        return _create_gairl_agent(actions_num, state_size, session,
                                   name=name, output_dir=output_dir,
                                   separate_logging=separate_logging,
                                   data_ranges=data_ranges)

    creation_method = _STR_TO_AGENT[agent_name]
    if 'session' in getfullargspec(creation_method).args:
        return creation_method(actions_num, state_size, session,
                               name=name, output_dir=output_dir,
                               separate_logging=separate_logging)

    return creation_method(actions_num, state_size)


def _create_random_agent(actions_num, state_size):
    return RandomAgent(actions_num, state_size)


def _create_dqn_agent(actions_num, state_size, session, name=None,
                      output_dir=None, separate_logging=True):
    output_dir = output_dir if output_dir else dqn_conf.OUTPUT_DIRECTORY
    name = name if name else 'DQN'
    logging_level = dqn_conf.LOGGING_LEVEL if separate_logging else None

    return DQNAgent(actions_num,
                    state_size,
                    dqn_conf.HIDDEN_LAYERS,
                    session,
                    output_dir,
                    name=name,
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
                    logging_level=logging_level,
                    max_checkpoints=dqn_conf.MAX_CHECKPOINTS,
                    save_freq=dqn_conf.SAVE_FREQUENCY,
                    load_path=dqn_conf.MODEL_LOAD_PATH)


def _create_rainbowdqn_agent(actions_num, state_size, session, name=None,
                             output_dir=None, separate_logging=True):
    output_dir = output_dir if output_dir else dqn_conf.OUTPUT_DIRECTORY
    name = name if name else 'RainbowDQN'
    logging_level = rainbow_conf.LOGGING_LEVEL if separate_logging else None

    return RainbowDQNAgent(actions_num,
                           state_size,
                           rainbow_conf.HIDDEN_LAYERS,
                           session,
                           output_dir,
                           name=name,
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
                           logging_level=logging_level,
                           max_checkpoints=rainbow_conf.MAX_CHECKPOINTS,
                           save_freq=rainbow_conf.SAVE_FREQUENCY,
                           load_path=rainbow_conf.MODEL_LOAD_PATH)


def _create_gairl_agent(actions_num,
                        state_size,
                        session,
                        name=None,
                        output_dir=None,
                        separate_logging=True,
                        data_ranges=(-1, 1)):
    output_dir = output_dir if output_dir else gairl_conf.OUTPUT_DIRECTORY
    name = name if name else 'GAIRL'

    # Create directory for RL agent and generative model
    os.mkdir(output_dir)

    gen_data_shape = (state_size + 1,)  # + 1 for is_terminal flag
    cond_data_shape = (state_size + actions_num,)
    gen_output_dir = os.path.join(output_dir, 'model')
    generative_model = create_generator(gairl_conf.GENERATIVE_MODEL_STR,
                                        gen_data_shape,
                                        session,
                                        data_ranges=data_ranges,
                                        conditional_shape=cond_data_shape,
                                        name=name,
                                        output_dir=gen_output_dir,
                                        separate_logging=False)

    rl_output_dir = os.path.join(output_dir, 'agent')
    rl_agent = create_agent(gairl_conf.RL_AGENT_STR,
                            actions_num,
                            state_size,
                            session=session,
                            name=name,
                            output_dir=rl_output_dir,
                            separate_logging=False)

    logging_level = gairl_conf.LOGGING_LEVEL if separate_logging else None
    return GAIRLAgent(actions_num,
                      state_size,
                      rl_agent,
                      generative_model,
                      output_dir,
                      replay_buffer=gairl_conf.REPLAY_BUFFER,
                      model_free_steps=gairl_conf.MODEL_FREE_STEPS,
                      model_training_steps=gairl_conf.MODEL_TRAINING_STEPS,
                      model_based_steps=gairl_conf.MODEL_BASED_STEPS,
                      logging_level=logging_level)


_STR_TO_AGENT = {
    'random': _create_random_agent,
    'dqn': _create_dqn_agent,
    'rainbowdqn': _create_rainbowdqn_agent,
    'gairl': _create_gairl_agent
}
