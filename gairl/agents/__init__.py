import os
from inspect import getfullargspec

import numpy as np

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
                 state_ranges=(-1, 1),
                 action_ranges=(-1, 1),
                 reward_range=(-1, 1)):
    if agent_name not in _STR_TO_AGENT.keys():
        raise AttributeError(f"There's no agent like {agent_name}.")

    if agent_name == 'gairl':
        return _create_gairl_agent(actions_num, state_size, session,
                                   name=name, output_dir=output_dir,
                                   state_ranges=state_ranges,
                                   action_ranges=action_ranges,
                                   reward_range=reward_range)

    creation_method = _STR_TO_AGENT[agent_name]
    if 'session' in getfullargspec(creation_method).args:
        return creation_method(actions_num, state_size, session,
                               name=name, output_dir=output_dir)

    return creation_method(actions_num, state_size)


def _create_random_agent(actions_num, state_size):
    return RandomAgent(actions_num, state_size)


def _create_dqn_agent(actions_num, state_size, session,
                      name=None, output_dir=None):
    output_dir = output_dir if output_dir else dqn_conf.OUTPUT_DIRECTORY
    name = name if name else 'DQN'

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
                    logging_level=dqn_conf.LOGGING_LEVEL,
                    max_checkpoints=dqn_conf.MAX_CHECKPOINTS,
                    save_freq=dqn_conf.SAVE_FREQUENCY,
                    load_path=dqn_conf.MODEL_LOAD_PATH)


def _create_rainbowdqn_agent(actions_num, state_size, session,
                             name=None, output_dir=None):
    output_dir = output_dir if output_dir else dqn_conf.OUTPUT_DIRECTORY
    name = name if name else 'RainbowDQN'

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
                           logging_level=rainbow_conf.LOGGING_LEVEL,
                           max_checkpoints=rainbow_conf.MAX_CHECKPOINTS,
                           save_freq=rainbow_conf.SAVE_FREQUENCY,
                           load_path=rainbow_conf.MODEL_LOAD_PATH)


def _create_gairl_agent(actions_num,
                        state_size,
                        session,
                        name=None,
                        output_dir=None,
                        state_ranges=(-1, 1),
                        action_ranges=(-1, 1),
                        reward_range=(-1, 1)):
    output_dir = output_dir if output_dir else gairl_conf.OUTPUT_DIRECTORY
    name = name if name else 'GAIRL'

    # Create directory for RL agent and generative model
    os.mkdir(output_dir)

    rl_output_dir = os.path.join(output_dir, 'agent')
    rl_agent = create_agent(gairl_conf.RL_AGENT_STR,
                            actions_num,
                            state_size,
                            session=session,
                            name=f'{name} - {gairl_conf.RL_AGENT_STR}',
                            output_dir=rl_output_dir)

    state_model, reward_model, terminal_model = \
        _create_generative_for_gairl(actions_num, state_size, session,
                                     name, output_dir,
                                     state_ranges=state_ranges,
                                     action_ranges=action_ranges,
                                     reward_range=reward_range)

    return GAIRLAgent(actions_num,
                      state_size,
                      rl_agent,
                      state_model,
                      reward_model,
                      terminal_model,
                      session,
                      output_dir,
                      name=name,
                      dtype=gairl_conf.DTYPE,
                      model_free_steps=gairl_conf.MODEL_FREE_STEPS,
                      model_training_steps=gairl_conf.MODEL_TRAINING_STEPS,
                      model_mem_size=gairl_conf.MODEL_MEMORY_SIZE,
                      model_batch_size=gairl_conf.MODEL_BATCH_SIZE,
                      model_test_size=gairl_conf.MODEL_TEST_SIZE,
                      model_based_steps=gairl_conf.MODEL_BASED_STEPS,
                      logging_freq=gairl_conf.LOGGING_FREQUENCY,
                      logging_level=gairl_conf.LOGGING_LEVEL)


def _create_generative_for_gairl(actions_num, state_size,
                                 session, name, output_dir,
                                 state_ranges=(-1, 1),
                                 action_ranges=(-1, 1),
                                 reward_range=(-1, 1)):
    # Define output dirs
    state_output_dir = os.path.join(output_dir, 'state')
    reward_output_dir = os.path.join(output_dir, 'reward')
    terminal_output_dir = os.path.join(output_dir, 'terminal')

    cond_data_shape = (state_size + actions_num,)

    # Make sure state ranges are of correct size
    full_s_ranges = np.array(state_ranges)
    while len(full_s_ranges.shape) == 1 or len(full_s_ranges) < state_size:
        full_s_ranges = np.vstack((full_s_ranges, state_ranges))
    full_a_ranges = np.array(action_ranges)
    while len(full_a_ranges.shape) == 1 or len(full_a_ranges) < actions_num:
        full_a_ranges = np.vstack((full_a_ranges, action_ranges))
    conditional_ranges = np.vstack((full_s_ranges, full_a_ranges))

    gen_name = f'{name} - {gairl_conf.GENERATIVE_MODEL_STR}'
    state_model = create_generator(gairl_conf.GENERATIVE_MODEL_STR,
                                   (state_size,),
                                   session,
                                   name=gen_name + ' - state',
                                   data_ranges=full_s_ranges,
                                   conditional_shape=cond_data_shape,
                                   conditional_ranges=conditional_ranges,
                                   output_dir=state_output_dir)
    reward_model = create_generator(gairl_conf.GENERATIVE_MODEL_STR,
                                    (1,),
                                    session,
                                    name=gen_name + ' - reward',
                                    data_ranges=reward_range,
                                    conditional_shape=cond_data_shape,
                                    conditional_ranges=conditional_ranges,
                                    output_dir=reward_output_dir)
    terminal_model = create_generator(gairl_conf.GENERATIVE_MODEL_STR,
                                      (1,),
                                      session,
                                      name=gen_name + ' - terminal',
                                      data_ranges=(0, 1),
                                      conditional_shape=cond_data_shape,
                                      conditional_ranges=conditional_ranges,
                                      output_dir=terminal_output_dir)
    return state_model, reward_model, terminal_model


_STR_TO_AGENT = {
    'random': _create_random_agent,
    'dqn': _create_dqn_agent,
    'rainbowdqn': _create_rainbowdqn_agent,
    'gairl': _create_gairl_agent
}
