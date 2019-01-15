from gairl.agents.random_agent import RandomAgent


def create_agent(agent_name, actions_num, state_shape):
    if agent_name not in _STR_TO_AGENT.keys():
        raise AttributeError(f"There's no agent like {agent_name}. You "
                             f"can choose only from {_STR_TO_AGENT.keys()}")

    creation_method = _STR_TO_AGENT[agent_name]
    return creation_method(actions_num, state_shape)


def _create_random_agent(actions_num, state_shape):
    return RandomAgent(actions_num, state_shape)


_STR_TO_AGENT = {
    'random': _create_random_agent
}


