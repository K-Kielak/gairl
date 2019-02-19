# Rainbow DQN for LunarLander environment

Purpose of the experiments was to improve results of DQN algorithm 
on the LunarLander environment using prioritized experience replay.

Random agent was used as a baseline. Logs and tensorboard files for the random
agent can be found at `outputs/lunarlander/random/`.

### Agent does not solve the environment and collapses after a while
*Agent outputs can be found at `outputs/lunarlander/pdqn/pdqn-6464rmsp5e-3&mem100000target5000/`*

Agent learns how to land properly but does it too slowly. Usually, after it 
lands it does not turn off the engine making it stand on the ground and lose
fuel. Additionally, after the peak behaviour (~470k steps) agent starts
performin worse until it collapses.

#### Hypothesis I - beta annealing to short
*Agent outputs can be found at `outputs/lunarlander/pdqn/pdqn-6464rmsp5e-3&mem100000target5000-2betaper200000/`*

Agent's performance starts decreasing at the same time as beta annealing
finished. Therefore, I hypothesised that it may be a direct cause.

Beta annealing period was doubled. It did not improve performance at all.

**Hypothesis rejected**

#### Hypothesis II - agent underperforms due to overestimation bias


Agent extremely overestimates its performance and thus prefers to hover
over the surface instead of landing (it thinks every step provides
positive reward) and it disregards high reward from landing when learning
(landing reward is not that much higher than hovering + next Q value).
Therefore I hypothesised agent with smaller bias will perform better.

Double DQN agent was implemented and tested on the environment, it did not
improve results.

**Hypothesis rejected**