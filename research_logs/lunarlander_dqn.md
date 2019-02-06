# DQN for LunarLander environment

Purpose of the experiments was to use DQN algorithm to solve more complex
OpenAI Gym environment. Experiments started from the DQN that was able to solve
all of the classic control environment. Then the DQN was adapted to the new
more complex situation.

Random agent was used as a baseline. Logs and tensorboard files for the random
agent can be found at `outputs/lunarlander/random/`.

### Agent worse then random because it learns to fly
*Agent outputs can be found at `outputs/lunarlander/dqn/2424gdsc5e-3batch256/`*

Agent performed worse then random. What was interesting, it is not because it
started crashing harder and faster but because it learnt to fly. It was flying
to long in the air losing fuel before the end of the episode. This was
generating such a high negative reward that even proper landing afterwards
was no able to increase the episode reward over average random agent reward.

#### Hypothesis I - memory too short
*Agent outputs can be found at `outputs/lunarlander/dqn/2424gdsc5e-3mem100000/`*

Because episodes in this environment can be infinitely long, I hypothesised
that once the agent learned to fly it was spending too much time in the air
causing replay buffer to be completely overflown with flying samples. The only
thing it knew is that flying produces rewards much less negative (when looking
at a single timstep) than crashing and thus tried to fly indefinitely ending 
up with rewards much worse than quick crash.

Increasing memory size made agent to outperform the default agent and for
a second event outperform random agent (though still getting cumulative reward
of ~-150). However after 500k steps it became completely unstable and its
performance plummeted.

**Hypothesis semi-confirmed - slight improvement**

#### Hypothesis II - network to small
*Agent outputs can be found at `outputs/lunarlander/dqn/6464gdsc5e-3eps&mem100000/`*

Network for the classic control problem consisted of only 2 hidden layers, 
24 nodes each. I hypothesised size of the network may be too small to properly
catch complex dependencies of the environment. Therefore network size was
changed to 2 hidden layers, 64 nodes each. Additionally, I increased epsilon
period 10-fold so new memory can be field with more exploratory samples and to 
prevent bigger network from overfitting exploitative policy.

Increasing network size allowed agent to, for the first time, finish a few
episodes with reward higher than 200 solving the environment! However, just
after it achieved that, the network completely lost stability and its rewards
plummeted to the record low rewards.

**Hypothesis semi-confirmed - it learnt but very unstable**

### Agent learns optimal policy but is extremely unstable
*Agent outputs can be found at `outputs/lunarlander/dqn-6464gdsc5e-3eps&mem100000`*

Once agent discovers optimal behaviour and its rewards quickly
increase, loss skyrockets causing whole network to destabilise and perform
extremely bad.

#### Hypothesis I - target update to frequent
*Agent outputs can be found at:
`outputs/lunarlander/dqn/6464rmsp5e-3eps&mem100000target5000/`
`outputs/lunarlander/dqn/6464rmps5e-3eps&mem100000target5000-2/`
`outputs/lunarlander/dqn/6464rmps5e-3eps&mem100000target10000/`*

I hypothesized that this is just caused by a standard lack of stability of
deep reinforcement learning algorithms. To circumvent it, I decided to increase
time between target network updates so the bellman update is less recursive trying
both 5000 and 10000 target update frequency. Additionally, I changed optimizer
from basic GradientDescent to RMSProp to improve learning speed of the networks.

The agent indeed became much more stable. It even manage to maintain positive
average reward per episode for over 180k steps in one run. However it always 
plummeted again afterwards. The slower the update frequency is, the more
stable the agent but also achieves lower maximum reward.

**Hypothesis confirmed - agent became much more stable though not enough**

#### Hypothesis II - RMSProp made stable configuration unstable
*Agent outputs can be found at: `outputs/lunarlander/dqn/6464gdsc5e-3eps&mem100000target5000/`*

Although increasing target update frequency highly stabilised the network,
it still was collapsing after some time of good policy. From previous
experience I noticed that optimizers with momentum rater interfere with 
reinforcement learning stability than help. Therefore I hypothesised that the
final causes of lack of stability were caused by using RMSProp optimizer
instead of standard GradientDescent.

Unfortunately, most probably RMSProp was really what improved performance of 
the network. Network using Gradient Descent performs terribly. That led to
another hypothesis below.

**Hypothesis completely rejected - RMSProp was the change that made it more stable**


#### Hypothesis III - RMSProp made confguration more stable, target update frequency had nothing to do with that
*Agent outputs can be found at: 
`outputs/lunarlander/dqn/6464rmsp5e-3eps&mem5000`
`outputs/lunarlander/dqn/6464rmsp5e-3eps&mem5000-2`*

Assumption in `Hypothesis I` was that higher higher target update improved
stability of the network. However, rejection of `Hypothesis II` showed that
network without RMSProp optimizer (that was introduced in `Hypothesis I`
together with the slower target update frequency) performs very badly and
indeed it could have been RMSProp that highly stabilised the network instead
of target update frequency.

Surprisingly when target update frequency was increase back to the previous 
level but keeping the RMSProp optimizer, the agent did not become necessarily 
less stable than the same agent with slower frequency. However, Q values with 
frequent target update skyrocketed causing high overestimation. Therefore agent
started perceiving hoovering just over landing zone as a better action than 
actually landing there.

**Hypothesis semi-confirmed - target update freuqency does not improve stability but improves network learning capabilities**

### Agent can learn optimal policy very unstably or be stable but not much better than random
*Agent outputs can be found at: 
`outputs/lunarlander/dqn/6464rmsp5e-3eps&mem100000target5000/`
`outputs/lunarlander/dqn/6464rmps5e-3eps&mem100000target10000/`*

There are a few well performing agents that trade of learnability of the agent
with its stability. Agent can either become stable but have difficulties with
learning optimal policy, or can be able to solve the environment but as soon
as he does it go extremely down with its performance.

#### Hypothesis I - network size is still not enough
*Agent outputs can be found at: 
`outputs/lunarlander/dqn/128128rmsp5e-3eps&mem100000target5000/`*

Based on the qualitative assessment of the agent's behaviour I hypothesized 
that the network still cannot catch complex states of the environment and thus
when more complex situations arise it goes completely crazy with its loss.
Therefore, the number of nodes at each hidden layer was changed from 64 to 128.

It did not improve agent's performance at all.

**Hypothesis rejected**

### Due to lack of progress experiments were halted until RainbowDQN is implemented