# DQN for CartPole environment

Purpose of the experiments was to debug implemented DQN algorithm and make
sure it works by fitting it into the simplest OpenAI Gym environment. Using 
simple environment can allow for fast debugging iterations where developer 
does not need to wait long time between runs.

Random agent was used as a baseline. Logs and tensorboard files for the random
agent can be found at `outputs/cartpole/dqn-random/`.

### Convergence to the worst policy
*Agent outputs can be found at `outputs/cartpole/dqn-bef-fix/`*

Agent was not properly updating anything. Average reward per episode, after the
initial exploration stage was converging to 10, what is the worst possible
reward (Random actions produce average reward of 20).

It was caused by the agent sticking to only single action for around 1000 steps 
each. Agent was just moving left or just moving right, making the CartPole 
quickly leave the episode area.

#### Hypothesis I - invalid weights/biases/output structure
I hypothesised from my previous experience that somehow network parameters
are not connected properly to the final network output. It could cause only 
last layer biases influence the actions or make biases the only updateable
parameters. This scenario would make network output almost completely 
independent from the input.

Hypothesis was rejected when checking tensorboard summaries of the network
parameters. All of the weights and biases were properly updated.

#### Hypothesis II - bug in the update graph
Knowing that online and target network outputs are properly wired. The next
thing to explore was to check if the replay buffer, online network,
target network setting for updating the online network is actually wired
correctly.

After cleaning up the tensorflow computational graph I discovered small 
anomaly in the outputs of one node. Namely it was returning element of shape
(None, None) where it was supposed to return element of shape (None,).
It happened when choosing Q values to update based on actions that
were coming from the replay buffer. tf.gather method didn't work as expected.
A few lines of tensorflow code instead were necessary to fix the error.

**Hypothesis confirmed - bug fixed.**


## Agent learns but not much better than random
*Agent outputs can be found at `outputs/cartpole/dqn-after-fix/`*


#### Hypothesis I - Adam optimizer causes the problem
From my experience Adam does not work well with reinforcement learning problems
(investigate why). 

It was replaced by basic gradient descent. It may be slower than more
sophisticated optimizers but for sure will not interfere with any reinforcement
learning specific properties. Change made agent learn properly.

**Hypothesis confirmed - agent starts learning properly.**


## Huge stability problems 
*Agent outputs can be found at `outputs/cartpole/dqn2424gdsc5e-3`*

Unfortunately, every time after learning how to master the environment agent's 
performance was plummeting. It was extremely unstable. However, what's very 
interesting it was highly overestimating the reward it gets. Q values were 
showing impossible to achieve results (average of 300) while receiving average
reward that was not much better than the random agent


#### Hypothesis - batch size is too small (outputs/dqn2424gdsc5e-3batch256)
If data batch is too small then the distribution of data won't be smooth. 
As the agent learns more optimal behaviors, episode length increases and thus 
relative amount of terminal states in the batch decreases. Because in the 
CartPole environment maximum epsiode length is 200 (and previous batch size 
was 32) it can be a potential problem.

Batch size was increased to 256 what allowed agent to quickly learn and 
solve the problem numerous times. After initial learning it stayed with
average reward per episode higher than 150! What's interesting, compared to
previous 'arogant' version of the agent this one was slightly underestimating
its capabilities predicting expected reward to be around 140.

**Hypothesis confirmed - agent stabilizes almost completely.**

Multiple runs of the final agent can be found at:
`outputs/cartpole/dqn2424gdsc5e-3batch256`
`outputs/cartpole/dqn2424gdsc5e-3batch256-2`
`outputs/cartpole/dqn2424gdsc5e-3batch256-3`
`outputs/cartpole/dqn2424gdsc5e-3batch256-saved`
where the last one also consists of a saved network model. Best performing
model was saved under checkpoint 90000