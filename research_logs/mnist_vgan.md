# Vaniilla GAN for MNIST dataset

Purpose of the experiments was to debug implemented GAN architecture and make
sure it works by fitting it into simples MNIST dataset. Using simple 
environment can allow for fast debugging iterations where developer does not 
need to wait long time between runs.

### Discriminator completely outperforms Generator
*Agent outputs can be found at 
`outputs/generation/mnist/vgan/default`
`outputs/generation/mnist/vgan/k2`*

As soon as training starts, Discriminator learns to properly discriminate 
between samples. It continues forever to completely outperform Generator
making it stuck at generated data that is not much better than the random
noise due to vanishing gradient.

#### Hypothesis I - k too high

I hypothesised that Discriminator performs just to many iterations for each
of the Generator iterations. Therefore I decided to decrease this value
to `k=1`.

It changed the power dynamics between the networks. Depending on random seed
sometimes Discriminator was the one dominating but sometimes also Generator
was the one beating up the other network. Discriminator stopped dominating
at every run.

**Hypothesis confirmed - Discriminator stopped always outperforming Generator**

## Always one network outperforms the other
*Agent outputs can be found at 
`outputs/generation/mnist/vgan/k1`*

Architecture started producing models where sometimes Discriminator was
outperforming Generator but sometimes also Generator was outperforming
Discriminator. No stable training yet unfortunately

#### Hypothesis I - initial parameters are too big
From my reinforcement learning experience I hypothesised that too high
initial parameters for the network cause huge initial instability between
the networks and huge instability of their gradients not allowing them to
find appropriate training equilibrium.

When decreasing size of the initial distribution of parameters architecture 
managed to get more stable and with time generate realistic samples of MNIST
data.

**Hypothesis confirmed - VGAN solves MNIST dataset**

Runs of the final agent can be found at:
`outputs/generation/mnist/vgan/k1adam2e-4,adam2e-4smallinit2`
`outputs/generation/mnist/vgan/k1adam2e-4,adam2e-4smallinit3`