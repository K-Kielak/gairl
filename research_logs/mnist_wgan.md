# Wasserstein GAN for MNIST dataset

Purpose of the experiments was to debug implemented GAN architecture and make
sure it works by fitting it into simples MNIST dataset. Using simple 
environment can allow for fast debugging iterations where developer does not 
need to wait long time between runs.

### Architecture does not converge at realistic images
*Agent outputs can be found at 
`outputs/generation/mnist/wgan/k1`
`outputs/generation/mnist/wgan/k1-2`*

Network starts learning something that is better than the random noise quickly
but loss never converges and images do not resemble MNIST digits.

#### Hypothesis I - k too small
*Agent outputs can be found at 
`outputs/generation/mnist/wgan/k5`
`outputs/generation/mnist/wgan/k5rmsp5e-4,rmsp5e-4`*

I hypothesised that Discriminator is not given a chance of properly estimating
the earth-mover distance because Generator is moving to quickly. Because
Discriminator converging at every Generator step is expected in the
Wasserstein GAN setting (not as in Vanilla GAN) I decided to increase `k` to
`k=5`.

It allowed network to stably converged with the earth-mover distance and
produce results that somewhat resembled MNIST digits. Though still lacking
a lot of quality to call results successful

**Hypothesis semi-confirmed - quality improved, loss converged**

#### Hypothesis II - parameters clipping to restrictive

I hypothesises that once the Wasserstein loss fully converges, Critic lacks
expression power to discriminate between better looking generated samples
and real samples due to too restrictive parameters clipping.

This I increased the range of possible parameters to `(-0.05, 0.05)`

### Architecture converges but still lacks crisp high quality results

## Always one network outperforms the other
*Agent outputs can be found at 
`outputs/generation/mnist/vgan/k1`*

Architecture started producing models where sometimes Discriminator was
outperforming Generator but sometimes also Generator was outperforming
Discriminator. No stable training yet unfortunately

#### Hypothesis I - wrong clipping parameters

I hypothesised that the problem are not properly tuned clipping parameters.
Paper on Improved Training of WGANs (https://arxiv.org/pdf/1704.00028.pdf)
proposed to use gradient clipping instead. Therefore I moved my research into
that area.

Summary of experiments with WGANGP
WGAN had high problems with convergence. It was either converging too fast
without generating realistic images, not converging well enough, or being not
stable enough.

First 2 problems were caused by Critic that was too weak compared to
the Generator. I realized increasing power of Critic always solves these
problems. Critic has too fully converge to the Wasserstein distance and 
overpower Generator to work properly. 

Last problem was caused by to high learning rate.

As per https://arxiv.org/pdf/1801.04406.pdf, WGAN with proper weights clipping
(or penalty gradient) always converges given strong enough Critic and low
enough learning rate.