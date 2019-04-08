# Generative Adversarial Imagination for Sample Efficient Deep Reinforcement Learning

Repository for the final year research project aiming to improve current deep
reinforcement learning state-of-the-art in terms of sample efficiency.

### Structure of the repository
The project is structured as follows:
 - **gairl** is the main source code directory.
 - **resources** contains all datasets that are required to run the experiments 
 successfully.
 - **tests** includes unit tests for the most critical and bug-prone parts
 of the code.
 - **requirements.txt** specifies Python libraries and their versions necessary 
 to run the code.

The main, functional source code is split into multiple sub-modules:
 - **agents** consists of 4 different types of reinforcement learning agents:
 random, DQN, Rainbow DQN, and GAIRL. Each algorithm, excluding random, comes
 with its own configuration file where all (hyper)parameters are defined.
 - **experiments** provides a set of possible experiments that can be run to
 evaluate algorithms. It includes both straight generative settings, as well 
 as traditional reinforcement learning environments.
 - **generators** includes 4 different types of generative models: multilayer 
 perceptron, GAN (Vanilla GAN in the code), Wasserstein GAN, and Wasserstein 
 GAN with Gradient Penalty. Each model comes with its own configuration file
 where all (hyper)parameters are defined.
 - **memory** consists of different types of memory modules that are necessary
 for both GAIRL and DQN algorithms.
 - **config.py** provides a very general configuration file for the software.
 - **neural_utils.py** comes with multiple useful helper methods for the
 development of neural processing systems in tensorflow.
 
To run the experiment:
1. Specify algorithms that will be used in the **config.py** file.
2. Customise algorithm's parameters in its own configuration file.
3. Run the chosen experiment.

Note: GAIRL framework is the only algorithm that is not self-sustained in its
own config file. Only the topology and dropout of algorithms that are chosen
for the MFM and IM modules can be defined there. More detailed parameters need
to be directly defined in the config files of employed algorithms.

### Requirements
 - Python>=3.6
 - swig
 - python-opengl
 - All python libraries as specified in the **requirements.txt** file