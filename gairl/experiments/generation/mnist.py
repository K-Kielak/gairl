import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images

from gairl.config import RESOURCES_DIR
from gairl.experiments.generation.config import GAN_STR
from gairl.generators import create_gan


MNIST_DATA_PATH = os.path.join(RESOURCES_DIR, 'mnist.gz')
TRAINING_STEPS = 100000
BATCH_SIZE = 100
NOISE_SIZE = 100


def main():
    with open(MNIST_DATA_PATH, 'rb') as mnist_file:
        data = extract_images(mnist_file)

    with tf.Session() as sess:
        gan = create_gan(GAN_STR, data.shape[1:], NOISE_SIZE, sess)
        for _ in range(TRAINING_STEPS):
            batch_indices = np.random.randint(data.shape[0], size=BATCH_SIZE)
            batch_data = data[batch_indices, :]
            noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_SIZE))
            gan.train_step(batch_data, noise)

if __name__ == '__main__':
    main()