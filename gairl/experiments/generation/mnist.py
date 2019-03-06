import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images

from gairl.config import RESOURCES_DIR
from gairl.config import GAN_STR, GAN_VIS_FREQ
from gairl.generators import create_gan


MNIST_DATA_PATH = os.path.join(RESOURCES_DIR, 'mnist_imgs.gz')
TRAINING_STEPS = 1000000
BATCH_SIZE = 100


def main():
    with open(MNIST_DATA_PATH, 'rb') as mnist_file:
        data = extract_images(mnist_file)

    with tf.Session() as sess:
        gan = create_gan(GAN_STR, data.shape[1:], sess,
                         data_ranges=(data.min(), data.max()))
        for t in range(TRAINING_STEPS):
            batch_indices = np.random.randint(data.shape[0], size=BATCH_SIZE)
            batch_data = data[batch_indices, :]
            gan.train_step(batch_data)
            if t % GAN_VIS_FREQ == 0:
                gen_images = gan.generate(4)
                gan.visualize_data(gen_images)

if __name__ == '__main__':
    main()
