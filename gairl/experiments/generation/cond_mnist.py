import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_labels

from gairl.config import RESOURCES_DIR
from gairl.config import GAN_STR
from gairl.generators import create_gan


MNIST_IMGS_PATH = os.path.join(RESOURCES_DIR, 'mnist_imgs.gz')
MNIST_LABELS_PATH = os.path.join(RESOURCES_DIR, 'mnist_labels.gz')
TRAINING_STEPS = 1000000
BATCH_SIZE = 100
NOISE_SIZE = 100


def main():
    with open(MNIST_IMGS_PATH, 'rb') as imgs_file:
        imgs = extract_images(imgs_file)
    with open(MNIST_LABELS_PATH, 'rb') as labels_file:
        labels = extract_labels(labels_file)

    with tf.Session() as sess:
        gan = create_gan(GAN_STR, imgs.shape[1:], NOISE_SIZE,
                         sess, labels_num=10)
        for _ in range(TRAINING_STEPS):
            batch_indices = np.random.randint(imgs.shape[0], size=BATCH_SIZE)
            batch_imgs = imgs[batch_indices, :]
            batch_labels = labels[batch_indices]
            noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_SIZE))
            gan.train_step(batch_imgs, noise, labels=batch_labels)

if __name__ == '__main__':
    main()
