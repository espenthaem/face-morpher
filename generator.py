import numpy as np
import cv2
from keras.utils import Sequence


class celebGenerator(Sequence):
    def __init__(self, imgs, batch_size=128, input_shape=(128, 128, 1), feature_wise_mean=None):
        self.imgs = imgs
        self.batch_size = batch_size
        self.input_shape = input_shape

        self.feature_wise_mean = feature_wise_mean

    def fit(self):

        index = 0
        batch_means = []
        while index * self.batch_size < len(self.imgs):
            batch_images = self.imgs[index * self.batch_size:(index + 1) * self.batch_size]
            batch = []
            for batch_image in batch_images:
                # Load and convert to greyscale
                image = cv2.imread(batch_image, 0)
                # Reshape
                image = cv2.resize(image, self.input_shape[:2])
                # Normalize inputs
                batch.append(image / 255.)
            batch_means.append(np.mean(batch))

        self.feature_wise_mean = np.mean(batch_means)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.imgs) / self.batch_size))

    def __getitem__(self, index):
        batch = []

        batch_images = self.imgs[index * self.batch_size:(index + 1) * self.batch_size]

        for batch_image in batch_images:
            # Load and convert to greyscale
            image = cv2.imread(batch_image, 0)
            # Reshape
            image = cv2.resize(image, self.input_shape[:2])
            # Normalize inputs
            batch.append(image / 127.5 - 1)

        batch = np.array(batch).astype('float32')

        if self.feature_wise_mean is not None:
            batch = batch - self.feature_wise_mean
        else:
            batch = batch - batch.mean()

        batch = batch.reshape(self.batch_size, self.input_shape[0], self.input_shape[1], 1)

        return batch, batch
