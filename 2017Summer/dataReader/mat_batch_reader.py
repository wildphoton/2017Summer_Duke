import numpy as np
import scipy.misc as misc
import scipy.io
import os


class BatchReader:
    images = []
    annotations = []
    labels = []
    mean_image = []

    batch_offset = 0
    epochs_completed = 0

    def __init__(self, matDir, num_matFiles):
        if not os.path.exists(matDir):
            raise IOError("Can not find data directory")
        self.num_matFiles = num_matFiles
        self.read_matFiles(matDir)

    def read_matFiles(self, matDir):
        for matfile in ["chunk_{:0>5}".format(i) for i in self.num_matFiles]:
            print("Loadding mat file: {}".format(matfile))
            temp_data = scipy.io.loadmat(os.path.join(matDir, matfile))
            if self.mean_image == []:
                self.mean_image = temp_data['dsTemp'][0][0][1]

            temp_patches = temp_data['dsTemp'][0][0][0][0][0][0]
            temp_truth = temp_data['dsTemp'][0][0][0][0][0][2]
            temp_labels = temp_data['dsTemp'][0][0][0][0][0][1][0]

            if self.images == []:
                self.images = temp_patches
                self.annotations = temp_truth
                self.labels = temp_labels
            else:
                self.images = np.concatenate((self.images, temp_patches), axis=-1)
                self.annotations = np.concatenate((self.annotations, temp_truth), axis=-1)
                self.labels = np.concatenate((self.labels, temp_labels), axis=-1)

        self.images = np.transpose(self.images, (3, 0, 1, 2))
        self.annotations = np.transpose(self.annotations, (3, 0, 1, 2))
        print(self.images.shape)
        print(self.annotations.shape)

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            self.shuffle_images()
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indices = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indices], self.annotations[indices]

    def get_batch_by_indices(self, indices):
        return self.images[indices], self.annotations[indices]

    def shuffle_images(self):
        perm = np.arange(self.images.shape[0])
        np.random.shuffle(perm)
        self.images = self.images[perm]
        self.annotations = self.annotations[perm]