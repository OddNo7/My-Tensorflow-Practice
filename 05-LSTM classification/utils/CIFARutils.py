import _pickle as pickle
import os
import tarfile
import glob
import numpy as np
from scipy.ndimage.interpolation import rotate


# This code is a little different from the one in 03-CNN.
# Some utility functions to manipulate cifar-10 dataset.

def load_data(mode='train'):
    """
    # A function to load cifar-10 dataset.
    :param mode: 'train' or 'test'. Specify the training set or test set.
    :return: A dictionary-like dataset.
    """
    if not os.path.exists('./cifar10/cifar-10-batches-py'):
        package = tarfile.open('cifar-10-python.tar.gz')
        package.extractall('./cifar10')
        package.close()
    root_dir = os.getcwd()
    os.chdir('./cifar10/cifar-10-batches-py')
    if mode == 'train':
        datas = glob.glob('data_batch*')
    elif mode == 'test':
        datas = glob.glob('test_batch')
    else:
        raise ValueError('Mode must be ''train'' or ''test''.')
    batches = []
    for name in datas:
        handle = open(name, 'rb')
        dic = pickle.load(handle, encoding='bytes')
        batches.append(dic)
    os.chdir(root_dir)
    return batches


def preprocess_image(raw):
    """
    # A function to split the dataset into data and labels.
    :param raw: Dataset loaded by 'load_data'.
    :return: Data and labels.
    """
    img_size = 32
    num_channels = 3
    data = np.zeros([50000, 32, 32, 3], dtype=np.float32)
    label = np.zeros([50000, ], dtype=np.float32)
    for i in range(len(raw)):
        batch = raw[i]
        batch_data = np.array(batch[b'data'], dtype=np.float32).reshape([-1, num_channels, img_size, img_size])
        data[i * 10000:(i + 1) * 10000, :, :, :] = batch_data.transpose([0, 2, 3, 1])
        label[i * 10000:(i + 1) * 10000, ] = np.array(batch[b'labels'], dtype=np.float32)
    data /= 255.0
    return data, label


class BatchGenerator(object):
    """
    An automatic, never-ending mini-batch generator
    """

    def __init__(self, data, label, batch_size, img_size=32, **kwargs):
        """
        # Initialization of the generator.
        :param data: input data.
        :param label: input labels.
        :param batch_size: mini-batch size.
        :param img_size: the size of input image. Default is square image of size 32.
        :param kwargs: Other parameters.
        """
        self.data = data
        self.label = label
        self.img_size = img_size
        self.batch_size = batch_size
        self.data_size = data.shape[0]
        self.hflip = None
        self.rotate = None
        for key, value in kwargs.items():
            if key == 'hflip':
                self.hflip = value
            if key == 'rotate':
                self.rotate = value

    def initialize(self):
        """
        # Start the generator.
        :return: A mini-batch of data and labels.
        """
        num_batches = self.data_size // self.batch_size
        count = 0
        while True:
            batch_data = self.data[count * self.batch_size:(count + 1) * self.batch_size, :, :]
            batch_label = self.label[count * self.batch_size:(count + 1) * self.batch_size]
            count = (count + 1) % num_batches
            if self.hflip:
                batch_data = np.flip(batch_data, axis=2)
            if self.rotate:
                batch_data = rotate(batch_data, self.rotate,
                                    axes=(1, 2), reshape=False)
            yield batch_data, batch_label
