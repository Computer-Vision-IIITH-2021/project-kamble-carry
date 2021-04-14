import os
import tensorflow as tf
import numpy as np
import time
import inspect

class Vgg19:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
