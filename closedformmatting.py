from __future__ import division
import argparse
import os
import scipy.misc as spm
import scipy.ndimage as spi
import scipy.sparse as sps
import numpy as np
import tensorflow as tf


def getLaplacian(img):
    h, w, _ = img.shape
    coo = getlaplacian1(img, np.zeros(shape=(h, w)), 1e-5, 1).tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)