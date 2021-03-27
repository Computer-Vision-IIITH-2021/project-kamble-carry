from __future__ import division, print_function

import numpy as np
import tensorflow as tf
from vgg import Vgg19
from PIL import Image
import time
from closedformmatting import getLaplacian
import math
from functools import partial
import copy
import os
import tensorflow_probability as tfp

VGG_MEAN = [103.939, 116.779, 123.68]

def rgb2bgr(rgb, vgg_mean=True):
    if vgg_mean:
        return rgb[:, :, ::-1] - VGG_MEAN
    else:
        return rgb[:, :, ::-1]

def bgr2rgb(bgr, vgg_mean=False):
    if vgg_mean:
        return bgr[:, :, ::-1] + VGG_MEAN
    else:
        return bgr[:, :, ::-1]

def load_seg(content_seg_path, style_seg_path, content_shape, style_shape):
    color_codes = ['BLUE', 'GREEN', 'BLACK', 'WHITE', 'RED', 'YELLOW', 'GREY', 'LIGHT_BLUE', 'PURPLE']
    def _extract_mask(seg, color_str):
        h, w, c = np.shape(seg)
        if color_str == "BLUE":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "GREEN":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "BLACK":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "WHITE":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "RED":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "YELLOW":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "GREY":
            mask_r = np.multiply((seg[:, :, 0] > 0.4).astype(np.uint8),
                                 (seg[:, :, 0] < 0.6).astype(np.uint8))
            mask_g = np.multiply((seg[:, :, 1] > 0.4).astype(np.uint8),
                                 (seg[:, :, 1] < 0.6).astype(np.uint8))
            mask_b = np.multiply((seg[:, :, 2] > 0.4).astype(np.uint8),
                                 (seg[:, :, 2] < 0.6).astype(np.uint8))
        elif color_str == "LIGHT_BLUE":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "PURPLE":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        return np.multiply(np.multiply(mask_r, mask_g), mask_b).astype(np.float32)

    # PIL resize has different order of np.shape
    content_seg = np.array(Image.open(content_seg_path).convert("RGB").resize(content_shape, resample=Image.BILINEAR), dtype=np.float32) / 255.0
    style_seg = np.array(Image.open(style_seg_path).convert("RGB").resize(style_shape, resample=Image.BILINEAR), dtype=np.float32) / 255.0

    color_content_masks = []
    color_style_masks = []
    for i in range(len(color_codes)):
        color_content_masks.append(tf.expand_dims(tf.expand_dims(tf.constant(_extract_mask(content_seg, color_codes[i])), 0), -1))
        color_style_masks.append(tf.expand_dims(tf.expand_dims(tf.constant(_extract_mask(style_seg, color_codes[i])), 0), -1))

    return color_content_masks, color_style_masks

def gram_matrix(activations):
    height = tf.shape(input=activations)[1]
    width = tf.shape(input=activations)[2]
    num_channels = tf.shape(input=activations)[3]
    gram_matrix = tf.transpose(a=activations, perm=[0, 3, 1, 2])
    gram_matrix = tf.reshape(gram_matrix, [num_channels, width * height])
    gram_matrix = tf.matmul(gram_matrix, gram_matrix, transpose_b=True)
    return gram_matrix

def content_loss(const_layer, var_layer, weight):
    return tf.reduce_mean(input_tensor=tf.math.squared_difference(const_layer, var_layer)) * weight

def style_loss(CNN_structure, const_layers, var_layers, content_segs, style_segs, weight):
    loss_styles = []
    layer_count = float(len(const_layers))
    layer_index = 0

    _, content_seg_height, content_seg_width, _ = content_segs[0].get_shape().as_list()
    _, style_seg_height, style_seg_width, _ = style_segs[0].get_shape().as_list()
    for layer_name in CNN_structure:
        layer_name = layer_name[layer_name.find("/") + 1:]

        # downsampling segmentation
        if "pool" in layer_name:
            content_seg_width, content_seg_height = int(math.ceil(content_seg_width / 2)), int(math.ceil(content_seg_height / 2))
            style_seg_width, style_seg_height = int(math.ceil(style_seg_width / 2)), int(math.ceil(style_seg_height / 2))

            for i in range(len(content_segs)):
                content_segs[i] = tf.image.resize(content_segs[i], tf.constant((content_seg_height, content_seg_width)), method=tf.image.ResizeMethod.BILINEAR)
                style_segs[i] = tf.image.resize(style_segs[i], tf.constant((style_seg_height, style_seg_width)), method=tf.image.ResizeMethod.BILINEAR)

        elif "conv" in layer_name:
            for i in range(len(content_segs)):
                # have some differences on border with torch
                content_segs[i] = tf.nn.avg_pool2d(input=tf.pad(tensor=content_segs[i], paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT"), \
                ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')
                style_segs[i] = tf.nn.avg_pool2d(input=tf.pad(tensor=style_segs[i], paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT"), \
                ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')

        if layer_name == var_layers[layer_index].name[var_layers[layer_index].name.find("/") + 1:]:
            print("Setting up style layer: <{}>".format(layer_name))
            const_layer = const_layers[layer_index]
            var_layer = var_layers[layer_index]

            layer_index = layer_index + 1

            layer_style_loss = 0.0
            for content_seg, style_seg in zip(content_segs, style_segs):
                gram_matrix_const = gram_matrix(tf.multiply(const_layer, style_seg))
                style_mask_mean   = tf.reduce_mean(input_tensor=style_seg)
                gram_matrix_const = tf.cond(pred=tf.greater(style_mask_mean, 0.),
                                        true_fn=lambda: gram_matrix_const / (tf.cast(tf.size(input=const_layer), dtype=tf.float32) * style_mask_mean),
                                        false_fn=lambda: gram_matrix_const
                                    )

                gram_matrix_var   = gram_matrix(tf.multiply(var_layer, content_seg))
                content_mask_mean = tf.reduce_mean(input_tensor=content_seg)
                gram_matrix_var   = tf.cond(pred=tf.greater(content_mask_mean, 0.),
                                        true_fn=lambda: gram_matrix_var / (tf.cast(tf.size(input=var_layer), dtype=tf.float32) * content_mask_mean),
                                        false_fn=lambda: gram_matrix_var
                                    )

                diff_style_sum    = tf.reduce_mean(input_tensor=tf.math.squared_difference(gram_matrix_const, gram_matrix_var)) * content_mask_mean

                layer_style_loss += diff_style_sum

            loss_styles.append(layer_style_loss * weight)
    return loss_styles

def total_variation_loss(output, weight):
    tv_loss = tf.reduce_sum(input_tensor=(output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) * (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) + \
              (output[:, :-1, :-1, :] - output[:, 1:, :-1, :]) * (output[:, :-1, :-1, :] - output[:, 1:, :-1, :])) / 2.0
    return tv_loss * weight

def affine_loss(output, M, weight):
    loss_affine = 0.0
    output_t = output / 255.
    for Vc in tf.unstack(output_t, axis=-1):
        Vc_ravel = tf.reshape(tf.transpose(a=Vc), [-1])
        loss_affine += tf.matmul(tf.expand_dims(Vc_ravel, 0), tf.sparse.sparse_dense_matmul(M, tf.expand_dims(Vc_ravel, -1)))

    return loss_affine * weight

def save_result(img_, str_):
    result = Image.fromarray(np.uint8(np.clip(img_, 0, 255.0)))
    result.save(str_)

iter_count = 0
min_loss, best_image = float("inf"), None

def print_loss(args, loss_content, loss_styles_list, loss_tv, loss_affine, overall_loss, output_image):
    global iter_count, min_loss, best_image
    if iter_count % args.print_iter == 0:
        print('Iteration {} / {}\n\tContent loss: {}'.format(iter_count, args.max_iter, loss_content))
        for j, style_loss in enumerate(loss_styles_list):
            print('\tStyle {} loss: {}'.format(j + 1, style_loss))
        print('\tTV loss: {}'.format(loss_tv))
        print('\tAffine loss: {}'.format(loss_affine))
        print('\tTotal loss: {}'.format(overall_loss - loss_affine))

    if overall_loss < min_loss:
        min_loss, best_image = overall_loss, output_image

    if iter_count % 100 == 0 and iter_count != 0:
        save_result(best_image[:, :, ::-1], os.path.join('./', 'out_iter_{}.png'.format(iter_count)))

    iter_count += 1
