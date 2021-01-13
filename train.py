from argparse import ArgumentParser
import os
import tensorflow as tf
#import tensorflow.compat.v1 as tf
import numpy as np
import time
from tqdm import tqdm
import shutil
import logging
import argparse
from importlib.machinery import SourceFileLoader
from mpunet.preprocessing.data_loader import get_train_generators
from mpunet.models.probabilistic_unet import ProbUNet
import mpunet.utils.training_utils as training_utils
tf.compat.v1.disable_eager_execution()

def train1(cf):
    """Perform training from scratch."""
    # do not use all gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = cf.cuda_visible_devices

    # initialize data providers
    data_provider = get_train_generators(cf)
    train_provider = data_provider['train']
    val_provider = data_provider['val']

    prob_unet = ProbUNet(latent_dim=cf.latent_dim, num_channels=cf.num_channels,
                         num_1x1_convs=cf.num_1x1_convs,
                         num_classes=cf.num_classes, num_convs_per_block=cf.num_convs_per_block,
                         initializers={'w': training_utils.he_normal(),
                                       'b': tf.compat.v1.truncated_normal_initializer(stddev=0.001)},
                         regularizers={'w': tf.keras.regularizers.l2(0.5 * (1.0))})

    x = tf.compat.v1.placeholder(tf.float32, shape=cf.network_input_shape)
    y = tf.compat.v1.placeholder(tf.uint8, shape=cf.label_shape)
    mask = tf.compat.v1.placeholder(tf.uint8, shape=cf.loss_mask_shape)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    if cf.learning_rate_schedule == 'piecewise_constant':
        learning_rate = tf.compat.v1.train.piecewise_constant(x=global_step, **cf.learning_rate_kwargs)
    else:
        learning_rate = tf.compat.v1.train.exponential_decay(learning_rate=cf.initial_learning_rate, global_step=global_step,
                                                   **cf.learning_rate_kwargs)
    with tf.device(cf.gpu_device):
        prob_unet(x, y, is_training=True, one_hot_labels=cf.one_hot_labels)
        prob_unet._build(x, y, is_training=True, one_hot_labels=cf.one_hot_labels)
        elbo = prob_unet.elbo(y, reconstruct_posterior_mean=cf.use_posterior_mean, beta=cf.beta, loss_mask=mask,
                              analytic_kl=cf.analytic_kl, one_hot_labels=cf.one_hot_labels)
        reconstructed_logits = prob_unet._rec_logits
        sampled_logits = prob_unet.sample()
        reg_loss = cf.regularizarion_weight * tf.reduce_sum(input_tensor=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
        loss = -elbo + reg_loss
        rec_loss = prob_unet._rec_loss_mean
        kl = prob_unet._kl
        mean_val_rec_loss = tf.compat.v1.placeholder(tf.float32, shape=(), name="mean_val_rec_loss")
        mean_val_kl = tf.compat.v1.placeholder(tf.float32, shape=(), name="mean_val_kl")

