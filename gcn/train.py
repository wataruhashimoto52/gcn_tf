# coding: utf-8

import os 
import sys 
import time 
import numpy as np
import tensorflow as tf


learning_rate = 0.01
training_epochs = 100
batch_size = 100 
display_step = 1


def training(cost, global_step):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op


def masked_softmax_xentropy(preds, y, mask):
    loss = tf.nn.softmax_cross_entropy_with_logits(preds, y)
    mask = tf.cast(loss, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask 
    return tf.reduce_mean(loss)


def inference(x):
    pass
