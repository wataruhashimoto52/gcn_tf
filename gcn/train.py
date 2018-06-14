# coding: utf-8

import os 
import sys 
import time 
import networkx as nx 
import scipy.sparse as sparse
import numpy as np
import tensorflow as tf
from gcn.layers import GraphConvLayer


learning_rate = 0.01
training_epochs = 100
batch_size = 100 
display_step = 1


def sparse_to_tuple(sparse_matrix):
    """
    convert sparse matrix to tuple representation.
    """

    def to_tuple(matrix):
        if not sparse.isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        
        coords = np.vstack((matrix.row, matrix.col)).transpose()
        values = matrix.data 
        shape = matrix.shape 
        return coords, values, shape 

    
    if isinstance(sparse_matrix, list):
        for i in range(len(sparse_matrix)):
            sparse_matrix[i] = to_tuple(sparse_matrix[i])
    else:
        sparse_matrix = to_tuple(sparse_matrix)

    
    return sparse_matrix


def training(cost, global_step):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op


def masked_softmax_xentropy(preds, labels, mask):
    loss = tf.nn.softmax_cross_entropy_with_logits(preds, labels)
    mask = tf.cast(loss, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask 
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask 
    reduced_accuracy = tf.reduce_mean(accuracy_all)
    tf.summary.scalar("accuracy", reduced_accuracy)
    return reduced_accuracy


def inference(x):
    with tf.variable_scope("graphconv_1"):
        fc1 = GraphConvLayer()

    with tf.variable_scope("graphconv_2"):
        fc2 = GraphConvLayer()

    with tf.variable_scope("graphconv_3"):
        fc3 = GraphConvLayer()

    with tf.variable_scope("graphconv_3"):
        output = GraphConvLayer(activation=tf.nn.softmax)

    
    return output

    
# tensorflow placeholders
ph = {
    'adj_norm': tf.sparse_placeholder(tf.float32, name="adj_matrix"),
    'x': tf.sparse_placeholder(tf.float32, name="x"),
    'labels': tf.placeholder(tf.float32, shape=(n_nodes, nb_classes)),
    'mask': tf.placeholder(tf.int32)
}


if __name__ == "__main__":
    
    # preprocess graphs


    with tf.Graph().as_default():
        with tf.variable_scope("graph_convolutional_networks"):

            output = inference(ph['x'])
            cost = masked_softmax_xentropy(output, ph['labels'], ph['mask'])
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = training(cost, global_step)
            eval_op = masked_accuracy(output, ph['labels'], ph['mask'])
            summary_op = tf.summary.merge_all()
            saver = tf.train.Saver()
            sess = tf.Session()
            summary_writer = tf.summary.FileWriter(
                "graphconv_logs",
                graph_def=sess.graph_def
            )
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            