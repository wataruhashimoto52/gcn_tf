# coding: utf-8

import networkx as nx 
import scipy.sparse as sparse
import numpy as np
import tensorflow as tf
from layers import GraphConvLayer
from tensorflow.python import debug as tf_debug


learning_rate = 0.001
epochs = 500
save_every = 50


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


def training(cost):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cost)
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
        fc1 = GraphConvLayer(input_dim=feat_x.shape[-1],
                            output_dim=l_sizes[0],
                            name="fc1",
                            activation=tf.nn.tanh)(adj_norm=ph['adj_norm'],x=ph['x'], sparse=True)

    with tf.variable_scope("graphconv_2"):
        fc2 = GraphConvLayer(input_dim=l_sizes[0],
                            output_dim=l_sizes[1],
                            name="fc2",
                            activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=fc1)

    with tf.variable_scope("graphconv_3"):
        fc3 = GraphConvLayer(input_dim=l_sizes[1],
                            output_dim=l_sizes[2],
                            name="fc3",
                            activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=fc2)

    with tf.variable_scope("graphconv_3"):
        output = GraphConvLayer(input_dim=l_sizes[2],
                            output_dim=l_sizes[3],
                            name="fc3",
                            activation=tf.nn.softmax)(adj_norm=ph['adj_norm'], x=fc3)

    
    return output


if __name__ == "__main__":
    

    # preprocess graphs
    g = nx.read_graphml("data/karate.graphml")
    adj = nx.adj_matrix(g)
    n_nodes = adj.shape[0]

    adj_tilde = adj + np.identity(adj.shape[0])
    d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))
    d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -0.5)
    d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
    adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)
    adj_norm_tuple = sparse_to_tuple(sparse.coo_matrix(adj_norm))

    feat_x = np.identity(adj.shape[0])
    feat_x_tuple = sparse_to_tuple(sparse.coo_matrix(feat_x))

    # semi-supervised
    memberships = [m - 1 for m in nx.get_node_attributes(g, 'membership').values()]

    nb_classes = len(set(memberships))
    targets = np.array([memberships], dtype=np.int32).reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]

    labels_to_keep = [np.random.choice(
        np.nonzero(one_hot_targets[:, c])[0]) for c in range(nb_classes)]

    y_train = np.zeros(shape=one_hot_targets.shape,dtype=np.float32)
    y_val = one_hot_targets.copy()

    train_mask = np.zeros(shape=(n_nodes, ), dtype=np.bool)
    val_mask = np.ones(shape=(n_nodes, ), dtype=np.bool)

    l_sizes = [4, 4, 2, nb_classes]

    for l in labels_to_keep:
        y_train[l, :] = one_hot_targets[l, :]
        y_val[l, :] = np.zeros(shape=(nb_classes,))
        train_mask[l] = True 
        val_mask[l] = False

    ph = {
        'adj_norm': tf.sparse_placeholder(tf.float32, name="adj_matrix"),
        'x': tf.sparse_placeholder(tf.float32, name="x"),
        'labels': tf.placeholder(tf.float32, shape=(n_nodes, nb_classes)),
        'mask': tf.placeholder(tf.int32)
    }

    feed_dict_train = {
        ph['adj_norm'] : adj_norm_tuple,
        ph['x'] : feat_x_tuple,
        ph['labels'] : y_train,
        ph['mask'] : train_mask
    }

    feed_dict_val = {
        ph['adj_norm'] : adj_norm_tuple,
        ph['x'] : feat_x_tuple,
        ph['labels'] : y_val,
        ph['mask'] : val_mask
    }

    print(ph['x'])
    print(ph['adj_norm'])


    with tf.Graph().as_default():
        with tf.variable_scope("graph_convolutional_networks"):

            output = inference(ph['x'])
            cost = masked_softmax_xentropy(output, ph['labels'], ph['mask'])
            train_op = training(cost)
            eval_op = masked_accuracy(output, ph['labels'], ph['mask'])
            summary_op = tf.summary.merge_all()
            saver = tf.train.Saver()
            sess = tf.Session()
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            summary_writer = tf.summary.FileWriter(
                "graphconv_logs",
                graph_def=sess.graph_def
            )
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            outputs = {}

            for epoch in range(epochs):
                # fit training
                sess.run(train_op, feed_dict=feed_dict_train)

                # compute loss
                train_loss = sess.run(masked_softmax_xentropy, feed_dict=feed_dict_train)

                # compute accuracy
                train_acc = sess.run(masked_accuracy, feed_dict=feed_dict_train)

                if epoch % save_every == 0:
                    
                    val_loss, val_acc = sess.run([masked_softmax_xentropy,masked_accuracy],
                                            feed_dict=feed_dict_val)

                    # print results
                    print("Epoch:", "%04d" % (epoch + 1),
                          "train_loss=", "{:.5f}".format(train_loss),
                          "train_acc=", "{:.5f}".format(train_acc),
                          "val_loss=", "{:.5f}".format(val_loss),
                          "val_acc", "{:.5f}".format(val_acc))

                    feed_dict_output = {
                        ph['adj_norm'] : adj_norm_tuple,
                        ph['x'] : feat_x_tuple
                    }

                    output = sess.run(inference, feed_dict=feed_dict_output)

                    outputs[epoch] = output
