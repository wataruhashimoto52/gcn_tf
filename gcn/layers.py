# coding: utf-8

import numpy as np 
import tensorflow as tf 

def dot(x, y, sparse=False):
    """
    Wrapper for tf.matmul
    """

    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    
    return res


class GraphConvLayer:
    def __init__(self, input_dim, output_dim, name, 
            activation=tf.nn.relu, bias=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation 
        self.bias = bias 

