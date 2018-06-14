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

        with tf.variable_scope(name):
            with tf.name_scope("weight"):
                self.w = tf.get_variable(name="W", 
                        shape=(self.input_dim, self.output_dim),
                        initializer=tf.contrib.layers.xavier_initializer())
                
            if self.bias:
                with tf.name_scope("bias"):
                    self.b = tf.get_variable(name="b",
                            initializer=tf.constant(0.1, shape=(self.output_dim, )))


    def forward(self, adj_norm, x, sparse=False):
        hw = dot(x, self.w, sparse=sparse)
        ahw = dot(adj_norm, hw, sparse=True)
        if not self.bias:
            return self.activation(ahw)
        
        return self.activation(tf.nn.bias_add(ahw, self.bias))
