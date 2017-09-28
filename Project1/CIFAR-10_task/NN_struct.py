#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:34:11 2017

@author: Weiyu Lee
"""

import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    height, width, depth = image_shape       
    x = tf.placeholder(tf.float32, [None, height * width * depth], 'x')
    
    return x

def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    y = tf.placeholder(tf.float32, [None, n_classes], 'y')
    
    return y

def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return keep_prob

def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    batch_size, tensor_size = x_tensor.get_shape().as_list()
    
    weight_fc = tf.Variable(tf.random_normal([tensor_size, num_outputs], mean=0, stddev=0.01))
    bias_fc = tf.Variable(tf.zeros([num_outputs]))
    
    x_tensor = tf.add(tf.matmul(x_tensor, weight_fc), bias_fc)
    
    return tf.nn.relu(x_tensor)

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    batch_size, tensor_size = x_tensor.get_shape().as_list()
    
    weight_out = tf.Variable(tf.random_normal([tensor_size, num_outputs], mean=0, stddev=0.01))
    bias_out = tf.Variable(tf.zeros([num_outputs]))
    
    x_tensor = tf.add(tf.matmul(x_tensor, weight_out), bias_out)
    
    return x_tensor

def fully_net(x, fully_outputs, final_out):
    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    fully_num_layers = len(fully_outputs)

    # Assign x as the 0 layer's output (1st cnn layer's input)
    DNN_out = list()
    
    for i in range(fully_num_layers):
        if i == 0:
            DNN_input = x
        else:
            DNN_input = DNN_out[i-1]

        DNN_out.append(fully_conn(DNN_input, fully_outputs[i]))      
        _, DNN_input_num = DNN_input.get_shape().as_list()

        print("DNN Layer%d: Input size = %d Output depth = %d\n" % (i+1, DNN_input_num, fully_outputs[i]))      
    
    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    sys_output = output(DNN_out[-1], final_out)
    print("Final Output Layer: Input size = %d Output depth = %d\n" % (fully_outputs[-1], final_out))      
    
    # TODO: return output
    return sys_output

