#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:21:56 2017

@author: Weiyu Lee

This project is based on the "MNIST For ML Beginners" (https://www.tensorflow.org/get_started/mnist/beginners).
"""

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def fully_DNN_conn(input_tensor, num_neuron):
    """
    Apply a fully connected layer to input_tensor using weight and bias
    : input_tensor: A 2-D tensor where the first dimension is batch size.
    : num_neuron: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    batch_size, tensor_size = input_tensor.get_shape().as_list()
    
    weight_fc = tf.Variable(tf.truncated_normal([tensor_size, num_neuron], mean=0, stddev=0.05))
    bias_fc = tf.Variable(tf.zeros([num_neuron]))
    
    input_tensor = tf.add(tf.matmul(input_tensor, weight_fc), bias_fc)
    
    return tf.nn.relu(input_tensor)

def output(input_tensor, num_outputs):
    """
    Apply a output layer to input_tensor using weight and bias
    : input_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    batch_size, tensor_size = input_tensor.get_shape().as_list()
    
    weight_out = tf.Variable(tf.truncated_normal([tensor_size, num_outputs], mean=0, stddev=0.05))
    bias_out = tf.Variable(tf.zeros([num_outputs]))
    
    output_tensor = tf.add(tf.matmul(input_tensor, weight_out), bias_out)
    
    return output_tensor


def fully_DNN_net(input_tensor, num_layers_neuron, num_out_layer):
    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    fully_num_layers = len(num_layers_neuron)

    # Assign x as the 0 layer's output (1st cnn layer's input)
    DNN_out = list()
    
    for i in range(fully_num_layers):
        if i == 0:
            DNN_input = input_tensor
        else:
            DNN_input = DNN_out[i-1]

        DNN_out.append(fully_DNN_conn(DNN_input, num_layers_neuron[i]))      
        _, DNN_input_num = DNN_input.get_shape().as_list()

        print("DNN Layer%d: Input size = %d Output depth = %d\n" % (i+1, DNN_input_num, num_layers_neuron[i]))      
    
    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    sys_output = output(DNN_out[-1], num_out_layer)
    print("Final Output Layer: Input size = %d Output depth = %d\n" % (num_layers_neuron[-1], num_out_layer))      
    
    # TODO: return output
    return sys_output

def main(_):
    input_dim = 784
    
    ##############################
    ###### Hyperparameters #######
    ##############################    
    learn_rate = 0.5
    batch_size = 128
    num_step = 1000
    
    num_layers_neuron = [64, 32]
    num_outputs = 10
    
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    
    # Create the model
    x = tf.placeholder(tf.float32, [None, input_dim])
    y = tf.placeholder(tf.float32, [None, num_outputs])
      
    #  W = tf.Variable(tf.zeros([784, 10]))
    #  b = tf.Variable(tf.zeros([10]))
    #  y = tf.matmul(x, W) + b
    
    # Model
    print("\n===============================================================================================")
    print("========================================Build the Model========================================")
    print("===============================================================================================")
    fully_out = fully_DNN_net(x, num_layers_neuron, num_outputs)
    print("===============================================================================================")
    print("===============================================================================================")
    print("===============================================================================================\n")    

    # Define loss and optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fully_out, labels=y))
    train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    # Train
    for _ in range(num_step):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, 
                 feed_dict={x: batch_xs, 
                            y: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(fully_out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    print("Test accuracy:", sess.run(accuracy, 
                                     feed_dict={x: mnist.test.images, 
                                                y: mnist.test.labels}))
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', 
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    
    
    