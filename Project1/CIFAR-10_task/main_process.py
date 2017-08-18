#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:39:50 2017

@author: Weiyu Lee
"""

import tensorflow as tf
import pickle

import helper
import NN_struct as NNs
import problem_unittests as tests
import download_cifar_10 as DL
import preprocess as preprc

cifar10_dataset_folder_path = "cifar-10-batches-py"
tar_gz_path = "cifar-10-python.tar.gz"

# Download the CIFAR-10 dataset if not exist.
DL.process(cifar10_dataset_folder_path, tar_gz_path)

# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(cifar10_dataset_folder_path, preprc.normalize, preprc.one_hot_encode)

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    session.run(optimizer, feed_dict={
            x:feature_batch,
            y:label_batch,
            keep_prob:keep_probability})  

# Unit Test Function
#tests.test_train_nn(train_neural_network)

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # TODO: Implement Function
    loss = session.run(cost, feed_dict={
        x: feature_batch,
        y: label_batch,
        keep_prob: 1.})
   
    valid_acc = session.run(accuracy, feed_dict={
        x: valid_features,
        y: valid_labels,
        keep_prob: 1.})
    
    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
    loss,
    valid_acc))

##############################
## Build the Neural Network ##
##############################

# Neural Network Parameters 
conv_ksize          = list()
conv_strides        = list()
conv_num_outputs    = list()

pool_ksize          = list()
pool_strides        = list()

fully_outputs       = list()

# CNN Layer 1
conv_ksize.append([3, 3])
conv_strides.append([1, 1, 1, 1])
conv_num_outputs.append(64)
pool_ksize.append([2, 2])
pool_strides.append([1, 2, 2, 1])

# CNN Layer 2
conv_ksize.append([3, 3])
conv_strides.append([1, 1, 1, 1])
conv_num_outputs.append(128)
pool_ksize.append([2, 2])
pool_strides.append([1, 2, 2, 1])

# CNN Layer 3
conv_ksize.append([3, 3])
conv_strides.append([1, 1, 1, 1])
conv_num_outputs.append(256)
pool_ksize.append([2, 2])
pool_strides.append([1, 2, 2, 1])

# DNN Layer 1
fully_outputs.append(2000)

# DNN Layer 2
fully_outputs.append(1200)

# DNN Layer 3
fully_outputs.append(325)

# Final Output Layer
final_out = 10

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = NNs.neural_net_image_input((32, 32, 3))
y = NNs.neural_net_label_input(10)
keep_prob = NNs.neural_net_keep_prob_input()

# Model
print("\n===============================================================================================")
print("========================================Build the Model========================================")
print("===============================================================================================")
conv_out = NNs.conv_net(x, conv_ksize, conv_strides, conv_num_outputs, pool_ksize, pool_strides)
fully_out = NNs.fully_net(conv_out, fully_outputs, final_out)
print("===============================================================================================")
print("===============================================================================================")
print("===============================================================================================\n")

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(fully_out, name='logits')

# Loss and Optimizer
softmax_logits = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
cost = tf.reduce_mean(softmax_logits)
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

# Unit Test Function
#tests.test_conv_net(NNs.conv_net)

##############################
###### Hyperparameters #######
##############################
epochs = 500
batch_size = 256
keep_probability = 0.75

# Train on a Single CIFAR-10 Batch
#print('Checking the Training on a Single Batch...')
#with tf.Session() as sess:
#    # Initializing the variables
#    sess.run(tf.global_variables_initializer())
#    
#    # Training cycle
#    for epoch in range(epochs):
#        batch_i = 1
#        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
#            print(len(batch_features), len(batch_labels))
#            #train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
#        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
#        print_stats(sess, batch_features, batch_labels, cost, accuracy)
        
save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
            
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
        
        



