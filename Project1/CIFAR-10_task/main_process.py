#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:39:50 2017

@author: Weiyu Lee
"""

import tensorflow as tf
import pickle
import time

import helper
import NN_struct as NNs
import download_cifar_10 as DL
import preprocess as preprc

import os
import random

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

def test_neural_network(session, feature_batch, label_batch, cost, accuracy):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    test_loss, test_acc = session.run([cost, accuracy], 
                                      feed_dict={
                                                  x:feature_batch,
                                                  y:label_batch,
                                                  keep_prob:1.
                                                })      
    print('Test Loss & Acc.: [{:.6f}, {:.6f}]'.format(test_loss, test_acc))

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
    train_loss, train_acc = session.run([cost, accuracy], feed_dict={
            x: feature_batch,
            y: label_batch,
            keep_prob: 1.})
   
    valid_loss, valid_acc = session.run([cost, accuracy], feed_dict={
            x: valid_features,
            y: valid_labels,
            keep_prob: 1.})
    
    print('Training Loss & Acc.: [{:.6f}, {:.6f}] | Validation Loss & Acc.: [{:.6f}, {:.6f}]'.format(
    train_loss, train_acc, 
    valid_loss, valid_acc))
    
    return train_loss, train_acc, valid_loss, valid_acc

cifar10_dataset_folder_path = "cifar-10-batches-py"
tar_gz_path = "cifar-10-python.tar.gz"

IMAGE_PIXEL = 3072
CLASS_NUM = 10

# Download the CIFAR-10 dataset if not exist.
DL.process(cifar10_dataset_folder_path, tar_gz_path)

# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(cifar10_dataset_folder_path, preprc.normalize, preprc.one_hot_encode)

# Load the Preprocessed Validation data
train_features, train_labels = pickle.load(open('preprocess_train.p', mode='rb'))
train_features = train_features.reshape((-1, IMAGE_PIXEL))
train_labels = train_labels.reshape((-1, CLASS_NUM))

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
valid_features = valid_features.reshape((-1, IMAGE_PIXEL))
valid_labels = valid_labels.reshape((-1, CLASS_NUM))

# Load the Preprocessed Test data
test_features, test_labels = pickle.load(open('preprocess_test.p', mode='rb'))
test_features = test_features.reshape((-1, IMAGE_PIXEL))
test_labels = test_labels.reshape((-1, CLASS_NUM))

##############################
## Build the Neural Network ##
##############################

# Neural Network Parameters 
fully_outputs       = list()

# DNN Layer 1
#fully_outputs.append(256)
fully_outputs.append(256)

# DNN Layer 2
#fully_outputs.append(128)

# DNN Layer 3
#fully_outputs.append(64)

# Final Output Layer
final_out = CLASS_NUM

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = NNs.neural_net_image_input((32, 32, 3))
y = NNs.neural_net_label_input(CLASS_NUM)
keep_prob = NNs.neural_net_keep_prob_input()

# Model
print("\n===============================================================================================")
print("========================================Build the Model========================================")
print("===============================================================================================")
fully_out = NNs.fully_net(x, fully_outputs, final_out)
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

##############################
###### Hyperparameters #######
##############################
epochs = 15
batch_size = 128
keep_probability = 0.75
        
save_model_path = './checkpoint'
if not os.path.exists(save_model_path):
    os.makedirs(os.path.join(save_model_path))

max_valid_acc = 0.50

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())   

    # Training cycle
    for epoch in range(epochs):       
        
        # Shuffle the index of data in every epoch
        idx = [n for n in range(0, len(train_features))]
        random.shuffle(idx)
        
        batch_count = 0
        for i in range(0, len(train_features), batch_size):
            batch_count += 1
            batch_features = train_features[idx[i:i+batch_size]].reshape((-1, IMAGE_PIXEL))
            batch_labels = train_labels[idx[i:i+batch_size]]
            
            start = time.time()
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            end = time.time()          
            
            print('Epoch {:>2}, CIFAR-10 Batch {}[{:.4f} sec/batch]:  '.format(epoch + 1, batch_count, (end-start)), end='')
            train_loss, train_acc, valid_loss, valid_acc = print_stats(sess, batch_features, batch_labels, cost, accuracy)

            if(valid_acc > 0.50) and (valid_acc > max_valid_acc):
                max_valid_acc = valid_acc                
                
                # Save Model
                saver = tf.train.Saver()
                saver.save(sess, os.path.join(save_model_path, 'ckpt'))
                
       
    # Load checkpoint        
    print("Max validation acc. {}".format(max_valid_acc))
    saver.restore(sess, os.path.join(save_model_path, 'ckpt'))
    test_neural_network(sess, test_features, test_labels, cost, accuracy)



