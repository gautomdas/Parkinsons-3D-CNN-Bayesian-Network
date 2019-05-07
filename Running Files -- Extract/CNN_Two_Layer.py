"""
Created on Fri Nov  3 19:37:44 2017

@author: Gautom Das
"""

#Initial Imports
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import scipy.ndimage as nd
import cv2
import csv

# Training Parameters
learning_rate = 0.001
batch_size = 100
hm_epochs = 100
dropout = 0.75
# Network Parameters
width = 175
height = 175
input_num_units = width*height #  data input (img shape: 28*28)
class_num = 5 #  total classes (0-9 digits)
#Convolution Parameters
pool_size = 10
strides = 10
filters = 64
kernel = 20
#Second Layer
pool_size_2 = 5
strides_2 = 5
filters_2 = 128
kernel_2 = 10
#Third Layer
pool_size_3 = 1
strides_3 = 1
filters_3 = 256
kernel_3 = 1


#Batch creation functions from MNIST tutorials
def dense_to_one_hot(labels_dense, num_classes=class_num):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def preproc(unclean_batch_x):
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    return temp_batch

def batch_creator(batch_size, dataset_length, dataset_name):
    batch_mask = rng.choice(dataset_length, batch_size)
    batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, input_num_units)
    batch_x = preproc(batch_x)
    if dataset_name == 'train':
        batch_y = eval(dataset_name).ix[batch_mask, 'label'].values
        batch_y = dense_to_one_hot(batch_y)
    return batch_x, batch_y

#Randomize learning
seed = 128
rng = np.random.RandomState(seed)

"""
All of the following rows of the program deal with file handling. 
"""

root_dir = os.path.abspath('./')
datafile = 'J_All_PM5_Image_Files' #'J_All_PM5_Image_Files'
print ("\nData file : ", datafile)
data_dir = os.path.join(root_dir,datafile )
sub_dir = os.path.join(root_dir, 'H_All_CSV_Files')
test_dir = os.path.join(root_dir, 'L_Test_Directory')

pre_dose = os.path.join(root_dir, 'PRE_DOSE_COST.csv')

# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)
os.path.exists(sub_dir)

train = pd.read_csv(os.path.join(sub_dir, 'post_dose_train_JPEG.csv'))
test = pd.read_csv(os.path.join(sub_dir, 'test_image_posttest.csv'))
sample_submission = pd.read_csv(os.path.join(sub_dir, 'Sample_Submission.csv'))

train.head()

temp = []
for img_name in train.filename:
    image_path = os.path.join(data_dir, img_name)
    img = nd.imread(image_path)

    list = np.array(img)
    img = cv2.resize(img, (width,height))
    new_image = np.array(img).reshape((width,height))
    temp.append(new_image)
train_x = np.stack(temp)

# Split 90 10
split_size = int(train_x.shape[0] * 0.9)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train.label.values[:split_size], train.label.values[split_size:]

#Test File
temp = []
for img_name in test.filename:
    image_path = os.path.join(data_dir, img_name)
    img = nd.imread(image_path)

    list = np.array(img)
    img = cv2.resize(img, (width,height))
    new_image = np.array(img).reshape((width,height))
    temp.append(new_image)

test_x = np.stack(temp)


# Place holders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, class_num])

# Convolutional network
def conv_net(inputs, dropout):
    # Use scope for tensorboard.
    with tf.variable_scope('Two Layer Convolutional Neural Network'):
        x = inputs
        x = tf.reshape(x, shape=[-1, width, height, 1])

        conv1 = tf.layers.conv2d(x, filters, kernel, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, strides, pool_size)

        conv2 = tf.layers.conv2d(conv1, filters_2, kernel_2, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, strides_2, pool_size_2)

        conv3 = tf.layers.conv2d(conv2, filters_3, kernel_3, activation=tf.nn.relu)
        conv3 = tf.layers.max_pooling2d(conv3, strides_3, pool_size_3)

        fc1 = tf.contrib.layers.flatten(conv3)
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, dropout)

        out = tf.layers.dense(fc1, class_num)

    return out

CostArray = []
def train_neural_network(x):
    prediction = conv_net(x, dropout )
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

    # Tensorboard Create a summary to monitor cost tensor
    tf.summary.scalar("loss", cost)
    summary_op = tf.summary.merge_all
    # Merge all summaries
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            avg_cost = 0
            total_batch = int(train.shape[0] / batch_size)
            for i in range(total_batch):
                epoch_x, epoch_y = batch_creator(batch_size, train_x.shape[0], 'train')
                _, c , summary= sess.run([optimizer, cost,merged_summary_op], feed_dict={x: epoch_x, y: epoch_y})
                avg_cost += c / total_batch

            print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost))

            pred_temp = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
            print("\nValidation Accuracy:", accuracy.eval({x: val_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)}))

            CostArray.append([(epoch + 1), float(avg_cost)])
            print(CostArray)

        print("\nTraining complete!")

        predict = tf.argmax(prediction, 1)
        pred = predict.eval({x: test_x.reshape(-1, input_num_units)})
        sample_submission.filename = test.filename
        sample_submission.label = pred
        sample_submission.to_csv(os.path.join(sub_dir, 'sub_3cnn_pre.csv'), index=False)

train_neural_network(x)

sample_dir = os.path.join(sub_dir, 'sub_3cnn_pre.csv')
test_dir = os.path.join(sub_dir, 'test_image_posttest.csv')

guess = np.genfromtxt(sample_dir, delimiter=',')
correct = np.genfromtxt(test_dir, delimiter=',')

counter = 0
correct_vals = 0
for each_row in guess:
    if(each_row[1] == correct[counter][1]):
        correct_vals += 1
    counter += 1

acc = (correct_vals/(counter-1) * 100)
print("\nCalculated Accuracy : ", acc)

myFile = open(pre_dose,'w' ,newline='')
with myFile:
    #write 2d array to csv
   writer = csv.writer(myFile)
   writer.writerows(CostArray)