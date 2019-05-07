"""
Created on Fri Nov  3 19:37:44 2017

@author: Gautom Das
"""

#Initial imports
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import os
import pandas as pd
import numpy as np
import random
import csv
import nibabel as nib


#Batch creation functions from MNIST tutorials
def dense_to_one_hot(labels_dense, num_classes=5):
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

def readcsv(filename):
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=",")

    rownum = 0
    a = []

    for row in reader:
        a.append(row)
        rownum += 1

    ifile.close()
    return a

def createTwo(array, fileOne, fileTwo):
    non = array[1:]
    print(non[0])
    random.shuffle(non)
    fin = non
    print(fin[0])
    training = get(fin, 0, len(fin)-101)
    test = get(fin, len(fin)-101, len(fin)-1)

    training.insert(0,["filename","label"])
    test.insert(0, ["filename", "label"])
    writeFile(fileOne, training)
    writeFile(fileTwo, test)


def writeFile(init_file, array):
    sub_dir = os.path.join(root_dir, 'Run_CSV')
    ini_file = os.path.join(sub_dir, init_file)
    myFile = open(ini_file, 'w', newline='')

    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(array)

def get(array, start, end):
    count = start
    final = []
    for eachPass in range(start, end):
        final.append(array[count])
        count+=1
    return final

#Randomize learning
seed = 128
rng = np.random.RandomState(seed)

"""
All of the following rows of the program deal with file handling. 
"""
root_dir = os.path.abspath('./')
data_dir = os.path.join(root_dir, 'Full_Size_No_Brain')
sub_dir = os.path.join(root_dir, 'Run_CSV')

# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)
os.path.exists(sub_dir)



allFiles = readcsv(os.path.join(sub_dir, 'hy_Pre_diagnosis.csv'))

createTwo(allFiles, "training.csv", "test.csv")

train = pd.read_csv(os.path.join(sub_dir, 'training.csv'))
train.head()
test = pd.read_csv(os.path.join(sub_dir, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(sub_dir, 'Sample_Submission.csv'))


temp = []
index=0
for img_name in train.filename:
    file = os.path.join(data_dir, img_name)
    if os.path.exists(file):
        img = nib.load(os.path.join(data_dir, img_name))
        k = np.array(img.dataobj)

        #fina = np.reshape(k, (k.shape[0], -1))
        #after = fina.reshape(91, 9919)

        temp.append(k)

    else:
        print(file)

print("___Done___")

train_x = np.stack(temp)

# Split 80 20
split_size = int(train_x.shape[0] * 0.8)

train_x, val_x = train_x[:split_size], train_x[split_size:]


#Test File
temp = []
index=0
for img_name in test.filename:
    file = os.path.join(data_dir, img_name)
    if os.path.exists(file):
        img = nib.load(os.path.join(data_dir, img_name))
        k = np.array(img.dataobj)

        #fina = np.reshape(k, (k.shape[0], -1))
        #after = fina.reshape(91, 9919)

        temp.append(k)

    else:
        print(file)

print("___Done___")
test_x = np.stack(temp)

train.head()
width = 91#128
height = 109#1408
depth = 91

##Mutable Parameters
# Training Parameters
learning_rate = 0.01
training_steps = 10#350 #epoch
batch_size = 50 #120
display_step = 5
# Network Parameters
num_input = 91 *91
timesteps = 109
num_hidden = 120
num_classes = 5
input_num_units = width*height*depth #128*1408

# tf Graph input
x = tf.placeholder("float", [None, timesteps, num_input])
y = tf.placeholder("float", [None, num_classes])

# Weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

keep_prob = tf.placeholder(tf.float32)


def RNN(x, weights, biases):
    x = tf.unstack(x, timesteps, 1)

    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

#To run model
logits = RNN(x, weights, biases)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Tensorboard Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss_op)
summary_op = tf.summary.merge_all
# Merge all summaries
merged_summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    counter = 0
    avg_acc = 0.0
    sum = 0.0
    for step in range(1, training_steps + 1):
        total_batch = int(train.shape[0] / batch_size)
        batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))

        sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0 or step == 1:
            counter += 1
            loss, acc , summary= sess.run([loss_op, accuracy, merged_summary_op], feed_dict={x: batch_x, y: batch_y})

            print("Step " + str(step) + ", Training Accuracy= " + "{:.6f}".format(acc))
            sum += float(acc)


    summary_writer = tf.summary.FileWriter("output", sess.graph)

    print("Optimization Finished!")

    pred_temp = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))

    print("\nTraining complete!")

    predict = tf.argmax(prediction, 1)
    pred = predict.eval({x: test_x.reshape(-1, 109, num_input)})
    sample_submission.filename = test.filename
    sample_submission.label = pred
    sample_submission.to_csv(os.path.join(sub_dir, 'sub_lstm.csv'), index=False)

    print("\nSample File complete!")

sample_dir = os.path.join(sub_dir, 'sub_lstm.csv')
test_dir = os.path.join(sub_dir, 'test.csv')

guess = np.genfromtxt(sample_dir, delimiter=',')
correct = np.genfromtxt(test_dir, delimiter=',')

counter = 0
correct_vals = 0
for each_row in guess:
    if (each_row[1] == correct[counter][1]):
        correct_vals += 1
    counter += 1

acc = (correct_vals / counter * 100)
print("\nCalculated Accuracy : ", acc)
