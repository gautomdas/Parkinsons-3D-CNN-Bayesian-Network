"""
Created on Fri Nov  3 19:37:44 2017

@author: Gautom Das
"""

from keras.layers.merge import concatenate
from keras.models import Model, Sequential
import keras
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv3D,MaxPooling3D, Flatten, Input, BatchNormalization
import os
import numpy as np
import nibabel as nib
import random
import csv
import tensorflow as tf

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


#--------------------------------------------------
class_numb = 2
#--------------------------------------------------

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

second = readcsv(os.path.join(sub_dir, "all.csv"))
all_identity = []
for row in second:
    rows = []
    for element in row:
        rows.append(element)
    all_identity.append(rows)

allFiles = readcsv(os.path.join(sub_dir, 'positive_negative.csv'))

def file_split(k, k_parts, seed):
    random.seed(seed)
    random.shuffle(allFiles)

    fin_array = []
    breaks = int(len(allFiles)/k_parts)
    print("Size of Each Set: "+str(breaks))

    for row in allFiles:
        fin_array.append(row)
    final = split_seq(fin_array, k_parts)
    print("pause")
    print(len(final))
    k_test = final[k]
    if (k>=len(final)-1):
        k = -1
    k_validate = final[k+1]
    final.remove(k_test)
    final.remove(k_validate)
    final.insert(0, k_validate)
    final.insert(0, k_test)
    return final

def split_seq(seq, size):
    newseq = []
    splitsize = 1.0 / size * len(seq)
    for i in range(size):
        newseq.append(seq[int(round(i * splitsize)):int(round((i + 1) * splitsize))])
    return newseq

def get_train_get_aux_input_get_label(count, tot_array):
    array = tot_array[count]
    print("starts with;")
    print(array[0:3])
    ones = 0
    zeroes = 0
    temp = []
    aux_inputs = []
    labels = []
    for row in array:
        label = row[1]
        img_name = row[0] + ".gz"
        file = os.path.join(data_dir, img_name)
        if os.path.exists(file):
            labels.append(label)
            if label == "0":
                zeroes += 1
            else:
                ones += 1
            img = nib.load(os.path.join(data_dir, img_name))
            k = np.float32(np.array(img.dataobj))
            name = img_name.split(".")[0].split("_")
            id_val = name[len(name)-1][1:]

            aux_inputs.append([])
            temp.append(k)
        else:
            print(file)
    print("_"*25)
    print("TRAIN DATA")
    print(ones)
    print(zeroes)
    tot = ones+zeroes
    print(tot)
    print("Total Lables: "+str(tot))
    print("Percent Zeroes: "+str(float(zeroes)/float(tot)))
    print("Percent Ones: "+str(float(ones)/float(tot)))
    print("_"*25)
    return np.stack(temp), np.asarray(aux_inputs), labels

import tensorflow as tf
import tensorflow_probability as tfp
import keras

model = tf.keras.Sequential([
    tf.keras.layers.Reshape([1, 182, 218, 160]),
    tfp.layers.Convolution3DFlipout(
        16, kernel_size=7, padding='SAME', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling3D(pool_size=[3, 3, 3],
                                 strides=[2, 2, 2],
                                 padding='SAME'),
    tf.keras.layers.Flatten(),
    tfp.layers.DenseFlipout(1),
])




counter = 0
splits = 40
k = 0
file = file_split(k, splits, 0)
val_history = []
loss_history = []


for each in range(0, 1):
    count = 1

    for each in range(0, (len(file)-2)):
        print("_"*25)
        print("Round: "+str(count))
        print("\nData section begins...\n")
        temp, aux_inputs, labels = get_train_get_aux_input_get_label(count, file)
        train_x = temp[:, np.newaxis]
        train_y =np.asarray([int(value) for value in labels]).astype('float32').reshape((-1,1))
        print(train_y.shape)

        logits = model(train_x)
        neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
            labels=train_y, logits=logits)
        kl = sum(model.losses)
        loss = neg_log_likelihood + kl
        train_op = tf.compat.v1.train.AdamOptimizer().minimize(loss)
        count += 1

        # Build metrics for evaluation. Predictions are formed from a single forward
        # pass of the probabilistic layers. They are cheap but noisy predictions.

        tf.metrics.Accuracy("example", [tf.argmax(input=train_y, axis=1),tf.argmax(input=logits, axis=1)]).result(write_summary=True)
