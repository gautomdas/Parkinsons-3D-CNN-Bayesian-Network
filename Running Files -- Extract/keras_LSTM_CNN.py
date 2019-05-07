import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv3D,MaxPooling3D, Flatten, Input, LSTM, Reshape
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import os
import csv
import random
import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.ndimage as spc

"""
Created on Fri Nov  3 19:37:44 2017

@author: Gautom Das
"""
import os
import numpy as np
import nibabel as nib
import pandas as pd
import random
import csv
from scipy import ndimage as nd
import tensorflow as tf
import scipy.ndimage as spc

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

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def ret(name, array):
    for row in array:
        if(row[1] == name):
            if(row[3]=='M'):
                return [0, int(row[4])]
            elif(row[3]=='F'):
                return [1, int(row[4])]

#--------------------------------------------------
class_numb = 2
#--------------------------------------------------

"""
All of the following rows of the program deal with file handling. 
"""
root_dir = os.path.abspath('./')
data_dir = os.path.join(root_dir, 'Zoom_Red')
sub_dir = os.path.join(root_dir, 'Run_CSV')

# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)
os.path.exists(sub_dir)

second = open(os.path.join(sub_dir, "all.csv"))
all_identity = []
for row in second:
    all_identity.append(row)

allFiles = readcsv(os.path.join(sub_dir, 'positive_negative.csv'))

createTwo(allFiles, "training.csv", "test.csv")

train = pd.read_csv(os.path.join(sub_dir, 'training.csv'))
train.head()
test = pd.read_csv(os.path.join(sub_dir, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(sub_dir, 'Sample_Submission.csv'))

all_fil = train.values

aux_inputs = []
temp = []
index=0
count = 0
labels = []
for img_name in train.filename:
    label = ''
    for vals in all_fil:
        if(vals[0]==img_name):
            label = vals[1]
    img_name = img_name+".gz"
    file = os.path.join(data_dir, img_name)
    if os.path.exists(file) and label != '':
        labels.append(label)
        img = nib.load(os.path.join(data_dir, img_name))
        k = np.array(img.dataobj)

        #fina = np.reshape(k, (k.shape[0], -1))
        #after = fina.reshape(91, 9919)
        name = img_name.split("_")[1]
        aux_inputs.append(ret(name, all_identity))
        temp.append(k)
    else:
        print(file)
    count += 1

print("___Done___")

train_x = np.stack(temp)
aux_inputs = np.asarray(aux_inputs)
# Split 80 20
split_size = int(train_x.shape[0] * 0.9)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = keras.utils.to_categorical(labels[:split_size], num_classes=class_numb), keras.utils.to_categorical(labels[split_size:], num_classes = class_numb)
#train_y, val_y = (train.label.values[:split_size]), (train.label.values[split_size:])

print(val_y.shape)


train_x= train_x[:, np.newaxis]
val_x = val_x[:, np.newaxis]

all_sec = test.values
test_aux = []
#Test File
temp = []
index=0

labels = []
for img_name in test.filename:
    label = ''
    for vals in all_sec:
        if (vals[0] == img_name):
            label = vals[1]
    img_name = img_name+".gz"
    file = os.path.join(data_dir, img_name)
    if os.path.exists(file) and label != '':
        labels.append(label)
        img = nib.load(os.path.join(data_dir, img_name))
        k = np.array(img.dataobj)

        #t = spc.zoom(k, 0.25)
        # fina = np.reshape(k, (k.shape[0], -1))
        # after = fina.reshape(91, 9919)

        test_aux.append(ret(name, all_identity))
        temp.append(k)

    else:
        print(file)

print("___Done___")
test_x = np.stack(temp)
test_x = test_x[:, np.newaxis]
test_aux = np.asarray(test_aux)
print(test_aux.shape)
test_y = keras.utils.to_categorical(labels, num_classes=class_numb)

model = Sequential()
model.add(Conv3D(16, kernel_size=(7, 7, 7), input_shape=(1, 91, 109, 80), border_mode='same', name="main", strides=(4,4,4)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same', strides=(2,2,2)))
model.add(Dropout(0.5))

model.add(LSTM(128))
model.add(Dense(2))
model.add(Activation('sigmoid'))


model.compile(loss=binary_crossentropy, optimizer=Adam(), metrics=['accuracy'])
model.summary()
#plot_model(model, show_shapes=True, to_file=('model.png'))

print(train_y.shape)
history = model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=20, epochs=150, verbose=1, shuffle=True)

print(model.evaluate(test_x, test_y, verbose=0))
model_json = model.to_json()

import json
with open('abc.txt', 'w') as outfile:
    json.dump(model_json, outfile)

import matplotlib.pyplot as plt

# Loss Curves
"""
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)
plt.show()
"""
# Accuracy Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['acc'], 'r', linewidth=3.0)
plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
plt.show()