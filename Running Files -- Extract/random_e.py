import numpy
import matplotlib
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution2D,MaxPooling2D, Flatten, Input
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
from scipy.misc import imread
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

def get_col(array, val):
    count = 0
    for row in array:
        if(row[0] == val):
            return count
        else:
            count += 1

#--------------------------------------------------
class_numb = 2
#--------------------------------------------------

"""
All of the following rows of the program deal with file handling. 
"""
root_dir = os.path.abspath('./')
data_dir = os.path.join(root_dir, 'Flast2.0')
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

aux_inputs = []
temp = []
index=0

alls = train.values.tolist()
labels = []

for img_name in train.filename:
    orig = img_name+""
    img_name = img_name.split(".")[0]+".jpg"
    file = os.path.join(data_dir, img_name)
    if os.path.exists(file):
        img = imread(file)
        k = np.array(img)

        labels.append(alls[get_col(alls, orig)][1])
        #t = spc.zoom(k, 0.5)
        #fina = np.reshape(k, (k.shape[0], -1))
        #after = fina.reshape(91, 9919)
        name = img_name.split("_")[1]
        aux_inputs.append(ret(name, all_identity))
        temp.append(k)

    else:
        print(file)

print("___Done___")

train_x = np.stack(temp)
print(train_x.shape)


aux_inputs = np.asarray(aux_inputs)
# Split 80 20
split_size = int(train_x.shape[0] * 0.9)

train_x, val_x = train_x[:split_size][:1], train_x[split_size:][:1]
train_y = keras.utils.to_categorical(labels[:split_size][:1], num_classes=2)
print(train_y.shape)
val_y = keras.utils.to_categorical(labels[split_size:][:1], num_classes=2)
#train_y, val_y = (train.label.values[:split_size]), (train.label.values[split_size:])


train_x= np.expand_dims(train_x, axis=3)
val_x = np.expand_dims(val_x, axis=3)

print(train_x.shape)

test_aux = []
#Test File
temp = []
labels = []
index=0
for img_name in test.filename:
    img_name = img_name.split(".")[0]+".jpg"
    file = os.path.join(data_dir, img_name)
    if os.path.exists(file):
        img = imread(file)
        k = np.array(img)

        labels.append(alls[get_col(alls, orig)][1])
        #t = spc.zoom(k, 0.5)
        # fina = np.reshape(k, (k.shape[0], -1))
        # after = fina.reshape(91, 9919)

        test_aux.append(ret(name, all_identity))
        temp.append(k)

    else:
        print(file)

print("___Done___")
temp = temp[:1]
test_x = np.stack(temp)

test_x= np.expand_dims(test_x, axis=3)
test_aux = np.asarray(test_aux)
print(test_aux.shape)
test_y = keras.utils.to_categorical(labels[:1], num_classes=2)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(10, activation='relu', input_shape=(1090, 1090, 1, )))
# set of FC => RELU layers
model.add(Dense(500))
model.add(Activation("relu"))

model.add(Convolution2D(50, 5, 5, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
# softmax classifier
model.add(Dense(class_numb))
model.add(Activation("softmax"))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
plot_model(model, show_shapes=True, to_file=('model.png'))

history = model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=5, epochs=15, verbose=1, shuffle=True)

print(model.evaluate(test_x, test_y, verbose=0))