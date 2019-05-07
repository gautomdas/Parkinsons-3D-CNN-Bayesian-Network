from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv3D,MaxPooling3D, Flatten, Input
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import os
import numpy as np
import nibabel as nib
import pandas as pd
import random
import csv
from scipy import ndimage as nd
import tensorflow as tf
import scipy.ndimage as spc

from keras.models import Sequential
from keras.layers import Dense
import numpy as np




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
data_dir = os.path.join(root_dir, '64Pix')
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
for img_name in train.filename:
    img_name = img_name+".gz"
    file = os.path.join(data_dir, img_name)
    if os.path.exists(file):
        img = nib.load(os.path.join(data_dir, img_name))
        k = np.array(img.dataobj)

        #fina = np.reshape(k, (k.shape[0], -1))
        #after = fina.reshape(91, 9919)
        name = img_name.split("_")[1]
        aux_inputs.append(ret(name, all_identity))
        temp.append(k)

    else:
        print(file)

print("___Done___")

train_x = np.stack(temp)
aux_inputs = np.asarray(aux_inputs)
# Split 80 20
split_size = int(train_x.shape[0] * 0.8)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = keras.utils.to_categorical(train.label.values[:split_size], num_classes=class_numb), keras.utils.to_categorical(train.label.values[split_size:], num_classes = class_numb)
#train_y, val_y = (train.label.values[:split_size]), (train.label.values[split_size:])

print(val_y.shape)


train_x= train_x[:, np.newaxis]
val_x = val_x[:, np.newaxis]

test_aux = []
#Test File
temp = []
index=0
for img_name in test.filename:
    img_name = img_name+".gz"
    file = os.path.join(data_dir, img_name)
    if os.path.exists(file):
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
test_y = keras.utils.to_categorical(test.label.values, num_classes=class_numb)


n_batch = 16
n_epoch = 10
numunits = 128+128

model = Sequential()
model.add(Dense(numunits, activation='relu', input_shape=(1,64,64,64)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))

model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

plot_model(model, show_shapes=True, to_file=('model.png'))

history =model.fit(train_x, train_y, epochs=n_epoch, batch_size=n_batch, verbose=0)
# evaluate
result = model.predict(test_x, batch_size=n_batch, verbose=0)

#for value in result[0,:,0]:
#	print('%.1f' % value)
print(result)
accuracy = model.evaluate(x=test_x,y=test_y,batch_size=n_batch)
print("Accuracy: ",accuracy)


