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

def ret(name, array):
    for row in array:
        if(row[0] == name):
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
            k = np.array(img.dataobj)
            name = img_name.split(".")[0].split("_")
            id_val = name[len(name)-1][1:]

            aux_inputs.append(ret(id_val, all_identity))
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

def get_val_get_aux_input_get_label(count, tot_array):
    array = tot_array[count]
    ones = 0
    zeroes = 0
    array = array[0:int(len(array)/2)]

    print("starts with;")
    print(array[0:3])
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
            k = np.array(img.dataobj)
            name = img_name.split(".")[0].split("_")
            id_val = name[len(name)-1][1:]

            aux_inputs.append(ret(id_val, all_identity))
            temp.append(k)
        else:
            print(file)
    print("_"*25)
    print("Validation DATA")
    tot = ones+zeroes
    print("Total Lables: "+str(tot))
    print("Percent Zeroes: "+str(float(zeroes)/float(tot)))
    print("Percent Ones: "+str(float(ones)/float(tot)))
    print("_"*25)
    return np.stack(temp), np.asarray(aux_inputs), labels

input = Input(shape=(1, 182, 218, 160, ), name="main_input")
model1_in = Conv3D(16, kernel_size=(7, 7, 7), input_shape=(1, 182, 218, 160), border_mode='same', strides=(4,4,4))(input)
model1_in = Activation('relu')(model1_in)
model1_in = MaxPooling3D(pool_size=(3, 3, 3), border_mode='same', strides=(2,2,2))(model1_in)
model1_in = BatchNormalization()(model1_in)
model1_in = Dropout(0.5)(model1_in)

model1_in = Conv3D(64, kernel_size=(5, 5, 5), border_mode='same', strides=(1,1,1))(model1_in)
model1_in = Activation('relu')(model1_in)
model1_in = MaxPooling3D(pool_size=(3, 3, 3), border_mode='same', strides=(1,1,1))(model1_in)
model1_in = BatchNormalization()(model1_in)
model1_in = Dropout(0.5)(model1_in)
model1_in = Flatten()(model1_in)

model1_out = Dense(4096, activation='sigmoid', name='layer_1')(model1_in)
model1_out = BatchNormalization()(model1_out)
model1_out = Dense(256,  activation='sigmoid', name='layer_9')(model1_out)
model1_out = BatchNormalization()(model1_in)
model1_out = Dense(16, activation='relu')(model1_out)


model2_in = Input(shape=(2, ), name="aux_input")
model2_out = Dense(2, input_dim=2, activation='relu', name='layer_2')(model2_in)
model2 = Model(model2_in, model2_out)


concatenated = concatenate([model1_out, model2_out])
out = Dense(2, activation='softmax', name='output_layer')(concatenated)

merged_model = Model([input, model2_in], out)
merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


counter = 0
splits = 12
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
        train_aux = aux_inputs
        train_y = keras.utils.to_categorical(labels, num_classes=class_numb)

        print("\nThree-quarters the way there...\n")
        temp, aux_inputs, labels = get_val_get_aux_input_get_label(1, file)
        val_x = temp[:, np.newaxis]
        val_aux = aux_inputs
        val_y = keras.utils.to_categorical(labels, num_classes=class_numb)
        print("\nData section ends.\n")
        print("_"*25)
        merged_model.fit([train_x, train_aux], train_y, validation_data=([val_x, val_aux], val_y), batch_size=20, epochs=15, verbose=1, shuffle=True)
        count += 1
    del temp, train_x, val_x
    temp, aux_inputs, labels = get_train_get_aux_input_get_label(0, file)
    test_x = temp[:, np.newaxis]
    test_aux = aux_inputs
    test_y = keras.utils.to_categorical(labels, num_classes=class_numb)

    print(merged_model.evaluate([test_x, test_aux], test_y, verbose=0))
    del temp, test_x
#model_json = model.to_json()
merged_model.save('merged.h5')
import json
#with open('abc.txt', 'w') as outfile:
    #json.dump(model_json, outfile)

import matplotlib.pyplot as plt

# # Loss Curves
# plt.figure(figsize=[8, 6])
# plt.plot(history.history['loss'], 'r', linewidth=3.0)
# plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
# plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
# plt.xlabel('Epochs ', fontsize=16)
# plt.ylabel('Loss', fontsize=16)
# plt.title('Loss Curves', fontsize=16)
# plt.show()
# # Accuracy Curves
# plt.figure(figsize=[8, 6])
# #plt.plot(history.history['acc'], 'r', linewidth=3.0)
# #plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
# plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
# plt.xlabel('Epochs ', fontsize=16)
# plt.ylabel('Accuracy', fontsize=16)
# plt.title('Accuracy Curves', fontsize=16)
# plt.show()