"""
Created on Fri Nov  3 19:37:44 2017

@author: Gautom Das
"""

from keras.layers.merge import concatenate
from keras.models import Model, Sequential
import keras
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D,MaxPooling2D, Flatten, Input, BatchNormalization
from keras.utils.vis_utils import plot_model
import os
import numpy as np
import nibabel as nib
import random
import csv
import scipy.ndimage as nd
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
data_dir = os.path.join(root_dir, 'Flats')
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

def file_split(k_parts, seed):
    random.seed(seed)
    random.shuffle(allFiles)

    fin_array = []
    breaks = int(len(allFiles)/k_parts)
    print("Size of Each Set: "+str(breaks))

    for row in allFiles:
        fin_array.append(row)
    final = split_seq(fin_array, k_parts)

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
        img_name = row[0].split(".")[0] + ".jpg"
        file = os.path.join(data_dir, img_name)
        if os.path.exists(file):
            labels.append(label)
            if label == "0":
                zeroes += 1
            else:
                ones += 1

            img = nd.imread(file)
            new_image = np.asarray(img)
            temp.append(new_image)

            name = img_name.split(".")[0].split("_")
            id_val = name[len(name)-1][1:]

            aux_inputs.append(ret(id_val, all_identity))
        #else:
            #print(file)
    print("_"*25)
    print("TRAIN DATA")
    tot = ones+zeroes
    print("Total Lables: "+str(tot))
    print("Percent Zeroes: "+str(zeroes/tot))
    print("Percent Ones: "+str(ones/tot))
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
        img_name = row[0].split(".")[0] + ".jpg"
        file = os.path.join(data_dir, img_name)
        if os.path.exists(file):
            labels.append(label)
            if label == "0":
                zeroes += 1
            else:
                ones += 1

            img = nd.imread(file)
            new_image =np.asarray(img)
            temp.append(new_image)
            name = img_name.split(".")[0].split("_")
            id_val = name[len(name)-1][1:]

            aux_inputs.append(ret(id_val, all_identity))
        #else:
            #print(file)
    print("_"*25)
    print("Validation DATA")
    tot = ones+zeroes
    print("Total Lables: "+str(tot))
    print("Percent Zeroes: "+str(zeroes/tot))
    print("Percent Ones: "+str(ones/tot))
    print("_"*25)
    return np.stack(temp), np.asarray(aux_inputs), labels



input = Input(shape=(1, 1092, 1308, ), name="main_input")
model1_in = Conv2D(16, kernel_size=(7, 7), border_mode='same', strides=(4,4))(input)
model1_in = Activation('relu')(model1_in)
model1_in = MaxPooling2D(pool_size=(3, 3), border_mode='same', strides=(2,2))(model1_in)
model1_in = BatchNormalization()(model1_in)
model1_in = Dropout(0.5)(model1_in)

model1_in = Conv2D(64, kernel_size=(5, 5), border_mode='same', strides=(1,1))(model1_in)
model1_in = Activation('relu')(model1_in)
model1_in = MaxPooling2D(pool_size=(3, 3), border_mode='same', strides=(1,1))(model1_in)
model1_in = BatchNormalization()(model1_in)
model1_in = Dropout(0.5)(model1_in)

model1_in = Flatten()(model1_in)

model1_out = Dense(2048, activation='sigmoid', name='layer_1')(model1_in)
model1_out = BatchNormalization()(model1_out)
model1_out = Dense(256,  activation='sigmoid', name='layer_7')(model1_out)
model1_out = BatchNormalization()(model1_out)
model1_out = Dense(16, activation='relu')(model1_out)


model2_in = Input(shape=(2, ), name="aux_input")
model2_out = Dense(2, input_dim=2, activation='relu', name='layer_2')(model2_in)
model2 = Model(model2_in, model2_out)


concatenated = concatenate([model1_out, model2_out])
out = Dense(2, activation='softmax', name='output_layer')(concatenated)

merged_model = Model([input, model2_in], out)
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

plot_model(merged_model, show_shapes=True, to_file=('model.png'))

counter = 0
for each in range(0, 1):
    count = 1
    splits = 10
    file = file_split(splits, 50+counter)
    print(len(file))
    for each in range(0, (splits-1)):
        print("_"*25)
        print("Round: "+str(count))
        print("\nData section begins...\n")
        temp, aux_inputs, labels = get_train_get_aux_input_get_label(count, file)
        train_x = temp[:, np.newaxis]
        #train_x = temp
        train_aux = aux_inputs
        train_y = keras.utils.to_categorical(labels, num_classes=class_numb)

        print("\nThree-quarters the way there...\n")
        temp, aux_inputs, labels = get_val_get_aux_input_get_label(count-1, file)
        val_x = temp[:, np.newaxis]
        #val_x = temp
        val_aux = aux_inputs
        val_y = keras.utils.to_categorical(labels, num_classes=class_numb)
        print("\nData section ends.\n")
        print("_"*25)
        merged_model.fit([train_x, train_aux], train_y, validation_data=([val_x, val_aux], val_y), batch_size=50, epochs=1, verbose=1, shuffle=True)
        count += 1

    temp, aux_inputs, labels = get_train_get_aux_input_get_label(0, file)
    test_x = temp[:, np.newaxis]
    #test_x = temp
    test_aux = aux_inputs
    test_y = keras.utils.to_categorical(labels, num_classes=class_numb)
    counter += 1

    print(merged_model.evaluate([test_x, test_aux], test_y, verbose=0))
