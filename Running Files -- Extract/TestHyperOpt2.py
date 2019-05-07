from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding

from keras.layers import TimeDistributed
from scipy.misc import imread
from keras.layers import Conv3D,MaxPooling3D, Flatten, Input
import nibabel as nib
import os
import csv
import random
import keras
import numpy as np
import pandas as pd


# --------------------------------------------------
class_numb = 2
# --------------------------------------------------


def data():
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
        training = get(fin, 0, len(fin) - 101)
        test = get(fin, len(fin) - 101, len(fin) - 1)

        training.insert(0, ["filename", "label"])
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
            count += 1
        return final

    def ret(name, array):
        for row in array:
            if (row[1] == name):
                if (row[3] == 'M'):
                    return [0, int(row[4])]
                elif (row[3] == 'F'):
                    return [1, int(row[4])]


    class_numb = 2


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
    index = 0
    for img_name in train.filename:
        img_name = img_name + ".gz"
        file = os.path.join(data_dir, img_name)
        if os.path.exists(file):
            img = nib.load(os.path.join(data_dir, img_name))
            k = np.array(img.dataobj)

            # fina = np.reshape(k, (k.shape[0], -1))
            # after = fina.reshape(91, 9919)
            name = img_name.split("_")[1]
            aux_inputs.append(ret(name, all_identity))
            temp.append(k)

        else:
            print(file)

    print("___Done___")

    train_x = np.stack(temp)
    aux_inputs = np.asarray(aux_inputs)
    # Split 80 20
    split_size = int(train_x.shape[0] * 0.9)

    train_x, val_x = train_x[:split_size], train_x[split_size:]
    train_y, val_y = keras.utils.to_categorical(train.label.values[:split_size],
                                                num_classes=class_numb), keras.utils.to_categorical(
        train.label.values[split_size:], num_classes=class_numb)
    # train_y, val_y = (train.label.values[:split_size]), (train.label.values[split_size:])

    print(val_y.shape)

    train_x = train_x[:, np.newaxis]
    val_x = val_x[:, np.newaxis]

    test_aux = []
    # Test File
    temp = []
    index = 0
    for img_name in test.filename:
        img_name = img_name + ".gz"
        file = os.path.join(data_dir, img_name)
        if os.path.exists(file):
            img = nib.load(os.path.join(data_dir, img_name))
            k = np.array(img.dataobj)

            # t = spc.zoom(k, 0.25)
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

    X_train=train_x
    y_train=train_y
    X_test=test_x
    y_test=test_y

    maxlen = 100
    max_features = 20000

    return X_train, X_test, y_train, y_test, max_features, maxlen


def model(X_train, X_test, y_train, y_test, max_features, maxlen):
    class_numb = 2

    cnn = Sequential()

    # define CNN model
    cnn.add((Conv3D(64, kernel_size=(4, 4, 4), input_shape=(1, 64, 64, 64), border_mode='same', name="main")))
    cnn.add(Activation('relu'))
    cnn.add((MaxPooling3D(pool_size=(4, 4, 4), border_mode='same')))
    cnn.add((Flatten()))

    model = Sequential()
    model.add(TimeDistributed(cnn,  input_shape=(1,64, 64, 64)))

    # define LSTM model
    model.add(LSTM(64))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense(class_numb, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    checkpointer = ModelCheckpoint(filepath='keras_weights.hdf5',
                                   verbose=2,
                                   save_best_only=True)

    model.fit(X_train, y_train,
              batch_size=50,
              nb_epoch=1,
              validation_split=0.08
              #,              callbacks=[early_stopping, checkpointer]
              )

    score, acc = model.evaluate(X_test, y_test, verbose=0)

    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    print(best_run)