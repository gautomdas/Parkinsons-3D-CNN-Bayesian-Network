from __future__ import print_function
from hyperopt import Trials, STATUS_OK, rand
from hyperas import optim
from hyperas.distributions import choice, uniform
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.optimizers import RMSprop, Adam
from scipy.misc import imread
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
    Y_train=train_y
    X_test=test_x
    Y_test=test_y

    return X_train, X_test, Y_train, Y_test


def model(X_train, X_test, Y_train, Y_test):
    model = Sequential()
    model.add(Dense(512, input_shape=(1, 64, 64, 64)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([400, 512, 600])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Flatten())


    model.add(Dense(2))
    model.add(Activation('softmax'))

    #rms = RMSprop()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    nb_epoch = 1
    batch_size = 50

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              verbose=2,
              validation_data=(X_test, Y_test))

    score, acc = model.evaluate(X_test, Y_test, verbose=0)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':

    X_train, X_test, Y_train, Y_test = data()

    '''
    Generate ensemble model from optimization run:
    First, run hyperas optimization on specified setup, i.e. 10 trials with TPE,
    then return the best 5 models and create a majority voting model from it.
    '''
    ensemble_model = optim.best_ensemble(nb_ensemble_models=5,
                                         model=model, data=data,
                                         algo=rand.suggest, max_evals=10,
                                         trials=Trials(),
                                         voting='hard')

    preds = ensemble_model.predict(X_test)

    y_test= Y_test
    pred1=   np_utils.to_categorical(preds,2)

    print("-----------------------------------------------")
    print("Accuracy Score :", accuracy_score(pred1, y_test))