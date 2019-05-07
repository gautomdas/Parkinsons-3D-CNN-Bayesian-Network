from keras.layers.merge import concatenate

from keras.utils.vis_utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Flatten
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
import csv
import random
import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.ndimage as spc



input = Input(shape=(1, 182, 218, 160, ), name="main_input")
model1_in = Conv3D(16, kernel_size=(7, 7, 7), input_shape=(1, 182, 218, 160), border_mode='same', strides=(4,4,4))(input)
model1_in = Activation('relu')(model1_in)
model1_in = MaxPooling3D(pool_size=(3, 3, 3), border_mode='same', strides=(2,2,2))(model1_in)
model1_in = Dropout(0.5)(model1_in)

model1_in = Conv3D(64, kernel_size=(5, 5, 5), border_mode='same', strides=(1,1,1))(model1_in)
model1_in = Activation('relu')(model1_in)
model1_in = MaxPooling3D(pool_size=(3, 3, 3), border_mode='same', strides=(1,1,1))(model1_in)
model1_in = Dropout(0.5)(model1_in)
model1_in = Flatten()(model1_in)

model1_out = Dense(16384, input_dim=(125000000, ), activation='sigmoid', name='layer_1')(model1_in)
model1_out = Dense(256,  activation='sigmoid', name='layer_1')(model1_in)
model1_out = Dense(16, activation='relu')(model1_out)


model2_in = Input(shape=(2, ))
model2_out = Dense(2, input_dim=2, activation='relu', name='layer_2')(model2_in)
model2 = Model(model2_in, model2_out)


concatenated = concatenate([model1_out, model2_out])
out = Dense(2, activation='softmax', name='output_layer')(concatenated)

merged_model = Model([input, model2_in], out)
merged_model.compile(loss='binary_crossentropy', optimizer='adam',
metrics=['accuracy'])

plot_model(merged_model, show_shapes=True, to_file=('model.png'))