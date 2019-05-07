from keras.models import load_model

"""
Created on Fri Nov  3 19:37:44 2017

@author: Gautom Das
"""

from keras.layers.merge import concatenate
from keras.models import Model, Sequential
import keras
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D,MaxPooling2D, Flatten, Input, BatchNormalization, GlobalAveragePooling2D
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
# returns a compiled model
# identical to the previous one
model = load_model('final.h5')

from vis.visualization import visualize_cam
from vis.visualization import visualize_activation
import matplotlib.pyplot as plt
import numpy as np

"""
All of the following rows of the program deal with file handling. 
"""
root_dir = os.path.abspath('./')
data_dir = os.path.join(root_dir, 'Second_Flats')
sub_dir = os.path.join(root_dir, 'Run_CSV')

output_dir = os.path.join(root_dir, 'scratch')
import numpy as np
import scipy.misc
def get_file(name):
    img = nd.imread(name)
    new_image = np.array(img)
    return new_image
import os
for filename in os.listdir(data_dir):
    true = filename+""
    filename = os.path.join(data_dir, filename)
    final = get_file(filename)
    final = np.reshape(final, (final.shape[0], final.shape[1], 1))
    print(final.shape)

    heat_map = visualize_cam(model=model, layer_idx = 3, filter_indices=None, seed_input=final, penultimate_layer_idx=None, backprop_modifier=None, grad_modifier=None)
    #heat_map = visualize_activation(model, layer_idx = 5, filter_indices=None, seed_input=None, input_range=(0, 255), backprop_modifier=None, grad_modifier=None, act_max_weight=1, lp_norm_weight=10, tv_weight=10)
    print(heat_map.shape)
    finalA = os.path.join(output_dir, true)
    print(finalA)
    print(os.path.exists(finalA))
    scipy.misc.imsave(finalA, heat_map)