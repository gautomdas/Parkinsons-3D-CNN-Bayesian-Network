import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np

PATH = os.getcwd()
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

LOG_DIR = PATH + '/mnist-tensorboard/log-1'
metadata = os.path.join(LOG_DIR, 'metadata.tsv')
root_dir = os.path.abspath('./')

data_dir = os.path.join(root_dir, 'Second_Flats')
sub_dir = os.path.join(root_dir, 'Run_CSV')

# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)
os.path.exists(sub_dir)

second = readcsv(os.path.join(sub_dir, "positive_negative.csv"))

all_identity = []
for row in second:
    rows = []
    for element in row:
        rows.append(element)
    all_identity.append(rows)


root_dir = os.path.abspath('./')
data_dir = os.path.join(root_dir, 'scratch')
sub_dir = os.path.join(root_dir, 'Run_CSV')

output_dir = os.path.join(root_dir, 'scratch')
import numpy as np
import scipy.misc
def get_file(name):
    img = nd.imread(name)
    new_image = np.array(img)
    return new_image
def get_id(name):
    for row in all_identity:
        if row[0].split(".")[0] == name.split(".")[0]:
            return row[1]
import os
data = []
ids = []
for filename in os.listdir(data_dir):
    true = filename+""
    print(get_id(true))
    ids.append(get_id(true))
    filename = os.path.join(data_dir, filename)
    final = get_file(filename)
    print(final.shape)
    final = np.reshape(final, ((final.shape[0] *final.shape[1]*final.shape[2])))
    print(final.shape)
    data.append(final)

data = np.asarray(data)
print(data.shape)
print(len(ids))
print("here")
images = tf.Variable(data, name='images')
del data
print("here")
# def save_metadata(file):
with open(metadata, 'w') as metadata_file:
    for row in ids:
        c = row
        metadata_file.write('{}\n'.format(c))

with tf.Session() as sess:
    saver = tf.train.Saver([images])

    sess.run(images.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

    # for calling the tensorboard you should be in that drive and call the entire path
    # tensorboard --logdir=/Technical_works/tensorflow/mnist-tensorboard/log-1 --port=6006