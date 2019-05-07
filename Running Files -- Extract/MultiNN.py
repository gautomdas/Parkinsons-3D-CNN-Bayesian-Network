"""
Created on Fri Nov  3 19:37:44 2017

@author: Gautom Das
"""

#Initial imports
import tensorflow as tf
import os
import pandas as pd
from scipy.misc import imread
import numpy as np
import scipy.ndimage as nd


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


#Batch creation functions from MNIST tutorials
def dense_to_one_hot(labels_dense, num_classes=5):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def preproc(unclean_batch_x):
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    return temp_batch

def batch_creator(batch_size, dataset_length, dataset_name):
    batch_mask = rng.choice(dataset_length, batch_size)
    batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, input_num_units)
    batch_x = preproc(batch_x)
    if dataset_name == 'train':
        batch_y = eval(dataset_name).ix[batch_mask, 'label'].values
        batch_y = dense_to_one_hot(batch_y)
    return batch_x, batch_y

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

#Randomize learning
seed = 128
rng = np.random.RandomState(seed)

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



allFiles = readcsv(os.path.join(sub_dir, 'hy_Pre_diagnosis.csv'))

createTwo(allFiles, "training.csv", "test.csv")

train = pd.read_csv(os.path.join(sub_dir, 'training.csv'))
train.head()
test = pd.read_csv(os.path.join(sub_dir, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(sub_dir, 'Sample_Submission.csv'))

temp = []
index=0
for img_name in train.filename:
    file = os.path.join(data_dir, img_name)
    if os.path.exists(file):
        img = nib.load(os.path.join(data_dir, img_name))
        k = np.array(img.dataobj)

        t = spc.zoom(k, 0.25)
        print(t.shape)
        #fina = np.reshape(k, (k.shape[0], -1))
        #after = fina.reshape(91, 9919)

        temp.append(t)

    else:
        print(file)

print("___Done___")

train_x = np.stack(temp)

# Split 80 20
split_size = int(train_x.shape[0] * 0.8)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train.label.values[:split_size], train.label.values[split_size:]

#Test File
temp = []
index=0
for img_name in test.filename:
    file = os.path.join(data_dir, img_name)
    if os.path.exists(file):
        img = nib.load(os.path.join(data_dir, img_name))
        k = np.array(img.dataobj)

        t = spc.zoom(k, 0.25)
        print(t.shape)
        # fina = np.reshape(k, (k.shape[0], -1))
        # after = fina.reshape(91, 9919)

        temp.append(t)

    else:
        print(file)

print("___Done___")
test_x = np.stack(temp)

#Mutable variables
n_nodes_hl1 = 500
n_nodes_hl2 = 250
n_nodes_hl3 = 125

n_classes = 5
batch_size = 120
input_num_units = 23*23*27
output_num_units = 5
learning_rate = 0.1
hm_epochs = 15

# Place holders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

#Three layer neural network
def neural_network_model(data, keep_prob):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([input_num_units, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes])), }

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    dropout = tf.nn.dropout(l3, keep_prob)

    output_layer = tf.matmul(dropout, output_layer['weights']) + output_layer['biases']
    return output_layer


def train_neural_network(x):
    prediction = neural_network_model(x, 1.0)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Tensorboard Create a summary to monitor cost tensor
    tf.summary.scalar("loss", cost)
    summary_op = tf.summary.merge_all
    # Merge all summaries
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            avg_cost = 0
            total_batch = int(train.shape[0] / batch_size)
            for i in range(total_batch):
                epoch_x, epoch_y = batch_creator(batch_size, train_x.shape[0], 'train')
                _, c , summary= sess.run([optimizer, cost,merged_summary_op], feed_dict={x: epoch_x, y: epoch_y})
                avg_cost += c / total_batch

            print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost))

        pred_temp = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
        print("\nValidation Accuracy:", accuracy.eval({x: val_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)}))

        print("\nTraining complete!")

        predict = tf.argmax(prediction, 1)
        pred = predict.eval({x: test_x.reshape(-1, input_num_units)})
        sample_submission.filename = test.filename
        sample_submission.label = pred
        sample_submission.to_csv(os.path.join(sub_dir, 'sub_cnn.csv'), index=False)

train_neural_network(x)

sample_dir = os.path.join(sub_dir, 'sub_cnn.csv')
test_dir = os.path.join(sub_dir, 'test.csv')

guess = np.genfromtxt(sample_dir, delimiter=',')
correct = np.genfromtxt(test_dir, delimiter=',')

counter = 0
correct_vals = 0
for each_row in guess:
    if(each_row[1] == correct[counter][1]):
        correct_vals += 1
    counter += 1

acc = (correct_vals/counter * 100)
print("\nCalculated Accuracy : ", acc)