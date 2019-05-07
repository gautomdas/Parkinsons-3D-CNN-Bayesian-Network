"""
Created on Fri Nov  3 19:37:44 2017

@author: Gautom Das
"""

import os
import csv

root_dir = os.path.abspath('./')
img_dir = os.path.join(root_dir, 'JPEG_Image')

pre_read = os.path.join(root_dir, 'pre_dose_patients.csv')
post_read = os.path.join(root_dir, 'post_dose_patients.csv')

pre_dose = os.path.join(root_dir, 'pre_dose_train_JPEG.csv')
post_dose = os.path.join(root_dir, 'post_dose_train_JPEG.csv')



def padZero(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 10
    vector[-pad_width[1]:] = 10
    return vector



def hoehn_yahr_data(file_name, file):
    myFile = open(file, 'r', newline='')
    ret_array = []

    with myFile:
        readerTest = csv.reader(myFile, delimiter=' ', quotechar=',')
        for line in myFile:
            a = ((line.split(',')))
            r = a[0]
            b = ((file_name.split('_'))[1])
            if(r == b):
                ret_array.append(a[1])
                ret_array.append(a[2])
                ret_array.append(a[3])
                if(a[5] == "1\r\n"):
                    a[5] = '1'
                if (a[5] == "2\r\n"):
                    a[5] = '2'
                if (a[5] == "0\r\n"):
                    a[5] = '0'
                if (a[5] == "3\r\n"):
                    a[5] = '3'
                if (a[5] == "4\r\n"):
                    a[5] = '4'
                if (a[5] == "5\r\n"):
                    a[5] = '5'
                ret_array.append(a[5])

                return ret_array


desired_shape = [300, 300]
pre_dose_array = []
post_dose_array = []

for subdir, dirs, files in os.walk(img_dir):
    counter = 0
    for file in files:
        current_pre = []
        current_post = []

        apple =  str(file)

        file_input = file.split("_")
        flag = True
        for no_one_cares in file_input:
            if(no_one_cares.lower() == "cor"):
                flag = False

        if(flag):
            current_pre = hoehn_yahr_data(apple, pre_read)
            current_post = hoehn_yahr_data(apple, post_read)

            if(current_pre != None):
                current_pre.append(str(file))
                pre_dose_array.append(current_pre)

            if (current_post != None):
                current_post.append(file)
                post_dose_array.append(current_post)
        print(counter)
        counter+=1

print(pre_dose_array)

print("Finish")

myFile = open(pre_dose,'w' ,newline='')
with myFile:
    #write 2d array to csv
   writer = csv.writer(myFile)
   writer.writerows(pre_dose_array)

myFile = open(post_dose,'w' ,newline='')
with myFile:
    #write 2d array to csv
   writer = csv.writer(myFile)
   writer.writerows(post_dose_array)