import os
import csv
import nibabel as nib
import numpy as np
import scipy.ndimage as spc


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

def findVisit(array, string):
   for row in array:
       if(row[0] == string and not(row[5]=="90" or row[5] == "21")):
           return row

def findHY(array, string):
   for row in array:
       if(row[1] == string):
           return row

def pad(array, final):
   returnArray = []
   for eachLayer in array:
       layer = []
       for eachRow in eachLayer:
           finRow = addZeroes(eachRow, final)
           layer.append(finRow)
       returnArray.append(layer)
   for each in range(0, final-len(array)):
       returnArray.append(np.zeros((final, final)))

   return np.array(returnArray)

def addZeroes(array, des):
   ret = array.tolist()
   for eachPass in range(0, des-len(array)):
       ret.append(0)
   ret = np.array(ret)
   return ret


root_dir = os.path.abspath('./')
img_dir = os.path.join(root_dir, '218PixelSet')
sec_dir = os.path.join(root_dir, 'CutFile')
csv_dir = os.path.join(root_dir, 'Run_CSV')
init_file = os.path.join(csv_dir, 'positive_negative.csv')



checked = []

import csv

results = readcsv(init_file)

count = 0
for subdir, dirs, files in os.walk(img_dir):
   counter = 0
   for file in files:
       k = nib.load(os.path.join(img_dir, file))
       img = np.array(k.dataobj)
       print(k.shape)
       #t = spc.zoom(k, 1.40659341)
       #t = pad(k, 218)
       #t = spc.zoom(k, (0.587155963/2))\\
       a = [[row[65:125] for row in plane] for plane in img.tolist()]
       t = np.array(a)
       output = nib.Nifti1Image(t, affine=np.eye(4))
       nib.save(output, os.path.join(sec_dir, file))
       #print(t.shape)
       print("______")

