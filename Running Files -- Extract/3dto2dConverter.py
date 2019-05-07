import os
import csv
import nibabel as nib
import numpy as np
import scipy.ndimage as spc
import scipy.ndimage as ned
import scipy.misc
import nibabel.volumeutils as nibA
#scipy.misc.imsave('outfile.jpg', image_array)


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
img_dir = os.path.join(root_dir, 'Full_Size_No_Brain')
sec_dir = os.path.join(root_dir, 'third_flats')
csv_dir = os.path.join(root_dir, 'Run_CSV')
init_file = os.path.join(csv_dir, 'positive_negative.csv')



checked = []

import csv

results = readcsv(init_file)

count = 0
for subdir, dirs, files in os.walk(img_dir):
   counter = 0
   for file in files:
       print(file)
       k = nib.load(os.path.join(img_dir, file))
       img = (np.array(k.dataobj))
       #img_two = np.asarray(k.dataobj)
       #print(k.shape)
       #t = spc.zoom(k, 1.40659341)
       #t = pad(k, 218)
       #tot = ned.zoom(img, (0.8, 0.8, 0.8), order=0)

       #final = np.asarray(tot)


       k = 80
       arr = []
       for each in range(0, 5):
           arrar = []
           for seach in range(0,5):
               layer = [[row[k] for row in plane] for plane in img.tolist()]
               if len(arrar) > 0:
                    count = 0
                    for row in layer:
                       arrar[count]+=row
                       count += 1
               else:
                   arrar = layer
               k+=1
           arr+=arrar

       arr = np.asarray(arr)
       scipy.misc.imsave(os.path.join(sec_dir, (file.split(".")[0]+".jpg")), arr)
       """
       #a = [[row[0:160] for row in plane] for plane in img.tolist()][0:182]
       #t = np.array(a)
       output = nib.Nifti1Image(final, affine=np.eye(4))
       nib.save(output, os.path.join(sec_dir, file))
       #print(t.shape)
       print("______")
       """""
