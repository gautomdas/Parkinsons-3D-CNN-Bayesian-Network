import os
import csv
import nibabel as nib


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


root_dir = os.path.abspath('./')
new = "/usr/local/fsl/data/standard"
img_dir = os.path.join(root_dir, 'Full_Size_No_Brain')


for subdir, dirs, files in os.walk(img_dir):
    counter = 0
    for file in files:
        print(file)
        example_filename = os.path.join(img_dir, file)
        img = nib.load(example_filename)
        print(img.shape)
        print(img.get_data_dtype())
        print("___________")