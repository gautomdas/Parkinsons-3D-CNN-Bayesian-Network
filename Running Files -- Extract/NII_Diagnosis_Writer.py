import os
import csv


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
img_dir = os.path.join(root_dir, 'Regular')
sec_dir = os.path.join(root_dir, 'Control')
csv_dir = os.path.join(root_dir, 'Run_CSV')
init_file = os.path.join(csv_dir, 'positive_negative.csv')



checked = []

import csv

results = readcsv(init_file)

count = 0
for subdir, dirs, files in os.walk(sec_dir):
    counter = 0
    for file in files:
        new = [file, 0]
        checked.append(new)

for subdir, dirs, files in os.walk(img_dir):
    counter = 0
    for file in files:
        new = [file, 1]
        checked.append(new)



myFile = open(init_file,'w' ,newline='')


with myFile:
    #write 2d array to csv
    print("a")
    writer = csv.writer(myFile)
    writer.writerows(checked)
    print("b")