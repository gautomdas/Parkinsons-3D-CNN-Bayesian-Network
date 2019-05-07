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
csv_dir = os.path.join(root_dir, 'All_CSV')
init_file = os.path.join(csv_dir, 'all.csv')
work_file = os.path.join(csv_dir, 'UPDRS_Pre-Dose.csv')

#write to
sec_dir = os.path.join(root_dir, 'Run_CSV')
all_files = os.path.join(sec_dir, 'updrs_Pre_diagnosis.csv')


all_image = []

import csv

results = readcsv(init_file)
pre_dose = readcsv(work_file)

for subdir, dirs, files in os.walk(img_dir):
    counter = 0
    for file in files:
        resut = []

        split_name = file.split(".")
        split_n = split_name[0].split("_")

        id_init = split_n[len(split_n)-1]
        id_fin = id_init[1:]
        row = findVisit(results, id_fin)

        if(not row is None):
            visit = int(findVisit(results, id_fin)[5])
            name = findVisit(results, id_fin)[1]

            rating = findHY(pre_dose, name)
            if(not rating is None and visit<16):
                rate = rating[visit+3]
                print(rate)
                resut = [file, rate]
        if(len(resut)>0):
            all_image.append(resut)

myFile = open(all_files,'w' ,newline='')
print(all_image)

with myFile:
    #write 2d array to csv
    print("a")
    writer = csv.writer(myFile)
    writer.writerows(all_image)
    print("b")