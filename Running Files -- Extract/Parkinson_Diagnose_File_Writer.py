import os
import csv

root_dir = os.path.abspath('./')
img_dir = os.path.join(root_dir, 'B_PM5_Compiled_Images')
control_img_dir = os.path.join(root_dir, 'D_Control_PM5_Compiled_Images')


all_image = os.path.join(root_dir, 'all_PM5_image.csv')



def padZero(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 10
    vector[-pad_width[1]:] = 10
    return vector

all_files = []

for subdir, dirs, files in os.walk(img_dir):
    counter = 0
    for file in files:
        current_array = []

        apple =  str(file)

        file_input = file.split("_")
        flag = True
        for no_one_cares in file_input:
            if(no_one_cares.lower() == "cor"):
                flag = False

        if(flag):
            outputFile = [apple, 1]
        print(counter)
        counter+=1
        all_files.append(outputFile)

for subdir, dirs, files in os.walk(control_img_dir):
    counter = 0
    for file in files:
        current_array = []

        apple =  str(file)

        file_input = file.split("_")
        flag = True
        for no_one_cares in file_input:
            if(no_one_cares.lower() == "cor"):
                flag = False

        if(flag):
            outputFile = [apple, 0]
        print(counter)
        counter+=1
        all_files.append(outputFile)

print("Finish")

myFile = open(all_image,'w' ,newline='')
with myFile:
    #write 2d array to csv
    print("a")
    writer = csv.writer(myFile)
    writer.writerows(all_files)
    print("b")
