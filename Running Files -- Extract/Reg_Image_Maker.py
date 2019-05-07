import os
import nibabel as nib
import numpy as np
import scipy.misc
import scipy.ndimage

root_dir = os.path.abspath('./')
img_dir = os.path.join(root_dir, 'F_Control_Image_Data')
output_dir = os.path.join(root_dir, 'C_Control_Compiled_Images')

desired_shape = [300, 300, 80]

for subdir, dirs, files in os.walk(img_dir):
    counter = 0
    for file in files:
        current_image = nib.load(os.path.join(img_dir, file))

        a = np.array(current_image.dataobj)
        a = a[:, :, :, 0]
        k = a.shape
        if (k[2] > 100):
            a = np.rot90(a, axes=(1, 2))
        a = np.rot90(a, axes = (0, 2))
        if (a.shape[0] > desired_shape[0] and False):
            scaleVal = (desired_shape[0])/(a.shape[1])
            a = scipy.ndimage.zoom(a, scaleVal)

        print(a.shape)
        fName = file.split(".")[0]+".jpg"
        print(fName)
        halfThr =int ((a.shape[0])/2)
        twoD = a[halfThr]
        print(twoD.shape)
        scipy.misc.imsave(os.path.join(output_dir, fName), twoD)
        """
        a1 = (desired_shape[0] - a.shape[0])
        a2 = (desired_shape[1] - a.shape[1])
        a3 = (desired_shape[2] - a.shape[2])

        b = np.pad(a, ((0, a1), (0, a2), (0, a3)), mode='constant')
        """
        counter += 1


