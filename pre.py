import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import nibabel as nib
import cv2
import os
'''
v_dir = './data/brain/train/volume/'
g_dir = './data/brain/train/GT/'
v_names = os.listdir(v_dir)
g_names = os.listdir(g_dir)
print(len(v_names))
print(len(g_names))
j=0
for v_name,g_name in zip(v_names,g_names):
    nii_image=nib.load(v_dir+v_name).get_fdata()
    gt_image=nib.load(g_dir+g_name).get_fdata()
   
    #print(nii_image.shape)
    _,_,slices,_ = nii_image.shape
    
    for i in range(slices):
        if np.sum(gt_image[:,:,i])>0:
            j=j+1
            myslice1 = nii_image[:,:,i,0]
            myslice1 = (myslice1-myslice1.min())/(myslice1.max() - myslice1.min())

            myslice2 = nii_image[:,:,i,1]
            myslice2 = (myslice2-myslice2.min())/(myslice2.max() - myslice2.min())

            myslice3 = nii_image[:,:,i,2]
            myslice3 = (myslice3-myslice3.min())/(myslice3.max() - myslice3.min())

            myslice4 = nii_image[:,:,i,3]
            myslice4 = (myslice4-myslice4.min())/(myslice4.max() - myslice4.min())

            cv2.imwrite('./data/brain/train/images/m1/'+v_name+str(i)+'.jpg',myslice1*255)
            cv2.imwrite('./data/brain/train/images/m2/'+v_name+str(i)+'.jpg',myslice2*255)
            cv2.imwrite('./data/brain/train/images/m3/'+v_name+str(i)+'.jpg',myslice3*255)
            cv2.imwrite('./data/brain/train/images/m4/'+v_name+str(i)+'.jpg',myslice4*255)
            cv2.imwrite('./data/brain/train/masks/'+v_name+str(i)+'.png',gt_image[:,:,i])
        else:
            pass
print(j)
'''

v_dir = './data/brain/train/masks/'
v_names = os.listdir(v_dir)

with open("./data/brain/train/train.txt","w") as f:
    for v_name in v_names:
        f.write(v_name[:-4]+'\n') 