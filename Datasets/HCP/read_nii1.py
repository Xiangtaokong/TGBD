import h5py
import os

import numpy as np
import os
from PIL import Image
import torch

def check_dir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
import numpy as np
import shutil


save_dir1=os.path.join('/home/notebook/data/personal/S9053103/TGBD/Datasets/HCP/fmri_mix5s')
check_dir(save_dir1)

dir_all='/home/notebook/data/personal/S9053103/mind_concept/Dataset/HCP/HCP_fmri'
subj_log='/home/notebook/data/personal/S9053103/TGBD/Brain_decoding/meta_info/log_subj.txt'

with open(subj_log,"r") as log:
    subj_num_list=log.readlines()[0].split('","')

print()

for dir_ in os.listdir(dir_all):
    if '.' not in dir_ and 'fmri' not in dir_:
        dir_path=os.path.join(dir_all,dir_)
        nii_num=len(os.listdir(dir_path))
        if nii_num == 6:
            for subjn in subj_num_list:
                if dir_ in subjn:
                    subj_num=subjn.split('_Subj')[1].replace('"',"")
                    print(subj_num)
            subj_name=dir_
            subj_name="{}_Subj{}".format(subj_name,subj_num)

            for nii in os.listdir(dir_path):
                
                if '2000' in nii:
                    nii_path=os.path.join(dir_path,nii)
                    m=nii.split('VIE')[1].split('_')[0]
                    print(nii_path)
                    nii_file = nib.load(nii_path)
                    # Access the data array
                    data = nii_file.get_fdata()
                    # Do further processing or analysis with the data
                    # For example, you can access specific voxels using indexing:
                    voxel_value = data[0, 0, 0]
                    _,_,_,b=data.shape
                    print(data.shape)

                    for i in range(b-6):

                        save_path1=os.path.join(save_dir1,"{}_M{}_{}.npy".format(subj_name,m,i))
                        current_fmri=data[:,:,:,i]
                        mix5s=(data[:,:,:,i]/5)+(data[:,:,:,i+1]/5)+(data[:,:,:,i+2]/5)+(data[:,:,:,i+3]/5)+(data[:,:,:,i+4]/5)

                        np.save(save_path1,mix5s)

        else:
            #print(dir_path)
            pass


#/home/notebook/data/personal/S9053103/mind_concept/Dataset/HCP/hcp_95/fmri_mix5s/100610_Subj1_M1_0.npy

