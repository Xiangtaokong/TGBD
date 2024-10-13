# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import torch
import os.path as op
import os
import numpy as np
import pickle as pkl
import torch.utils.data as data
import torch.nn.functional as F
from torch import nn

from torchvision import transforms
from sklearn.model_selection import train_test_split

import h5py
from PIL import Image


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize



def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class Train_Data_NSD(data.Dataset):
    def __init__(self, train=True, **kargs):
        # Set all input args as attributes
        self.__dict__.update(locals())

        self.kargs = kargs

        self.check_files()
        self.process_img=_transform(224)
        self.global_avg_pool_pre = nn.AvgPool1d(kernel_size=8, stride=8)


    def check_files(self):
 
        with open(self.kargs['train_data_list']) as fin:
            self.path_list = [line.strip() for line in fin if any(subject in line for subject in self.kargs['train_subjs'])]
        
        # self.path_list=self.path_list[0:1000]

        print('Train_Data')
        print(len(self.path_list))

    def __len__(self):
        return len(self.path_list)


    def __getitem__(self, idx):
        if 'vision' in self.kargs['fmri_type']:
            fmri_dirs=os.path.join(self.kargs['data_path'],'train','fmri_general')
        else:
            fmri_dirs=os.path.join(self.kargs['data_path'],'train','fmri_whole')

        img_dirs=os.path.join(self.kargs['data_path'],'train','imgs')

        path = self.path_list[idx]
        filename = path

        img_path = os.path.join(img_dirs,filename.split('_')[1].split('_')[0]+'.png')
        fmri_path = os.path.join(fmri_dirs,filename+'.npy')


        img = Image.open(img_path)
        img = self.process_img(img)

        fmri = torch.from_numpy(np.load(fmri_path)).float()

        if self.kargs['fmri_type'] == 'whole':
            # 使用F.interpolate函数进行大小调整
            # F.interpolate 需要输入形状为 (batch_size, channels, depth, height, width)，因此需要先调整输入形状
            fmri = fmri.unsqueeze(0).unsqueeze(0) # 增加batch和channel维度
            fmri = F.interpolate(fmri, size=(113, 136, 113), mode='trilinear', align_corners=False)
            #fmri = F.interpolate(fmri, size=(84, 106, 85), mode='trilinear', align_corners=False)
            fmri = fmri.squeeze()  # 移除batch和channel维度
            fmri = fmri.unsqueeze(0)
        elif self.kargs['fmri_type'] == 'whole_o':
            fmri = fmri.unsqueeze(0)
        elif self.kargs['fmri_type'] == 'vision':
            size_aug=18000
            fmri = F.interpolate(fmri.unsqueeze(0).unsqueeze(0), size=size_aug, mode='linear', align_corners=True)
            fmri = fmri.squeeze()
            fmri = F.pad(fmri, (0, 18000 - fmri.size(0)))
        
        elif self.kargs['fmri_type'] == 'vision_o':
            pass

        else:
            print(ddd)

        return fmri, img, img_path, fmri_path

class Val_Data_NSD(data.Dataset):
    def __init__(self, train=False, val_subj='',**kargs):
        # Set all input args as attributes
        self.__dict__.update(locals())

        self.kargs = kargs
        self.val_subj=val_subj

        self.check_files()
        self.process_img=_transform(224)

    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value. 

        # with open('/home/notebook/data/personal/S9053103/mind_concept/Dataset/NSD/NSD_test.txt') as fin:
        #     self.path_list = [line.replace('\n','') for line in fin]

        with open(self.kargs['val_data_list']) as fin:
            self.path_list = [line.strip() for line in fin if self.val_subj in line]
        
        print('Val_Data')
        print(len(self.path_list))


    def __len__(self):
        return len(self.path_list)


    def __getitem__(self, idx):
        if 'vision' in self.kargs['fmri_type']:
            fmri_dirs=os.path.join(self.kargs['data_path'],'test','fmri_general')
        else:
            fmri_dirs=os.path.join(self.kargs['data_path'],'test','fmri_whole')
        img_dirs=os.path.join(self.kargs['data_path'],'test','imgs')

        path = self.path_list[idx]
        filename = path

        img_path = os.path.join(img_dirs,filename.split('_')[1]+'.png')
        fmri_path = os.path.join(fmri_dirs,filename+'.npy')

        img = Image.open(img_path)
        img = self.process_img(img)

        ########################################################################################3
        #全脑
        fmri = torch.from_numpy(np.load(fmri_path)).float()
        fmri = torch.mean(fmri[0:3, :], dim=0)

        if self.kargs['fmri_type'] == 'whole':
            # 使用F.interpolate函数进行大小调整
            # F.interpolate 需要输入形状为 (batch_size, channels, depth, height, width)，因此需要先调整输入形状
            fmri = fmri.unsqueeze(0).unsqueeze(0) # 增加batch和channel维度
            fmri = F.interpolate(fmri, size=(113, 136, 113), mode='trilinear', align_corners=False)
            fmri = fmri.squeeze()  # 移除batch和channel维度
            fmri = fmri.unsqueeze(0)
        elif self.kargs['fmri_type'] == 'whole_o':
            fmri = fmri.unsqueeze(0)
        elif self.kargs['fmri_type'] == 'vision':
            size_aug=18000
            fmri = F.interpolate(fmri.unsqueeze(0).unsqueeze(0), size=size_aug, mode='linear', align_corners=True)
            fmri = fmri.squeeze()
            fmri = F.pad(fmri, (0, 18000 - fmri.size(0)))
        elif self.kargs['fmri_type'] == 'vision_o':
            pass
        else:
            print(ddd)

        # ##########################################################################################
        return fmri, img, img_path, fmri_path

class Test_Data_NSD(data.Dataset):
    def __init__(self, train=False, test_subj='',**kargs):
        # Set all input args as attributes
        self.__dict__.update(locals())

        self.kargs = kargs
        self.test_subj=test_subj

        self.check_files()
        self.process_img=_transform(224)

    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value. 

        # with open('/home/notebook/data/personal/S9053103/mind_concept/Dataset/NSD/NSD_test.txt') as fin:
        #     self.path_list = [line.replace('\n','') for line in fin]

        with open(self.kargs['test_data_list']) as fin:
            self.path_list = [line.strip() for line in fin if self.test_subj in line]

        print('Test_Data')
        print(len(self.path_list))



    def __len__(self):
        return len(self.path_list)


    def __getitem__(self, idx):
        if 'vision' in self.kargs['fmri_type']:
            fmri_dirs=os.path.join(self.kargs['data_path'],'test','fmri_general')
        else:
            fmri_dirs=os.path.join(self.kargs['data_path'],'test','fmri_whole')
        img_dirs=os.path.join(self.kargs['data_path'],'test','imgs')

        path = self.path_list[idx]
        filename = path

        img_path = os.path.join(img_dirs,filename.split('_')[1]+'.png')
        fmri_path = os.path.join(fmri_dirs,filename+'.npy')

        img = Image.open(img_path)
        img = self.process_img(img)

        ########################################################################################3
        #全脑
        fmri = torch.from_numpy(np.load(fmri_path)).float()
        fmri = torch.mean(fmri[0:3, :], dim=0)

        if self.kargs['fmri_type'] == 'whole':
            # 使用F.interpolate函数进行大小调整
            # F.interpolate 需要输入形状为 (batch_size, channels, depth, height, width)，因此需要先调整输入形状
            fmri = fmri.unsqueeze(0).unsqueeze(0) # 增加batch和channel维度
            fmri = F.interpolate(fmri, size=(113, 136, 113), mode='trilinear', align_corners=False)
            fmri = fmri.squeeze()  # 移除batch和channel维度
            fmri = fmri.unsqueeze(0)
        elif self.kargs['fmri_type'] == 'whole_o':
            fmri = fmri.unsqueeze(0)
        elif self.kargs['fmri_type'] == 'vision':
            size_aug=18000
            fmri = F.interpolate(fmri.unsqueeze(0).unsqueeze(0), size=size_aug, mode='linear', align_corners=True)
            fmri = fmri.squeeze()
            fmri = F.pad(fmri, (0, 18000 - fmri.size(0)))
        elif self.kargs['fmri_type'] == 'vision_o':
            pass
        else:
            print(ddd)

        # ##########################################################################################


        return fmri, img, img_path, fmri_path
