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


class Train_Data_HCP(data.Dataset):
    def __init__(self, train=True, **kargs):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.kargs = kargs

        self.check_files()
        self.process_img=_transform(224)
        self.global_avg_pool_pre = nn.AvgPool1d(kernel_size=8, stride=8)



    def check_files(self):
        with open(self.kargs['train_data_list']) as fin:
            self.path_list = [line.strip() for line in fin]
            self.path_list = self.path_list
        # self.path_list=self.path_list[0:1000]
        print('Train_Data')
        print(len(self.path_list))

    def __len__(self):
        return len(self.path_list)


    def __getitem__(self, idx):
        fmri_dirs=self.kargs['fmri_data_path']
        img_dirs=self.kargs['img_data_path']

        path = self.path_list[idx]
        filename = path

        img_path = os.path.join(img_dirs,filename+'.png')
        img = Image.open(img_path)
        img = self.process_img(img)

        subj=random.choice(self.kargs['train_subjs'])
        if self.kargs['fmri_type'] == 'after4':
            filenum = filename.split('_')[1]
            M=filename.split('_')[0]
            filenum = str(int(filenum)+4)
            filename = M+'_'+filenum
        else:
            pass

        fmri_path = os.path.join(fmri_dirs,"{}_{}.npy".format(subj,filename))
        fmri = torch.from_numpy(np.load(fmri_path)).float()
        fmri = fmri.unsqueeze(0)

        return fmri, img, img_path, fmri_path

class Val_Data_HCP(data.Dataset):
    def __init__(self, train=False, val_subj='',**kargs):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.kargs = kargs
        self.val_subj=val_subj

        self.check_files()
        self.process_img=_transform(224)
        self.global_avg_pool_pre = nn.AvgPool1d(kernel_size=8, stride=8)


    def check_files(self):

        with open(self.kargs['val_data_list']) as fin:
            self.path_list = [line.strip() for line in fin]
            self.path_list = self.path_list
        # self.path_list=self.path_list[0:1000]
        print('Val_Data')
        print(len(self.path_list))

    def __len__(self):
        return len(self.path_list)


    def __getitem__(self, idx):
        fmri_dirs=self.kargs['fmri_data_path']
        img_dirs=self.kargs['img_data_path']

        path = self.path_list[idx]
        filename = path

        img_path = os.path.join(img_dirs,filename+'.png')
        img = Image.open(img_path)
        img = self.process_img(img)

        subj=self.val_subj

        if self.kargs['fmri_type'] == 'after4':
            filenum = filename.split('_')[1]
            M=filename.split('_')[0]
            filenum = str(int(filenum)+4)
            filename = M+'_'+filenum
        else:
            pass

        fmri_path = os.path.join(fmri_dirs,"{}_{}.npy".format(subj,filename))
        fmri = torch.from_numpy(np.load(fmri_path)).float()
        fmri = fmri.unsqueeze(0)

        return fmri, img, img_path, fmri_path

class Test_Data_HCP(data.Dataset):
    def __init__(self, train=False, test_sub='',**kargs):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.kargs = kargs
        self.test_sub=test_sub

        self.check_files()
        self.process_img=_transform(224)
        self.global_avg_pool_pre = nn.AvgPool1d(kernel_size=8, stride=8)


    def check_files(self):

        with open(self.kargs['test_data_list']) as fin:
            self.path_list = [line.strip() for line in fin]
            self.path_list = self.path_list
        # self.path_list=self.path_list[0:1000]
        print('Test_Data')
        print(len(self.path_list))

    def __len__(self):
        return len(self.path_list)


    def __getitem__(self, idx):
        fmri_dirs=self.kargs['fmri_data_path']
        img_dirs=self.kargs['img_data_path']

        path = self.path_list[idx]
        filename = path

        img_path = os.path.join(img_dirs,filename+'.png')
        img = Image.open(img_path)
        img = self.process_img(img)

        subj=self.test_sub

        if self.kargs['fmri_type'] == 'after4':
            filenum = filename.split('_')[1]
            M=filename.split('_')[0]
            filenum = str(int(filenum)+4)
            filename = M+'_'+filenum
        else:
            pass

        fmri_path = os.path.join(fmri_dirs,"{}_{}.npy".format(subj,filename))
        fmri = torch.from_numpy(np.load(fmri_path)).float()
        fmri = fmri.unsqueeze(0)

        return fmri, img, img_path, fmri_path
