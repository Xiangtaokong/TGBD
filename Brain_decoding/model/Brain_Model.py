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

import inspect
import torch
import importlib
from torch.nn import functional as F
import pytorch_lightning as pl
import sys
import numpy as np
sys.path.append('/home/notebook/data/personal/S9053103/TGBD/CLIP')
from torch import nn
import clip
from PIL import Image
from torch.cuda.amp import autocast
import datetime
import os
import math
from sklearn.metrics import average_precision_score, roc_auc_score, hamming_loss
from pytorch_lightning.utilities import rank_zero_only

import kornia
from kornia.augmentation.container import AugmentationSequential

from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam, AdamW
import torchvision.transforms as transforms
import random

from arch.network import (
    BrainNetwork_mindeye,BrainNetwork_mindeye_3d,BrainNetwork_1dcnn_3d,BrainNetwork_3dcnn_3d,BrainNetwork_transformer_3d,BrainNetwork_mindeye_3d_1d

)

def avg_list(num_list):
    return sum(num_list) / len(num_list)


def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate_top_k_accuracy( logits, k):
    correct_topk = 0
    for i in range(logits.shape[0]):
        sorted_indices = torch.argsort(logits[i], descending=True)
        if i in sorted_indices[:k]:
            correct_topk += 1
    topk_accuracy = correct_topk / logits.shape[0]
    return topk_accuracy

def calculate_mAP( logits, device):
    APs = []
    for i in range(logits.shape[0]):
        sorted_indices = torch.argsort(logits[i], descending=True).to(device)
        relevant = (sorted_indices == i).nonzero().flatten().to(device)
        if relevant.numel() == 0:
            APs.append(0)
        else:
            precision_at_k = (torch.arange(relevant.size(0), device=device) + 1) / (relevant + 1).float()
            print(torch.arange(relevant.size(0), device=device) + 1)
            AP = precision_at_k.mean().item()
            APs.append(AP)
    mAP = sum(APs) / len(APs)
    return mAP

def calculate_mAP_sk( logits, device):
    APs = []
    for i in range(logits.shape[0]):
        y_true = np.zeros(logits.shape[1])
        y_true[i] = 1  
        y_scores = logits[i].detach().cpu().numpy()
        AP = average_precision_score(y_true, y_scores)
        APs.append(AP)
    mAP = np.mean(APs)
    return mAP

def calculate_auc( logits, device):
    aucs = []
    for i in range(logits.shape[0]):
        y_true = np.zeros(logits.shape[1])
        y_true[i] = 1 
        y_scores = logits[i].detach().cpu().numpy()
        auc = roc_auc_score(y_true, y_scores)
        aucs.append(auc)
    return np.mean(aucs)

def calculate_hamming_loss( logits, device):
    y_true = np.zeros((logits.shape[0], logits.shape[1]))
    y_pred = np.zeros((logits.shape[0], logits.shape[1]))
    for i in range(logits.shape[0]):
        y_true[i, i] = 1  
        y_pred[i] = (logits[i] > 0.5).detach().cpu().numpy() 
    return hamming_loss(y_true, y_pred)

def load_brainnetwork(network,fmri_type,subj):
    if "_o" in fmri_type:
        pass
    else:
        subj='all'

    if network == 'BrainNetwork_mindeye':
        brain_network = BrainNetwork_mindeye(subj=subj)
    elif network == 'BrainNetwork_mindeye_3d':
        brain_network = BrainNetwork_mindeye_3d(subj=subj)
    elif network == 'BrainNetwork_mindeye_3d_1d':
        brain_network = BrainNetwork_mindeye_3d_1d(subj=subj)
    elif network == 'BrainNetwork_1dcnn_3d':
        brain_network = BrainNetwork_1dcnn_3d(subj=subj)
    elif network == 'BrainNetwork_3dcnn_3d':
        brain_network = BrainNetwork_3dcnn_3d(subj=subj)
    elif network == 'BrainNetwork_transformer_3d':
        brain_network = BrainNetwork_transformer_3d(subj=subj)
    else:
        print(networkerror)


    return brain_network


def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, distributed=False, 
              accelerator=None, local_rank=None, bidirectional=True):
    brain_clip = (preds @ targs.T)/temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    else:
        loss =  F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
        return loss


def mixco(voxels, beta=0.15, s_thresh=0.5):
    perm = torch.randperm(voxels.shape[0])
    voxels_shuffle = voxels[perm].to(voxels.device,dtype=voxels.dtype)
    betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device,dtype=voxels.dtype)
    select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1]*(len(voxels.shape)-1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select

def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))



class Brain_Model(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()

        self.kargs=kargs

        self.model_clip, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.model_clip.eval()

        self.loss_log=0
        self.brain_network=load_brainnetwork(kargs['network'],kargs['fmri_type'],kargs['train_subjs'])

        self.soft_loss_temps= cosine_anneal(0.004, 0.0075, self.kargs['max_epoch'] - int(0.33 * self.kargs['max_epoch']))


        os.makedirs(self.kargs['default_root_dir'], exist_ok=True)

        self.log_file_path = self.kargs['default_root_dir']+f"/{self.kargs['exp_name']}_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        self.log_file = None
        self.setup_log_file()

        self.test_fmri_embeddings = []
        self.test_img_embeddings = []
        self.test_fmri_paths = []
        self.test_img_paths = []
        self.mse_loss = torch.nn.MSELoss()
        try:
            for subj in self.kargs['val_subjs']:
                self.test_fmri_embeddings.append([])
                self.test_img_embeddings.append([])
                self.test_fmri_paths.append([])
                self.test_img_paths.append([])
        except:
            for subj in self.kargs['test_subjs']:
                self.test_fmri_embeddings.append([])
                self.test_img_embeddings.append([])
                self.test_fmri_paths.append([])
                self.test_img_paths.append([])
        
        self.HCP_top1_accuracy_list_indis=[]
        self.HCP_top3_accuracy_list_indis=[]
        self.HCP_top5_accuracy_list_indis=[]
        self.HCP_top10_accuracy_list_indis=[]
        self.HCP_mAP_list_indis=[]
        self.HCP_AUC_list_indis=[]
        self.HCP_Hamming_list_indis=[]

        self.HCP_top1_accuracy_list_outdis=[]
        self.HCP_top3_accuracy_list_outdis=[]
        self.HCP_top5_accuracy_list_outdis=[]
        self.HCP_top10_accuracy_list_outdis=[]
        self.HCP_mAP_list_outdis=[]
        self.HCP_AUC_list_outdis=[]
        self.HCP_Hamming_list_outdis=[]


        self.img_augment = AugmentationSequential(
            kornia.augmentation.RandomResizedCrop((224,224), (0.6,1), p=0.3),
            kornia.augmentation.Resize((224, 224)),
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3),
            kornia.augmentation.RandomGrayscale(p=0.3),
        )

    @rank_zero_only
    def setup_log_file(self):
        self.log_file = open(self.log_file_path, 'w')

    def configure_optimizers(self):
        if self.kargs['lr_scheduler'] == 'OneCycleLR':
            if self.kargs['optimizer'] == 'Adam':
                optimizer = Adam(self.brain_network.parameters(), lr=float(float(self.kargs['lr'])/25), weight_decay=float(self.kargs['weight_decay']))
            elif self.kargs['optimizer'] == 'AdamW' :
                optimizer = AdamW(self.brain_network.parameters(), lr=float(float(self.kargs['lr'])/25), weight_decay=float(self.kargs['weight_decay']))
        else:
            if self.kargs['optimizer'] == 'Adam':
                optimizer = Adam(self.brain_network.parameters(), lr=float(self.kargs['lr']), weight_decay=float(self.kargs['weight_decay']))
            elif self.kargs['optimizer'] == 'AdamW' :
                optimizer = AdamW(self.brain_network.parameters(), lr=float(self.kargs['lr']), weight_decay=float(self.kargs['weight_decay']))
        

        if self.kargs['lr_scheduler'] == 'OneCycleLR':
                    ############################################################################################################
            with open(self.kargs['train_data_list']) as fin:
                if self.kargs['dataset_type'] =='NSD':
                    self.path_list = [line.strip() for line in fin if any(subject in line for subject in self.kargs['train_subjs'])]
                else:
                    self.path_list = [line.strip() for line in fin]

            print('Train_Data')
            train_num=len(self.path_list)
            print(len(self.path_list))

            steps_per_epoch=math.ceil(train_num/self.kargs['batch_size'])
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(self.kargs['lr']),
                steps_per_epoch=steps_per_epoch, 
                epochs=self.kargs['max_epoch'],
                pct_start=0.1,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1e4,
            )
            for param in self.model_clip.parameters():
                param.requires_grad = False

            return [optimizer]

        else:
            scheduler = StepLR(optimizer, step_size=int(self.kargs['max_epoch']/4), gamma=0.5)
            for param in self.model_clip.parameters():
                param.requires_grad = False

            return [optimizer],[scheduler]

    def forward(self, fmri, img):

        img_emb,img_features = self.model_clip.encode_image(img)
        _,fmri_features = self.brain_network.encode_fmri(fmri)
        logits_per_img, logits_per_fmri = self.model_clip(img_features,fmri_features,'fea')


        return img_features,fmri_features,logits_per_img, logits_per_fmri

        
    def training_step(self, batch, batch_idx):
        fmri, img, img_path, fmri_path = batch

        if self.kargs['img_aug'] == True:
            img = self.img_augment(img)
        else:
            pass
        
        if self.kargs['loss_type']=='CLIPLoss':

            img_features,fmri_features,logits_per_img, logits_per_fmri = self(fmri, img)

            labels_gt = torch.arange(logits_per_img.shape[0], dtype=torch.long).to(self.device)

            contrastive_loss = (
                F.cross_entropy(logits_per_img, labels_gt) +
                F.cross_entropy(logits_per_fmri, labels_gt)
            ) / 2

            loss=contrastive_loss
        
        elif self.kargs['loss_type']=='BiMixCo_SoftCLIP':

            if self.current_epoch < int(0.33 * self.kargs['max_epoch']):
                fmri, perm, betas, select = mixco(fmri)

            img_features,fmri_features,logits_per_img, logits_per_fmri = self(fmri, img)

            clip_voxels_norm = nn.functional.normalize(fmri_features.flatten(1), dim=-1)
            clip_target_norm = nn.functional.normalize(img_features.flatten(1), dim=-1)


            if self.current_epoch < int(0.33 * self.kargs['max_epoch']):
                loss_nce = mixco_nce(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=.006, 
                    perm=perm, betas=betas, select=select)
                loss=loss_nce
            else:
                epoch_temp = self.soft_loss_temps[self.current_epoch - int(0.33*self.kargs['max_epoch'])]
                loss_nce = soft_clip_loss(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=epoch_temp)
                loss=loss_nce

        elif self.kargs['loss_type']=='MSE':

            img_features,fmri_features,logits_per_img, logits_per_fmri = self(fmri, img)
            mse_loss=self.mse_loss(img_features,fmri_features)
            loss=mse_loss


        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=len(batch),sync_dist=True)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=False, prog_bar=True, logger=True,batch_size=len(batch))

        self.loss_log = loss
        torch.cuda.empty_cache()

        return loss

    @rank_zero_only
    def on_train_start(self) -> None:

        self.log_file.write("Hyperparameters:\n")
        for key, value in self.hparams.items():
            self.log_file.write(f"{key}: {value}\n")
        self.log_file.write("\n\n")

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total_params = trainable_params + non_trainable_params
        
        params_info = f"""
        ---------------------------------------------------------
        Trainable params: {int(trainable_params/1000000)}M
        Non-trainable params: {int(non_trainable_params/1000000)}M
        Total params: {int(total_params/1000000)}M
        ---------------------------------------------------------
        """

        self.log_file.write("Model Summary:\n")
        self.log_file.write(params_info)
        self.log_file.write(self.model_summary())
        self.log_file.write("\n\n")
        self.log_file.flush()


    @rank_zero_only
    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0) -> None:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        if self.kargs['lr_scheduler'] == 'OneCycleLR':
            self.scheduler.step()

        self.log_file.write(f'Epoch: {self.current_epoch}, Batch: {batch_idx}, Time: {current_time}, Loss: {self.loss_log.item()}, LR: {lr}\n')
        self.log_file.flush()
        
    @rank_zero_only    
    def on_train_end(self) -> None:
        self.log_file.close()
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss=0
        id_=0
        for subj in self.kargs['val_subjs']:

            if dataloader_idx == id_:
                
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                fmri, img, img_path, fmri_path = batch
                img_features,fmri_features,logits_per_img, logits_per_fmri = self(fmri, img)

                self.test_fmri_embeddings[id_].append(fmri_features)
                self.test_img_embeddings[id_].append(img_features)

                self.test_img_paths[id_].append(img_path)
                self.test_fmri_paths[id_].append(fmri_path)

            id_+=1


        torch.cuda.empty_cache()

        return loss


    def on_validation_epoch_end(self):

        id_=0
        for subj in self.kargs['val_subjs']:

            # merge batches
            all_fmri_embeddings_ = torch.cat(self.test_fmri_embeddings[id_],dim=0)
            all_img_embeddings_ = torch.cat(self.test_img_embeddings[id_],dim=0)

            # to main GPU card
            all_fmri_embeddings_ = self.all_gather(all_fmri_embeddings_)
            all_img_embeddings_ = self.all_gather(all_img_embeddings_)

            # combine data from different GPU
            all_fmri_embeddings_ = torch.cat([x for x in all_fmri_embeddings_], dim=0)
            all_img_embeddings_ = torch.cat([x for x in all_img_embeddings_], dim=0)

            if self.trainer.global_rank == 0:

                all_fmri_embeddings = all_fmri_embeddings_
                all_img_embeddings = all_img_embeddings_

                print('all_fmri_embeddings',all_fmri_embeddings.shape)

                if self.kargs['dataset_type'] == "NSD":
                    logits_per_img, logits_per_fmri = self.model_clip(all_img_embeddings[:300].float(),all_fmri_embeddings[:300].float(),'fea')
                else:
                    logits_per_img, logits_per_fmri = self.model_clip(all_img_embeddings.float(),all_fmri_embeddings.float(),'fea')

                logits_per_fmri_sigmod=torch.sigmoid(logits_per_fmri)

                top1_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=1)
                top3_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=3)
                top5_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=5)
                top10_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=10)
                
                mAP = calculate_mAP_sk(logits_per_fmri, device=self.device)

                AUC = calculate_auc(logits_per_fmri_sigmod, device=self.device)
                Hamming = calculate_hamming_loss(logits_per_fmri_sigmod, device=self.device)

                self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n")
                self.log_file.write("Testing on "+subj+"\n")
                self.log_file.write(f'test_top1_accuracy: {top1_accuracy} \n')
                self.log_file.write(f'test_top3_accuracy: {top3_accuracy} \n')
                self.log_file.write(f'test_top5_accuracy: {top5_accuracy} \n')
                self.log_file.write(f'test_top10_accuracy: {top10_accuracy} \n')
                self.log_file.write(f'test_mAP: {mAP} \n')
                self.log_file.write(f'test_AUC: {AUC} \n')
                self.log_file.write(f'test_Hamming: {Hamming} \n')
                self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n")

                self.log_file.flush()
                
                print("Testing on "+subj)
                print('test_top1_accuracy', top1_accuracy)
                print('test_top3_accuracy', top3_accuracy)
                print('test_top5_accuracy', top5_accuracy)
                print('test_top10_accuracy', top10_accuracy)
                print('test_mAP', mAP)
                print('test_AUC', AUC)
                print('test_Hamming', Hamming)

            self.test_fmri_embeddings[id_] = []
            self.test_img_embeddings[id_] = []

            self.test_img_paths[id_]=[]
            self.test_fmri_paths[id_]=[]
            
            id_+=1

    def test_step(self, batch, batch_idx, dataloader_idx=0):

        if self.kargs['dataset_type'] == "NSD":
            id_=0
            for subj in self.kargs['test_subjs']:

                if dataloader_idx == id_:
                    
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    fmri, img, img_path, fmri_path = batch
                    img_features,fmri_features,logits_per_img, logits_per_fmri = self(fmri, img)

                    self.test_fmri_embeddings[id_].append(fmri_features)
                    self.test_img_embeddings[id_].append(img_features)

                    self.test_img_paths[id_].append(img_path)
                    self.test_fmri_paths[id_].append(fmri_path)

                id_+=1
        else:
            id_=0
            for subj in self.kargs['test_subjs']:

                if dataloader_idx == id_:
                    
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    fmri, img, img_path, fmri_path = batch
                    img_features,fmri_features,logits_per_img, logits_per_fmri = self(fmri, img)

                    # self.test_fmri_embeddings[id_].append(fmri_features)
                    # self.test_img_embeddings[id_].append(img_features)

                    # self.test_img_paths[id_].append(img_path)
                    # self.test_fmri_paths[id_].append(fmri_path)

                    logits_per_img, logits_per_fmri = self.model_clip(img_features.float(),fmri_features.float(),'fea')

                    logits_per_fmri_sigmod=torch.sigmoid(logits_per_fmri)

                    top1_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=1)
                    top3_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=3)
                    top5_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=5)
                    top10_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=10)
                    
                    mAP = calculate_mAP_sk(logits_per_fmri, device=self.device)

                    AUC = calculate_auc(logits_per_fmri_sigmod, device=self.device)
                    Hamming = calculate_hamming_loss(logits_per_fmri_sigmod, device=self.device)

                    self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n")
                    self.log_file.write("Testing on "+subj+"\n")
                    self.log_file.write(f'test_top1_accuracy: {top1_accuracy} \n')
                    self.log_file.write(f'test_top3_accuracy: {top3_accuracy} \n')
                    self.log_file.write(f'test_top5_accuracy: {top5_accuracy} \n')
                    self.log_file.write(f'test_top10_accuracy: {top10_accuracy} \n')
                    self.log_file.write(f'test_mAP: {mAP} \n')
                    self.log_file.write(f'test_AUC: {AUC} \n')
                    self.log_file.write(f'test_Hamming: {Hamming} \n')
                    self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n")

                    self.log_file.flush()
                    
                    print("Testing on "+subj)
                    print('test_top1_accuracy', top1_accuracy)
                    print('test_top3_accuracy', top3_accuracy)
                    print('test_top5_accuracy', top5_accuracy)
                    print('test_top10_accuracy', top10_accuracy)
                    print('test_mAP', mAP)
                    print('test_AUC', AUC)
                    print('test_Hamming', Hamming)

                    ####################################################################################################################################################################
                    # save top5 image

                    # top5_indices = logits_per_fmri.topk(5, dim=1).indices


                    # for i in range(100):
                    #     correct_img = Image.open(img_path[i])

                    #     top5_imgs = [Image.open(img_path[idx]) for idx in top5_indices[i]]
                    #     combined_img = Image.new('RGB', ((correct_img.width + 20 )* 6+60, correct_img.height),"white")
                    #     combined_img.paste(correct_img, (0, 0))
                    #     for j, img in enumerate(top5_imgs):
                    #         combined_img.paste(img, ((j + 1) * (correct_img.width + 20 ) + 60, 0))

                    #     save_dir= f"{self.kargs['default_root_dir']}/{subj}/top5_img_100"
                    #     check_dir(save_dir)

                    #     combined_img_path = f"{self.kargs['default_root_dir']}/{subj}/top5_img_100/top5_images_index_{i}.png"
                    #     combined_img.save(combined_img_path)

                    #     with open(f"{self.kargs['default_root_dir']}/{subj}/top5_img_100/image_paths.txt", 'a') as f:
                    #         f.write(f"Correct Image: {os.path.basename(img_path[i])}\n")
                    #         top_n=1
                    #         for idx in top5_indices[i]:
                    #             f.write(f"Top-{top_n} Image: {os.path.basename(img_path[idx])}\n")
                    #             top_n+=1
                        
                    ####################################################################################################################################################################



                    if subj in ["100610_Subj1","102311_Subj2","102816_Subj3","104416_Subj4","105923_Subj5","108323_Subj6","109123_Subj7","111312_Subj8","111514_Subj9","114823_Subj10"]:
                        self.HCP_top1_accuracy_list_outdis.append(top1_accuracy)
                        self.HCP_top3_accuracy_list_outdis.append(top3_accuracy)
                        self.HCP_top5_accuracy_list_outdis.append(top5_accuracy)
                        self.HCP_top10_accuracy_list_outdis.append(top10_accuracy)
                        self.HCP_mAP_list_outdis.append(mAP)
                        self.HCP_AUC_list_outdis.append(AUC)
                        self.HCP_Hamming_list_outdis.append(Hamming)
                    else:
                        self.HCP_top1_accuracy_list_indis.append(top1_accuracy)
                        self.HCP_top3_accuracy_list_indis.append(top3_accuracy)
                        self.HCP_top5_accuracy_list_indis.append(top5_accuracy)
                        self.HCP_top10_accuracy_list_indis.append(top10_accuracy)
                        self.HCP_mAP_list_indis.append(mAP)
                        self.HCP_AUC_list_indis.append(AUC)
                        self.HCP_Hamming_list_indis.append(Hamming)

                id_+=1



        torch.cuda.empty_cache()

    def on_test_epoch_end(self):

        id_=0

        for subj in self.kargs['test_subjs']:

            if self.kargs['dataset_type'] == "NSD":

                # merge batches
                all_fmri_embeddings_ = torch.cat(self.test_fmri_embeddings[id_],dim=0)
                all_img_embeddings_ = torch.cat(self.test_img_embeddings[id_],dim=0)

                # to main GPU card
                all_fmri_embeddings_ = self.all_gather(all_fmri_embeddings_)
                all_img_embeddings_ = self.all_gather(all_img_embeddings_)

                # combine data from different GPU
                all_fmri_embeddings_ = torch.cat([x for x in all_fmri_embeddings_], dim=0)
                all_img_embeddings_ = torch.cat([x for x in all_img_embeddings_], dim=0)

                if self.trainer.global_rank == 0:

                    all_fmri_embeddings = all_fmri_embeddings_
                    all_img_embeddings = all_img_embeddings_

                    print('all_fmri_embeddings',all_fmri_embeddings.shape)

                    top1_accuracy_list=[]
                    top3_accuracy_list=[]
                    top5_accuracy_list=[]
                    top10_accuracy_list=[]
                    mAP_list=[]
                    AUC_list=[]
                    Hamming_list=[]

                    for ci in range(30):
                        indices = list(range(len(all_img_embeddings)))
                        sampled_indices = random.sample(indices, 300)

                        all_img_embeddings_sample = torch.stack([all_img_embeddings[i] for i in sampled_indices])
                        all_fmri_embeddings_sample = torch.stack([all_fmri_embeddings[i] for i in sampled_indices])

                        logits_per_img, logits_per_fmri = self.model_clip(all_img_embeddings_sample.float(),all_fmri_embeddings_sample.float(),'fea')

                        logits_per_fmri_sigmod=torch.sigmoid(logits_per_fmri)

                        top1_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=1)
                        top3_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=3)
                        top5_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=5)
                        top10_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=10)
                        
                        mAP = calculate_mAP_sk(logits_per_fmri, device=self.device)

                        AUC = calculate_auc(logits_per_fmri_sigmod, device=self.device)
                        Hamming = calculate_hamming_loss(logits_per_fmri_sigmod, device=self.device)

                        self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n")
                        self.log_file.write("Testing times: "+str(ci)+"\n")               
                        self.log_file.write("Testing on "+subj+"\n")
                        self.log_file.write(f'test_top1_accuracy: {top1_accuracy} \n')
                        self.log_file.write(f'test_top3_accuracy: {top3_accuracy} \n')
                        self.log_file.write(f'test_top5_accuracy: {top5_accuracy} \n')
                        self.log_file.write(f'test_top10_accuracy: {top10_accuracy} \n')
                        self.log_file.write(f'test_mAP: {mAP} \n')
                        self.log_file.write(f'test_AUC: {AUC} \n')
                        self.log_file.write(f'test_Hamming: {Hamming} \n')
                        self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n\n")

                        top1_accuracy_list.append(top1_accuracy)
                        top3_accuracy_list.append(top3_accuracy)
                        top5_accuracy_list.append(top5_accuracy)
                        top10_accuracy_list.append(top10_accuracy)
                        mAP_list.append(mAP)
                        AUC_list.append(AUC)
                        Hamming_list.append(Hamming)

                        self.log_file.flush()
                        
                        print("Testing on "+subj)
                        print('test_top1_accuracy', top1_accuracy)
                        print('test_top3_accuracy', top3_accuracy)
                        print('test_top5_accuracy', top5_accuracy)
                        print('test_top10_accuracy', top10_accuracy)
                        print('test_mAP', mAP)
                        print('test_AUC', AUC)
                        print('test_Hamming', Hamming)


                    self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n")          
                    self.log_file.write("Final Average Results on 300 "+subj+"\n")
                    self.log_file.write(f'test_top1_accuracy: {avg_list(top1_accuracy_list):.2f} \n')
                    self.log_file.write(f'test_top3_accuracy: {avg_list(top3_accuracy_list):.2f} \n')
                    self.log_file.write(f'test_top5_accuracy: {avg_list(top5_accuracy_list):.2f} \n')
                    self.log_file.write(f'test_top10_accuracy: {avg_list(top10_accuracy_list):.2f} \n')
                    self.log_file.write(f'test_mAP: {avg_list(mAP_list)} \n')
                    self.log_file.write(f'test_AUC: {avg_list(AUC_list)} \n')
                    self.log_file.write(f'test_Hamming: {avg_list(Hamming_list)} \n')
                    self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n\n")


                    top1_accuracy_list=[]
                    top3_accuracy_list=[]
                    top5_accuracy_list=[]
                    top10_accuracy_list=[]
                    mAP_list=[]
                    AUC_list=[]
                    Hamming_list=[]

                    for ci in range(30):
                        indices = list(range(len(all_img_embeddings)))
                        sampled_indices = random.sample(indices, 100)

                        all_img_embeddings_sample = torch.stack([all_img_embeddings[i] for i in sampled_indices])
                        all_fmri_embeddings_sample = torch.stack([all_fmri_embeddings[i] for i in sampled_indices])

                        logits_per_img, logits_per_fmri = self.model_clip(all_img_embeddings_sample.float(),all_fmri_embeddings_sample.float(),'fea')

                        logits_per_fmri_sigmod=torch.sigmoid(logits_per_fmri)

                        top1_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=1)
                        top3_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=3)
                        top5_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=5)
                        top10_accuracy = calculate_top_k_accuracy(logits_per_fmri, k=10)
                        
                        mAP = calculate_mAP_sk(logits_per_fmri, device=self.device)

                        AUC = calculate_auc(logits_per_fmri_sigmod, device=self.device)
                        Hamming = calculate_hamming_loss(logits_per_fmri_sigmod, device=self.device)

                        self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n")
                        self.log_file.write("Testing times: "+str(ci)+"\n")               
                        self.log_file.write("Testing on "+subj+"\n")
                        self.log_file.write(f'test_top1_accuracy: {top1_accuracy} \n')
                        self.log_file.write(f'test_top3_accuracy: {top3_accuracy} \n')
                        self.log_file.write(f'test_top5_accuracy: {top5_accuracy} \n')
                        self.log_file.write(f'test_top10_accuracy: {top10_accuracy} \n')
                        self.log_file.write(f'test_mAP: {mAP} \n')
                        self.log_file.write(f'test_AUC: {AUC} \n')
                        self.log_file.write(f'test_Hamming: {Hamming} \n')
                        self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n\n")

                        top1_accuracy_list.append(top1_accuracy)
                        top3_accuracy_list.append(top3_accuracy)
                        top5_accuracy_list.append(top5_accuracy)
                        top10_accuracy_list.append(top10_accuracy)
                        mAP_list.append(mAP)
                        AUC_list.append(AUC)
                        Hamming_list.append(Hamming)

                        self.log_file.flush()
                        
                        print("Testing on "+subj)
                        print('test_top1_accuracy', top1_accuracy)
                        print('test_top3_accuracy', top3_accuracy)
                        print('test_top5_accuracy', top5_accuracy)
                        print('test_top10_accuracy', top10_accuracy)
                        print('test_mAP', mAP)
                        print('test_AUC', AUC)
                        print('test_Hamming', Hamming)


                    self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n")          
                    self.log_file.write("Final Average Results on 100 "+subj+"\n")
                    self.log_file.write(f'test_top1_accuracy: {avg_list(top1_accuracy_list):.2f} \n')
                    self.log_file.write(f'test_top3_accuracy: {avg_list(top3_accuracy_list):.2f} \n')
                    self.log_file.write(f'test_top5_accuracy: {avg_list(top5_accuracy_list):.2f} \n')
                    self.log_file.write(f'test_top10_accuracy: {avg_list(top10_accuracy_list):.2f} \n')
                    self.log_file.write(f'test_mAP: {avg_list(mAP_list)} \n')
                    self.log_file.write(f'test_AUC: {avg_list(AUC_list)} \n')
                    self.log_file.write(f'test_Hamming: {avg_list(Hamming_list)} \n')
                    self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n\n")

                self.test_fmri_embeddings[id_] = []
                self.test_img_embeddings[id_] = []

                self.test_img_paths[id_]=[]
                self.test_fmri_paths[id_]=[]
                
                id_+=1

        if self.kargs['dataset_type'] == "HCP":
            self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n")          
            self.log_file.write("Final Average Results on outdis (1-10) "+"\n")
            self.log_file.write(f'test_top1_accuracy: {avg_list(self.HCP_top1_accuracy_list_outdis):.2f} \n')
            self.log_file.write(f'test_top3_accuracy: {avg_list(self.HCP_top3_accuracy_list_outdis):.2f} \n')
            self.log_file.write(f'test_top5_accuracy: {avg_list(self.HCP_top5_accuracy_list_outdis):.2f} \n')
            self.log_file.write(f'test_top10_accuracy: {avg_list(self.HCP_top10_accuracy_list_outdis):.2f} \n')
            self.log_file.write(f'test_mAP: {avg_list(self.HCP_mAP_list_outdis)} \n')
            self.log_file.write(f'test_AUC: {avg_list(self.HCP_AUC_list_outdis)} \n')
            self.log_file.write(f'test_Hamming: {avg_list(self.HCP_Hamming_list_outdis)} \n')
            self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n\n")
            
            self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n")          
            self.log_file.write("Final Average Results on indis (11-177) "+"\n")
            self.log_file.write(f'test_top1_accuracy: {avg_list(self.HCP_top1_accuracy_list_indis):.2f} \n')
            self.log_file.write(f'test_top3_accuracy: {avg_list(self.HCP_top3_accuracy_list_indis):.2f} \n')
            self.log_file.write(f'test_top5_accuracy: {avg_list(self.HCP_top5_accuracy_list_indis):.2f} \n')
            self.log_file.write(f'test_top10_accuracy: {avg_list(self.HCP_top10_accuracy_list_indis):.2f} \n')
            self.log_file.write(f'test_mAP: {avg_list(self.HCP_mAP_list_indis)} \n')
            self.log_file.write(f'test_AUC: {avg_list(self.HCP_AUC_list_indis)} \n')
            self.log_file.write(f'test_Hamming: {avg_list(self.HCP_Hamming_list_indis)} \n')
            self.log_file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"+"\n\n")


    def model_summary(self):
        return str(self)


