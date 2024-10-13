import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from data.NSD_Datasets import Train_Data_NSD,Val_Data_NSD,Test_Data_NSD
from data.HCP_Datasets import Train_Data_HCP,Val_Data_HCP,Test_Data_HCP

class Brain_Dataloader(pl.LightningDataModule):

    def __init__(self, num_workers=0, stage=0,
                 **kargs):
        super().__init__()
        self.num_workers = num_workers
        self.kargs = kargs

        self.setup(stage)


    def setup(self, stage=None):
        if self.kargs['dataset_type'] == "HCP":
            if stage == 'fit' or stage is None:
                # Assign train/val datasets for use in dataloaders
                self.trainset = Train_Data_HCP(train=True,**self.kargs)
                self.valsets={}
                for subj in self.kargs['val_subjs']:
                    valset = Val_Data_HCP(train=False,val_subj=subj,**self.kargs)
                    self.valsets[subj]=valset

            # Assign test dataset for use in dataloader(s)
            if stage == 'test' or stage is None:
                self.testsets={}
                for subj in self.kargs['test_subjs']:
                    testset = Test_Data_HCP(train=False,test_sub=subj,**self.kargs)
                    self.testsets[subj]=testset


        elif self.kargs['dataset_type'] == "NSD":
            # Assign train/val datasets for use in dataloaders
            if stage == 'fit' or stage is None:
                # Assign train/val datasets for use in dataloaders
                self.trainset = Train_Data_NSD(train=True,**self.kargs)
                self.valsets={}
                for subj in self.kargs['val_subjs']:
                    valset = Val_Data_NSD(train=False,val_subj=subj,**self.kargs)
                    self.valsets[subj]=valset

            # Assign test dataset for use in dataloader(s)
            if stage == 'test' or stage is None:
                self.testsets={}
                for subj in self.kargs['test_subjs']:
                    testset = Test_Data_NSD(train=False,test_subj=subj,**self.kargs)
                    self.testsets[subj]=testset

        else:
            print(wrongdateset)


    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.kargs['batch_size'], num_workers=self.num_workers, shuffle=True,persistent_workers=True)

    def val_dataloader(self):
        return [DataLoader(self.valsets[subj], batch_size=self.kargs['batch_size'], num_workers=self.num_workers, shuffle=False) for subj in self.kargs['val_subjs']]

    def test_dataloader(self):
        return [DataLoader(self.testsets[subj], batch_size=self.kargs['batch_size'], num_workers=self.num_workers, shuffle=False) for subj in self.kargs['test_subjs']]
