import os
import pytorch_lightning as pl
from argparse import ArgumentParser,Namespace
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger,CSVLogger

from model.Brain_Model import Brain_Model
from data.Brain_Dataloader import Brain_Dataloader
from utils import load_model_path_by_args
from torch.cuda.amp import autocast
import torch
import yaml




def main(args):
    pl.seed_everything(args.seed)

    data_module = Brain_Dataloader(**vars(args))

    model = Brain_Model(**vars(args))


    checkpoint_callback = plc.ModelCheckpoint(
    filename='{epoch}', # 模型文件命名格式
    save_top_k=-1, # 设置为-1表示保存所有周期的模型，如果只想保存最好的N个模型，这里可以设置为N
    every_n_epochs=args.save_every_n_epochs, # 设置模型保存的间隔为每10个epoch
    )

    # 初始化Logger
    tensorboard_logger = TensorBoardLogger(save_dir=args.default_root_dir, name="tb_logger")

    trainer = Trainer(
        strategy='ddp_find_unused_parameters_true',
        accelerator="gpu", 
        devices=args.gpu_num, 
        callbacks=[checkpoint_callback],
        max_epochs=args.max_epoch,
        default_root_dir=args.default_root_dir,
        fast_dev_run=False,
        # limit_train_batches=False, #0.25 4
        # limit_val_batches=False, #0.25 4
        # limit_test_batches=False,
        logger=tensorboard_logger,
        profiler="simple",
        #val_check_interval = 20
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        )

    trainer.fit(model, data_module,ckpt_path=args.ckpt_path)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--opt', default='0', type=str)
    args = parser.parse_args()
    config_file=args.opt
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)


    # 创建 argparse 的 Namespace 对象，并将 YAML 配置加载到该对象
    args = Namespace(**config)

    print(args)
    args.default_root_dir = os.path.join(args.default_root_dir,args.exp_name)

    main(args)
