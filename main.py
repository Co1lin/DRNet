import os
import random
from argparse import ArgumentParser
import hydra
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
import torch
from data.dr_datamodule import DRDataModule
from models.drnet import DRNet

def init_wandb():
    os.environ['WANDB_API_KEY'] = 'a5044d4b533063065587a9fce532ad394071fc48'
    wandb_logger = WandbLogger(
        entity='co1lin',
        project='DRNet',
    )
    return wandb_logger

model: pl.LightningModule = None
datamodule: pl.LightningDataModule = None
ckpt_callback_list: list = None
output_dir: str = None

def train(cfg):
    if not cfg.task.resume:
        ckpt_callback_list = [
            ModelCheckpoint(
                monitor='val_loss',
                dirpath=output_dir,
                filename=f'{cfg.task.name}-' + '{epoch}-{val_loss:.2f}-best',
                save_top_k=3,
                mode='min',
            ),
            ModelCheckpoint(
                dirpath=output_dir,
                filename=f'{cfg.task.name}-' + '{epoch}-{val_loss:.2f}',
            )
        ]
        wandb_logger = init_wandb()
        trainer = pl.Trainer(
            max_epochs=cfg.task.epochs,
            gpus=cfg.gpus, 
            logger=wandb_logger, 
            accelerator='ddp', 
            log_every_n_steps=20, 
            callbacks=ckpt_callback_list,
            #plugins=DDPPlugin(find_unused_parameters=False),
        )
        trainer.fit(model, datamodule=datamodule)
    else:
        trainer = Trainer(
            resume_from_checkpoint=cfg.weight,
        )
        trainer.fit(model)

def test(cfg):
    trainer = pl.Trainer(
        gpus=cfg.gpus,
    )
    trainer.test(model, datamodule=datamodule)


def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path='config', config_name='config')
def main(cfg):
    global model
    global datamodule
    global ckpt_callback_list
    global output_dir

    output_dir = os.path.abspath(os.curdir)
    os.chdir(hydra.utils.get_original_cwd())

    seed_all(cfg.seed)
    
    if cfg.task.mode == 'train':
        if not cfg.task.resume:
            if not cfg.task.finetune or not os.path.exists(cfg.weight):
                model = DRNet(cfg)
            else:   # finetuning based on previous weights
                model = DRNet.load_from_checkpoint(cfg.weight, cfg)
        else:   # resume
            model = DRNet() # DRNet.load_from_checkpoint(cfg.weight)
    else:
        model = DRNet.load_from_checkpoint(cfg.weight)
    
    datamodule = DRDataModule(cfg)

    mode = cfg.task.mode
    if mode == 'train':
        train(cfg)
    elif mode == 'test':
        test(cfg)

if __name__ == '__main__':
    main()

# python main.py task=train_comp 'gpus=[0,1,2,7]'