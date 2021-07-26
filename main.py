import os
import random
from argparse import ArgumentParser
import hydra
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.loggers import WandbLogger
import torch
from data.dr_datamodule import DRDataModule
from models.drnet import DRNet
from config.common import *

def init_wandb():
    os.environ['WANDB_API_KEY'] = 'a5044d4b533063065587a9fce532ad394071fc48'
    wandb_logger = WandbLogger(
        entity='co1lin',
        project='DRNet',
    )
    return wandb_logger

def train(model, datamodule, cfg: BasicConfig):
    wandb_logger = init_wandb()
    trainer = pl.Trainer(gpus=cfg.gpus, logger=wandb_logger, accelerator='ddp')
    #trainer = pl.Trainer(gpus=[1], logger=wandb_logger)
    trainer.fit(model, datamodule)

def test(model, datamodule, cfg: BasicConfig):
    trainer = pl.Trainer(gpus=cfg.gpus)
    trainer.test(model, datamodule=datamodule)

def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

@hydra.main(config_path=None, config_name='config')
def main(config: Config):
    os.chdir(hydra.utils.get_original_cwd())
    cfg: BasicConfig = config.config
    seed_all(cfg.seed)
    
    drdm = DRDataModule(cfg)
    drnet = DRNet(cfg)

    mode = cfg.mode
    if mode == 'train':
        train(drnet, drdm, cfg)
    elif mode == 'test':
        test(drnet, drdm, cfg)

if __name__ == '__main__':
    main()