import os
import random
from argparse import ArgumentParser
import hydra
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
import torch
from data.dr_datamodule import DRDataModule
from models.drnet import DRNet

def init_wandb(cfg):
    os.environ['WANDB_API_KEY'] = 'a5044d4b533063065587a9fce532ad394071fc48'
    wandb_logger = WandbLogger(
        entity='co1lin',
        project='DRNet',
        group=cfg.task.name,
    )
    return wandb_logger

model: pl.LightningModule = None
datamodule: pl.LightningDataModule = None
ckpt_callback_list: list = None
output_dir: str = None

def train(cfg):
    ckpt_callback_list = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=output_dir,
            filename=f'{cfg.task.name}-' + '{epoch}-{val_loss:.2f}-best',
            save_top_k=3,
            save_last=True,
            mode='min',
        ),
    ]
    wandb_logger = init_wandb(cfg)
    if not cfg.task.resume:
        trainer = pl.Trainer(
            max_epochs=cfg.task.epochs,
            gpus=cfg.gpus, 
            logger=wandb_logger, 
            accelerator='ddp', 
            log_every_n_steps=10, 
            callbacks=ckpt_callback_list,
            sync_batchnorm=True,
            #plugins=DDPPlugin(find_unused_parameters=False),
        )
    else:
        trainer = pl.Trainer(
            resume_from_checkpoint=cfg.weight,
            max_epochs=cfg.task.epochs,
            gpus=cfg.gpus,
            logger=wandb_logger, 
            accelerator='ddp', 
            log_every_n_steps=10, 
            callbacks=ckpt_callback_list,
            sync_batchnorm=True,
            #plugins=DDPPlugin(find_unused_parameters=False),
        )
    # end if
    trainer.fit(model, datamodule=datamodule)

def test(cfg):
    trainer = pl.Trainer(
        gpus=[cfg.gpus[0]],
    )
    trainer.test(model, datamodule=datamodule)


def seed_all(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


@hydra.main(config_path='config', config_name='config')
def main(cfg):
    # increase ulimit
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (16384, rlimit[1]))

    global model
    global datamodule
    global ckpt_callback_list
    global output_dir

    output_dir = os.path.abspath(os.curdir) # get hydra output dir
    os.chdir(hydra.utils.get_original_cwd())    # set working dir to the original one

    seed_all(cfg.seed)
    
    if cfg.task.mode == 'train':
        if cfg.task.resume:
            model = DRNet(cfg)  # DRNet.load_from_checkpoint(cfg.weight)
        else:
            if not cfg.task.finetune or not os.path.exists(cfg.weight):
                model = DRNet(cfg)  # train from scratch
            else:   # finetuning based on previous weights
                # set strict to False to ignore the missing params in the previous phase
                model = DRNet.load_from_checkpoint(cfg.weight, config=cfg, strict=False)
    elif cfg.task.mode == 'test':
        if os.path.exists(cfg.weight):
            model = DRNet.load_from_checkpoint(cfg.weight, config=cfg)
        else:
            model = DRNet(cfg)  # test from scratch
    else:
        raise ValueError(f'Invalid task mode: {cfg.task.mode}')
    
    datamodule = DRDataModule(cfg)

    mode = cfg.task.mode
    if mode == 'train':
        train(cfg)
    elif mode == 'test':
        test(cfg)

if __name__ == '__main__':
    main()

# python main.py task=train_comp gpus='[0,1,2,7]'