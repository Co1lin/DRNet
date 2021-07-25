import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from data.dr_datamodule import DRDataModule
from models.drnet import DRNet

def init_wandb():
    os.environ['WANDB_API_KEY'] = 'a5044d4b533063065587a9fce532ad394071fc48'
    wandb_logger = WandbLogger(
        entity='co1lin',
        project='DRNet',
    )
    return wandb_logger

def train():
    wandb_logger = init_wandb()
    trainer = pl.Trainer(gpus=[3], logger=wandb_logger)
    trainer.fit(drnet, drdm)

def test():
    trainer = pl.Trainer(gpus=[3])
    trainer.test(drnet, datamodule=drdm)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')

    args = parser.parse_args()

    phase = 'detection'
    drdm = DRDataModule(phase)
    drnet = DRNet(phase, export_shape=False, generate_mesh=False, evaluate_mesh_mAP=False)
    
    if args.mode == 'train':
        test()
    elif args.mode == 'test':
        test()