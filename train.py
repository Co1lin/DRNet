import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from data.dr_datamodule import DRDataModule
from models.drnet import DRNet

if __name__ == '__main__':

    phase = 'detection'

    os.environ['WANDB_API_KEY'] = 'a5044d4b533063065587a9fce532ad394071fc48'
    wandb_logger = WandbLogger(
        entity='co1lin',
        project='DRNet',
    )

    drdm = DRDataModule(phase)
    
    drnet = DRNet(phase, export_shape=False, generate_mesh=False, evaluate_mesh_mAP=False)

    trainer = pl.Trainer(gpus=[3], logger=wandb_logger)
    trainer.fit(drnet, drdm)