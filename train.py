import pytorch_lightning as pl

from data.dr_datamodule import DRDataModule
from models.drnet import DRNet



if __name__ == '__main__':

    phase = 'detection'

    drdm = DRDataModule(phase)

    drnet = DRNet(phase, export_shape=False, generate_mesh=False, evaluate_mesh_mAP=False)

    trainer = pl.Trainer(gpus=[3])

    trainer.fit(drnet, drdm)