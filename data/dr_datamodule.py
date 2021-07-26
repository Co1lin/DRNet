import pytorch_lightning as pl
from data.drset import DRSet, get_dataloader

class DRDataModule(pl.LightningDataModule):

    def __init__(self, config = None):
        r"""
        :param phase: 'detection' or 'completion'
        """
        super().__init__()
        self.cfg = config
    
    # def prepare_data(self):
    #     pass

    # def setup(self, stage: str = None):
    #     r"""
    #     :param stage: 'fit' or 'test' 
    #     """
    #     if stage == 'fit':
    #         self.dataset_train = DRSet('train', self.cfg.task.phase)
    #         self.dataset_val = DRSet('val', self.cfg.task.phase)
    #     elif stage == 'test':
    #         self.dataset_test = DRSet('test', self.cfg.task.phase)

    def train_dataloader(self):
        return get_dataloader('train', self.cfg)

    def val_dataloader(self):
        return get_dataloader('val', self.cfg)

    def test_dataloader(self):
        return get_dataloader('test', self.cfg)
    