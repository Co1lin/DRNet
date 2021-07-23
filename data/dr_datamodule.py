import pytorch_lightning as pl
from data.drset import DRSet, get_dataloader

class DRDataModule(pl.LightningDataModule):

    def __init__(self, phase: str = None, 
                 datasets_root_path = 'datasets'):
        super.__init__()
        self.phase = phase
        self.datasets_root_path = datasets_root_path
    
    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        r"""
        :param stage: 'fit' or 'test' 
        """
        if stage == 'fit':
            self.dataset_train = DRSet('train', self.phase)
            self.dataset_val = DRSet('val', self.phase)
        elif stage == 'test':
            self.dataset_test = DRSet('test', self.phase)

    def train_dataloader(self):
        return get_dataloader('train', self.phase)

    def val_dataloader(self):
        return get_dataloader('val', self.phase)

    def test_dataloader(self):
        return get_dataloader('test', self.phase)
    