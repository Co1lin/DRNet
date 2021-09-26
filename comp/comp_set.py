import os
import random
import json
from re import S
from typing import List
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from net_utils.transforms import SubsamplePoints

from torch.utils.data.distributed import DistributedSampler
from utils.tools import read_json_file, read_pkl_file
from utils import pc_util
from net_utils.libs import random_sampling_by_instance, rotz, flip_axis_to_camera
from external import binvox_rw

def get_splited_data(path, mode: str) -> List[str]:
    split_file_path = os.path.join(path, 'split.json')
    if os.path.exists(split_file_path):
        with open(split_file_path) as f:
            d = json.load(f)
            return d[mode]
    else:
        with open(split_file_path, 'w') as f:
            files = os.listdir(path)
            num = len(files)
            random.shuffle(files)
            d = {}  # 8:1:1
            portion = [0.8, 0.9]
            d['train'] = [os.path.join(path, file) for file in files[:int(num*portion[0])]]
            d['val'] = [os.path.join(path, file) for file in files[int(num*portion[0]) + 1 : int(num*portion[1])]]
            d['test'] = [os.path.join(path, file) for file in files[int(num*portion[1]) + 1 :]]
            json.dump(d, f)
            return d[mode]

class CompSet(Dataset):

    def __init__(self, mode: str):
        super().__init__()

        self.shapenet_pc_path = 'datasets/ShapeNetv2_data/point'
        self.shapenetid_to_name = {
            '04256520': 'sofa',
            '04379243': 'table',
            '02747177': 'trash bin',
            '02933112': 'cabinet',
            '02871439': 'bookshelf',
            '02808440': 'bathtub',
            '03001627': 'chair',
            '03211117': 'display',
        }
        self.classname_to_shapenetid = {v:k for k, v in self.shapenetid_to_name.items()}
        self.shapenetid_to_classid = {
            '04256520': 3,
            '04379243': 0,
            '02747177': 4,
            '02933112': 5,
            '02871439': 2,
            '02808440': 7,
            '03001627': 1,
            '03211117': 6,
        }
        self.classid_to_shapenetid = {v:k for k, v in self.shapenetid_to_classid.items()}

        classes_path = [os.path.join(self.shapenet_pc_path, class_id) for class_id in self.shapenetid_to_name.keys()]
        classes_path = [os.path.join(self.shapenet_pc_path, '02871439')]
        self.objs_path = []
        for class_path in classes_path:
            self.objs_path += get_splited_data(class_path, mode)

        self.mode = mode
        self.n_points_object = [1024, 1024]
        self.points_transform = SubsamplePoints([1024, 1024], mode)
        self.points_unpackbits = True
    
    def __len__(self):
        return len(self.objs_path)
    
    def __getitem__(self, index):
        r"""
        :return: { 'points': (2048, 3), 'occ': (2048,), 'volume': (1,), }
        """
        try:
            obj = dict(np.load(self.objs_path[index], allow_pickle=True))
        except:
            return self.__getitem__(index - 1)

        points = obj['points']
        points = points.astype(np.float32)
        if points.dtype == np.float16 and self.mode == 'train':
            points += 1e-4 * np.random.randn(*points.shape)
        
        occupancies = obj['occupancies']
        if self.points_unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        data = { 'points':points, 'occ': occupancies, 'index': index }
        data['file_name'] = os.path.basename(self.objs_path[index]).split('.npz')[0]
        if self.points_transform is not None:
            data = self.points_transform(data)
        
        for shapenetid in self.shapenetid_to_classid.keys():
            if shapenetid in self.objs_path[index]:
                data['obj_class'] = self.shapenetid_to_classid[shapenetid]
                break
        
        return data
        
def get_dataloader(mode: str = None, config = None):
    r"""
    :param mode: 'train', 'val' or 'test'
    """
    default_collate = torch.utils.data.dataloader.default_collate
    def collate_fn(batch):
        '''
        data collater
        :param batch:
        :return: collated_batch
        '''
        collated_batch = {}
        for key in batch[0]:
            if key not in ['shapenet_catids', 'shapenet_ids']:
                collated_batch[key] = default_collate([elem[key] for elem in batch])
            else:
                collated_batch[key] = [elem[key] for elem in batch]

        return collated_batch

    # Ref: https://pytorch.org/docs/master/notes/randomness.html#dataloader
    g = torch.Generator()
    g.manual_seed(0)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataset = CompSet(mode)
    '''
    val: 4535
    train: 15892
    test: 2267
    '''
    # sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        # sampler=sampler,
        num_workers=config.num_workers,
        batch_size=config.task.batch_size,
        shuffle=(mode == 'train'),
        # collate_fn=collate_fn,
        #worker_init_fn=seed_worker,
        generator=g,
    )
    return dataloader


class CompDataModule(pl.LightningDataModule):

    def __init__(self, config = None):
        super().__init__()
        self.cfg = config

    def train_dataloader(self):
        return get_dataloader('train', self.cfg)

    def val_dataloader(self):
        return get_dataloader('val', self.cfg)

    def test_dataloader(self):
        return get_dataloader('test', self.cfg)