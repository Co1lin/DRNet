import os
import numpy as np
import torch
from tqdm import tqdm
import pytorch_lightning as pl
from torch import optim
from torch._C import device
import torch.nn as nn
from multiprocessing import Pool
from functools import partial
from utils.tools import read_json_file
from utils import pc_util
from models.pointnet2backbone import Pointnet2Backbone
from models.skip_propagation import SkipPropagation
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.occupancy_net import ONet
from external.common import compute_iou
from net_utils.nn_distance import nn_distance
from net_utils.ap_helper import parse_predictions, parse_groundtruths, assembly_pred_map_cls, assembly_gt_map_cls, APCalculator
from net_utils.libs import flip_axis_to_depth, extract_pc_in_box3d, flip_axis_to_camera
from net_utils.box_util import get_3d_box
from data.scannet_config import ScannetConfig
from models.loss import DetectionLoss, ONet_Loss, compute_objectness_loss, chamfer_func

Shapenetid_to_name = {
    '04256520': 'sofa',
    '04379243': 'table',
    '02747177': 'trash bin',
    '02933112': 'cabinet',
    '02871439': 'bookshelf',
    '02808440': 'bathtub',
    '03001627': 'chair',
    '03211117': 'display',
}
Classname_to_shapenetid = {v:k for k, v in Shapenetid_to_name.items()}
Shapenetid_to_classid = {
    '04256520': 3,
    '04379243': 0,
    '02747177': 4,
    '02933112': 5,
    '02871439': 2,
    '02808440': 7,
    '03001627': 1,
    '03211117': 6,
}
Classid_to_shapenetid = {v:k for k, v in Shapenetid_to_classid.items()}

latent_dim = 256

class CompNet(pl.LightningModule):
    
    def __init__(self, config = None):
        super().__init__()

        self.save_hyperparameters()
        # configs
        self.cfg = self.hparams.config.task

        self.completion_net = ONet(self.cfg)

        if True: #self.cfg.mode == 'train':
            self.latent_layers = nn.ModuleList([
                nn.Linear(1, latent_dim) for _ in range(18200)
            ])

        self.completion_loss = ONet_Loss(0.005)

        self.eval_config = {
            'remove_empty_box': self.cfg.mode == 'test',
            'use_3d_nms': True,
            'nms_iou': 0.25,
            'use_old_type_nms': False,
            'cls_nms': True,
            'per_class_proposal': True,
            'conf_thresh': 0.05,
        }


    def forward(self, batch):
        r"""
        :param batch: { 
                        'points':       (batch_size, 2048, 3),
                        'occ':          (batch_size, 2048),
                        'index':        (batch_size),
                        'obj_class':    (batch_size),
                        'volume':       (batch_size),
                        'file_name':    (batch_size),
                    }
        """
        batch_size = self.cfg.batch_size
        this_batch_size = batch['obj_class'].shape[0]
        device = self.device
        input_points_for_completion = batch['points']
        input_points_occ_for_completion = batch['occ']
        cls_codes_for_completion = nn.functional.one_hot(batch['obj_class'], num_classes=8)
        
        min_loss = 0x7fffffff
        with torch.enable_grad():
            latent_layers = [ nn.Linear(1, latent_dim).to(self.device) for _ in range(this_batch_size) ]
            optim_params = []
            for layer in latent_layers:
                optim_params += list(layer.parameters())
            optimizer = optim.AdamW(optim_params, lr=1e-3)

            not_decrease_steps  = 0
            pbar = tqdm(total=1e6)
            counter = 0;
            while not_decrease_steps < 1000 and counter < 5000:
                counter += 1
                optimizer.zero_grad()
                object_input_features = torch.stack([latent_layer(torch.ones(1).to(self.device))
                                    for latent_layer in latent_layers
                                ])
                completion_loss = 0
                compl_loss, _ = self.completion_net.compute_loss(
                    object_input_features,
                    input_points_for_completion,
                    input_points_occ_for_completion,
                    cls_codes_for_completion,
                    False,
                )
                # object_input_features0 = object_input_features[0].unsqueeze(0)
                # input_points_for_completion0 = input_points_for_completion[0].unsqueeze(0)
                # input_points_occ_for_completion0 = input_points_occ_for_completion[0].unsqueeze(0)
                # cls_codes_for_completion0 = cls_codes_for_completion[0].unsqueeze(0)
                # compl_loss0, _ = self.completion_net.compute_loss(
                #     object_input_features0,
                #     input_points_for_completion0,
                #     input_points_occ_for_completion0,
                #     cls_codes_for_completion0,
                #     False,
                # )
                completion_loss += compl_loss
                if completion_loss < min_loss:
                    min_loss = completion_loss
                    completion_loss.backward()
                    optimizer.step()
                    not_decrease_steps = 0
                else:
                    not_decrease_steps += 1
                pbar.update(1)
                pbar.set_description(f'loss: {completion_loss}', refresh=True)
            # end while
        # end with
        end_points = { 'loss': min_loss }
        object_input_features = torch.stack([latent_layer(torch.ones(1).to(self.device))
                                    for latent_layer in latent_layers
                                ])
        meshes = self.completion_net.generator.generate_mesh(
            object_input_features, cls_codes_for_completion
        )
        end_points['meshes'] = meshes
        return end_points


    def _common_step(self, batch, batch_idx):
        r"""
        :param batch: { 
                        'points':       (batch_size, 2048, 3),
                        'occ':          (batch_size, 2048),
                        'index':        (batch_size),
                        'obj_class':    (batch_size),
                        'volume':       (batch_size),
                    }
        :param batch_idx: start from zero
        """
        batch_size = self.cfg.batch_size
        this_batch_size = batch['obj_class'].shape[0]
        device = self.device
        latent_layers = [ self.latent_layers[i] for i in batch['index'] ]
        object_input_features = torch.stack([latent_layer(torch.ones(1).to(self.device))
                                    for latent_layer in latent_layers
                                ])
        input_points_for_completion = batch['points']
        input_points_occ_for_completion = batch['occ']
        cls_codes_for_completion = nn.functional.one_hot(batch['obj_class'], num_classes=8)
        completion_loss = 0

        compl_loss, _ = self.completion_net.compute_loss(
            object_input_features,
            input_points_for_completion,
            input_points_occ_for_completion,
            cls_codes_for_completion,
            False,
        )
        completion_loss += compl_loss

        compl_loss = torch.cat([completion_loss.unsqueeze(0),
                                    torch.zeros(1).to(self.device)], dim = 0)
        compl_loss = self.completion_loss(compl_loss.unsqueeze(0))

        end_points = {}
        end_points['losses'] = compl_loss
        end_points['loss'] = compl_loss['total_loss']
        return end_points
    
    def training_step(self, batch, batch_idx, optimizer_idx=None):
        out = self._common_step(batch, batch_idx)
        loss = out['losses']
        self.log('train_loss', out['loss'])
        # print learning rates
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for i, optimizer in enumerate(optimizers):
            self.log(
                f'lr#{i}',
                optimizer.param_groups[0]['lr'],
                prog_bar=True, on_step=True
            )
        # print other losses
        self.log('train_comp_loss', loss['completion_loss'], prog_bar=True, on_step=True)
        
        return out

    def validation_step(self, batch, batch_idx):
        r"""
        :param batch: { 
                        'points':       (batch_size, 2048, 3),
                        'occ':          (batch_size, 2048),
                        'index':        (batch_size),
                        'obj_class':    (batch_size),
                        'volume':       (batch_size),
                    }
        :param batch_idx: start from zero
        """
        # min_loss = 1000
        # self.log('val_loss', min_loss, prog_bar=True, on_step=True)
        # return min_loss
        batch_size = self.cfg.batch_size
        this_batch_size = batch['obj_class'].shape[0]
        device = self.device
        input_points_for_completion = batch['points']
        input_points_occ_for_completion = batch['occ']
        cls_codes_for_completion = nn.functional.one_hot(batch['obj_class'], num_classes=8)
        
        min_loss = 0x7fffffff
        with torch.enable_grad():
            latent_layers = [ nn.Linear(1, latent_dim).to(self.device) for _ in range(this_batch_size) ]
            optim_params = []
            for layer in latent_layers:
                optim_params += list(layer.parameters())
            optimizer = optim.AdamW(optim_params, lr=1e-3)

            not_decrease_steps  = 0
            while not_decrease_steps < 500:
                optimizer.zero_grad()
                object_input_features = torch.stack([latent_layer(torch.ones(1).to(self.device))
                                    for latent_layer in latent_layers
                                ])
                completion_loss = 0
                compl_loss, _ = self.completion_net.compute_loss(
                    object_input_features,
                    input_points_for_completion,
                    input_points_occ_for_completion,
                    cls_codes_for_completion,
                    False,
                )
                completion_loss += compl_loss
                if completion_loss < min_loss:
                    min_loss = completion_loss
                    completion_loss.backward()
                    optimizer.step()
                    not_decrease_steps = 0
                else:
                    not_decrease_steps += 1
            # end while
        # end with
        self.log('val_loss', min_loss, prog_bar=True, on_step=True)
        return min_loss

    def test_step(self, batch, batch_idx):
        out = self(batch)
        # export
        meshes = out['meshes']
        i = 0
        for mesh in meshes:
            net_idx = batch['obj_class'][i].item()
            mesh_dir_path = os.path.join('comp/meshes',
                f'{net_idx}_{ Shapenetid_to_name[Classid_to_shapenetid[net_idx]] }')
            if not os.path.exists(mesh_dir_path):
                os.makedirs(mesh_dir_path)
            mesh_path = os.path.join(mesh_dir_path,
                f'{ batch["file_name"][i] }_{ int(out["loss"]) }.ply')
            mesh.export(mesh_path)
            i += 1
        return out

    def configure_optimizers(self):
        optim_cfg = self.cfg.optimizer
        optim_params = []
        # latent
        optim_params.append({
            'params': list(self.latent_layers.parameters()),
            'lr': optim_cfg.latent.lr,
            'betas': optim_cfg.latent.betas,
            'eps': optim_cfg.latent.eps,
            'weight_decay': optim_cfg.latent.weight_decay,
        })
        # onet
        optim_params.append({
            'params': list(self.completion_net.parameters()),
            'lr': optim_cfg.onet.lr,
            'betas': optim_cfg.onet.betas,
            'eps': optim_cfg.onet.eps,
            'weight_decay': optim_cfg.onet.weight_decay,
        })
        optimizer = optim.AdamW(optim_params)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=optim_cfg.onet.patience,
            factor=optim_cfg.onet.factor,
            threshold=optim_cfg.onet.threshold,
        )
        #return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def _bak_configure_optimizers(self):
        optimizers = schedulers = {}
        optim_cfg = self.cfg.optimizer

        optimizers['latent'] = optim.AdamW(
            list(self.latent_layers.parameters()),
            lr=optim_cfg.latent.lr,
            betas=optim_cfg.latent.betas,
            eps=optim_cfg.latent.eps,
            weight_decay=optim_cfg.latent.weight_decay,
        )

        optimizers['onet'] = optim.AdamW(
            list(self.completion_nets.parameters()),
            lr=optim_cfg.onet.lr,
            betas=optim_cfg.onet.betas,
            eps=optim_cfg.onet.eps,
            weight_decay=optim_cfg.onet.weight_decay,
        )

        optimizers_list = [v for _, v in optimizers.items()]

        schedulers['latent'] = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizers['latent'],
            patience=optim_cfg.latent.patience,
            factor=optim_cfg.latent.factor,
            threshold=optim_cfg.latent.threshold,
        )

        schedulers['onet'] = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizers['onet'],
            patience=optim_cfg.onet.patience,
            factor=optim_cfg.onet.factor,
            threshold=optim_cfg.onet.threshold,
        )

        schedulers_list = [ {'scheduler': v, 'monitor': 'val_loss'} for _, v in schedulers.items() ]

        return optimizers_list, schedulers_list