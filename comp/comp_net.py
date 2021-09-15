import os
import numpy as np
import torch
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

class CompNet(pl.LightningModule):
    
    def __init__(self, config = None):
        super().__init__()

        self.save_hyperparameters()
        # configs
        self.cfg = self.hparams.config.task

        self.completion_nets = nn.ModuleList([
            ONet(self.cfg) for _ in range(8)
        ])
        self.latent_layers = nn.ModuleList([
            nn.Linear(1, 128) for _ in range(15892)
        ])
        if self.cfg.mode == 'test':
            self.latent_layers = nn.ModuleList([
                nn.Linear(1, 128) for _ in range(15892)
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
                    }
        """
        pass


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
        for net_idx in range(8):
            compl_loss, _ = self.completion_nets[net_idx].compute_loss_with_cls_mask(
                object_input_features,
                input_points_for_completion,
                input_points_occ_for_completion,
                cls_codes_for_completion,
                net_idx,
                False,
            )
            completion_loss += 1/8 * compl_loss

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
        out = self._common_step(batch, batch_idx)
        loss = out['losses']
        self.log('val_loss', out['loss'])
        # print other losses
        self.log('val_comp_loss', loss['completion_loss'], prog_bar=True)
        return out

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
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