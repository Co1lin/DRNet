# Occupancy Networks
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.nn import functional as F
import pytorch_lightning as pl

from models.encoder_latent import Encoder_Latent
from models.occ_decoder import DecoderCBatchNorm

from external.common import make_3d_grid
from data.scannet_config import ScannetConfig

class ONet(pl.LightningModule):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    '''
    def __init__(self, cfg):
        super().__init__()

        self.dataset_config = ScannetConfig()
        self.generate_mesh = cfg.generation.generate_mesh

        '''Parameter Configs'''
        decoder_kwargs = {}
        encoder_latent_kwargs = {}
        self.z_dim = cfg.data.z_dim
        dim = 3
        self.use_cls_for_completion = cfg.data.use_cls_for_completion
        if not True:
            c_dim = self.use_cls_for_completion*cfg.dataset_config.num_class + 128
        else:   # always skip propagate
            c_dim = self.use_cls_for_completion * self.dataset_config.num_class + 256 # TODO: 512
        self.threshold = 0.5

        '''Module Configs'''
        if self.z_dim != 0:
            self.encoder_latent = Encoder_Latent(dim=dim, z_dim=self.z_dim, c_dim=c_dim, **encoder_latent_kwargs)
        else:
            self.encoder_latent = None

        self.decoder = DecoderCBatchNorm(dim=dim, z_dim=self.z_dim, c_dim=c_dim, **decoder_kwargs)

        '''Mount mesh generator'''
        if self.generate_mesh:
            from models.generator import Generator3D
            self.generator = Generator3D(self,
                                         threshold=0.5,
                                         resolution0=32,
                                         upsampling_steps=0,
                                         sample=False,
                                         refinement_step=0,
                                         simplify_nfaces=None,
                                         preprocessor=None,
                                         use_cls_for_completion=self.use_cls_for_completion)

    def compute_loss(self, input_features_for_completion, input_points_for_completion, input_points_occ_for_completion,
                     cls_codes_for_completion, export_shape=False):
        '''
        Compute loss for OccNet
        :param input_features_for_completion (N_B x D): Number of bounding boxes x Dimension of proposal feature.
        :param input_points_for_completion (N_B, N_P, 3): Number of bounding boxes x Number of Points x 3.
        :param input_points_occ_for_completion (N_B, N_P): Corresponding occupancy values.
        :param cls_codes_for_completion (N_B, N_C): One-hot category codes.
        :param export_shape (bool): whether to export a shape voxel example.
        :return:
        '''
        batch_size = input_features_for_completion.size(0)
        if self.use_cls_for_completion:
            cls_codes_for_completion = cls_codes_for_completion.float()
            input_features_for_completion = torch.cat([input_features_for_completion, cls_codes_for_completion], dim=-1)

        kwargs = {}
        '''Infer latent code z.'''
        if self.z_dim > 0:
            q_z = self.infer_z(input_points_for_completion, input_points_occ_for_completion, input_features_for_completion, **kwargs)
            z = q_z.rsample()
            # KL-divergence
            p0_z = self.get_prior_z(self.z_dim)
            kl = dist.kl_divergence(q_z, p0_z).sum(dim=-1)
            loss = kl.mean()
        else:
            z = torch.empty(size=(batch_size, 0))
            loss = 0.

        '''Decode to occupancy voxels.'''
        logits = self.decode(input_points_for_completion, z, input_features_for_completion, **kwargs).logits
        loss_i = F.binary_cross_entropy_with_logits(
            logits, input_points_occ_for_completion, reduction='none')
        loss = loss + loss_i.sum(-1).mean()

        '''Export Shape Voxels.'''
        if export_shape:
            shape = (16, 16, 16)
            p = make_3d_grid([-0.5 + 1/32] * 3, [0.5 - 1/32] * 3, shape)
            p = p.expand(batch_size, *p.size())
            z = self.get_z_from_prior((batch_size,), sample=False)
            kwargs = {}
            p_r = self.decode(p, z, input_features_for_completion, **kwargs)

            occ_hat = p_r.probs.view(batch_size, *shape)
            voxels_out = (occ_hat >= self.threshold)
        else:
            voxels_out = None

        return loss, voxels_out

    def compute_loss_with_cls_mask(self, input_features_for_completion, 
                                   input_points_for_completion, input_points_occ_for_completion,
                                   cls_codes_for_completion, cls_idx, export_shape=False):
        '''
        Compute loss for OccNet
        :param input_features_for_completion (N_B x D): Number of bounding boxes x Dimension of proposal feature.
        :param input_points_for_completion (N_B, N_P, 3): Number of bounding boxes x Number of Points x 3.
        :param input_points_occ_for_completion (N_B, N_P): Corresponding occupancy values.
        :param cls_codes_for_completion (N_B, N_C): One-hot category codes.
        :param export_shape (bool): whether to export a shape voxel example.
        :return:
        '''
        batch_size = input_features_for_completion.size(0)
        if self.use_cls_for_completion:
            cls_codes_for_completion = cls_codes_for_completion.float()
            input_features_for_completion = torch.cat([input_features_for_completion, cls_codes_for_completion], dim=-1)

        kwargs = {}
        '''Infer latent code z.'''
        if self.z_dim > 0:
            q_z = self.infer_z(input_points_for_completion, input_points_occ_for_completion, input_features_for_completion, **kwargs)
            z = q_z.rsample()
            # KL-divergence
            p0_z = self.get_prior_z(self.z_dim)
            kl = dist.kl_divergence(q_z, p0_z).sum(dim=-1)
            loss = kl.mean()
        else:
            z = torch.empty(size=(batch_size, 0))
            loss = 0.

        '''Decode to occupancy voxels.
        cls_codes_for_completion: N_obj, 8
        logits: N_obj, 2048
        loss_i: N_obj, 2048
        '''
        logits = self.decode(input_points_for_completion, z, input_features_for_completion, **kwargs).logits
        loss_i = F.binary_cross_entropy_with_logits(
            logits, input_points_occ_for_completion, reduction='none')
        loss_i = loss_i[ torch.argmax(cls_codes_for_completion, dim=-1) == cls_idx ]
        if loss_i.shape[0] > 0:
            loss += loss_i.sum(-1).mean()

            '''Export Shape Voxels.'''
            if export_shape:
                shape = (16, 16, 16)
                p = make_3d_grid([-0.5 + 1/32] * 3, [0.5 - 1/32] * 3, shape)
                p = p.expand(batch_size, *p.size())
                z = self.get_z_from_prior((batch_size,), sample=False)
                kwargs = {}
                p_r = self.decode(p, z, input_features_for_completion, **kwargs)

                occ_hat = p_r.probs.view(batch_size, *shape)
                voxels_out = (occ_hat >= self.threshold)
            else:
                voxels_out = None

            return loss, voxels_out
        else:
            return 0, None

    def forward(self, input_points_for_completion, input_features_for_completion, cls_codes_for_completion, sample=False, **kwargs):
        '''
        Performs a forward pass through the network.
        :param input_points_for_completion (tensor): sampled points
        :param input_features_for_completion (tensor): conditioning input
        :param cls_codes_for_completion: class codes for input shapes.
        :param sample (bool): whether to sample for z
        :param kwargs:
        :return:
        '''
        if self.use_cls_for_completion:
            cls_codes_for_completion = cls_codes_for_completion.to(self.device).float()
            input_features_for_completion = torch.cat([input_features_for_completion, cls_codes_for_completion], dim=-1)
        '''Encode the inputs.'''
        batch_size = input_points_for_completion.size(0)
        z = self.get_z_from_prior((batch_size,), sample=sample)
        p_r = self.decode(input_points_for_completion, z, input_features_for_completion, **kwargs)
        return p_r

    def get_z_from_prior(self, size=torch.Size([]), sample=False):
        ''' Returns z from prior distribution.

        Args:
            size (Size): size of z
            sample (bool): whether to sample
        '''
        p0_z = self.get_prior_z(self.z_dim)
        if sample:
            z = p0_z.sample(size)
        else:
            z = p0_z.mean
            z = z.expand(*size, *z.size())

        return z

    def decode(self, input_points_for_completion, z, features, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        :param input_points_for_completion: points
        :param z: latent code z
        :param features: latent conditioned features
        :return:
        '''
        logits = self.decoder(input_points_for_completion, z, features, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def infer_z(self, p, occ, c, **kwargs):
        '''
        Infers latent code z.
        :param p : points tensor
        :param occ: occupancy values for occ
        :param c: latent conditioned code c
        :param kwargs:
        :return:
        '''
        if self.encoder_latent is not None:
            mean_z, logstd_z = self.encoder_latent(p, occ, c, **kwargs)
        else:
            batch_size = p.size(0)
            mean_z = torch.empty(batch_size, 0).to(self.device)
            logstd_z = torch.empty(batch_size, 0).to(self.device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        return q_z

    def get_prior_z(self, z_dim):
        ''' Returns prior distribution for latent code z.

        Args:
            zdim: dimension of latent code z.
        '''
        p0_z = dist.Normal(
            torch.zeros(z_dim).to(self.device),
            torch.ones(z_dim).to(self.device)
        )

        return p0_z
