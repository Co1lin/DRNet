import numpy as np
import torch
import pytorch_lightning as pl

from models.pointnet2backbone import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.occupancy_net import ONet
from external.common import compute_iou
from net_utils.ap_helper import parse_predictions, parse_groundtruths, assembly_pred_map_cls, assembly_gt_map_cls
from data.scannet_config import ScannetConfig

class DRNet(pl.LightningModule):

    def __init__(self, 
                 phase: str = None,
                 export_shape: bool = False,
                 generate_mesh: bool = False,
                 evaluate_mesh_mAP: bool = False):
        r"""
        :param phase: 'detection' or 'completion'
        :param export_shape: if output shape voxels for visualization
        """
        super().__init__()

        self.dataset_config = ScannetConfig()
        
        self.phase = phase
        self.export_shape = export_shape
        self.generate_mesh = generate_mesh
        self.evaluate_mesh_mAP = evaluate_mesh_mAP

        self.backbone = Pointnet2Backbone()
        self.voting = VotingModule()
        self.proposal = ProposalModule()
        self.completion = ONet()

    def forward(self, x):
        

        if shape_example is not None:
            gt_voxels = batch['object_voxels'][0][BATCH_PROPOSAL_IDs[0,..., 1]]
            ious = compute_iou(shape_example.detach().cpu().numpy(), gt_voxels.detach().cpu().numpy())
            cls_labels = BATCH_PROPOSAL_IDs[0, ..., 2].cpu().numpy()
            iou_stats = {'cls':cls_labels, 'iou':ious}
        else:
            iou_stats = None

        if self.generate_mesh:
            meshes = self.completion.generator.generate_mesh(object_input_features, cls_codes_for_completion)
        else:
            meshes = None

        pass

    def training_step(self, batch, batch_idx):
        r"""
        :return: 
        """
        inputs = {'point_clouds': batch['point_clouds']}
        end_points = {}
        
        # pointnet++ backbone
        end_points = self.backbone(inputs['point_clouds'], end_points)

        # hough voting
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features

        xyz, features = self.voting(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        # detection
        if_proposal_feature = (self.phase == 'completion')
        end_points, proposal_features = self.proposal(xyz, features, end_points, if_proposal_feature)

        eval_dict, parsed_predictions = parse_predictions(end_points, batch, self.cfg.eval_config)
        parsed_gts = parse_groundtruths(batch, self.cfg.eval_config)

        # completion
        evaluate_mesh_mAP = True if self.phase == 'completion' and self.generate_mesh and self.evaluate_mesh_mAP else False

        if self.phase == 'completion':
            # use 3D NMS to generate sample ids.
            batch_sample_ids = eval_dict['pred_mask']
            # dump_threshold = self.cfg.eval_config['conf_thresh'] if evaluate_mesh_mAP else self.cfg.config['generation']['dump_threshold']
            dump_threshold = 0.05 if evaluate_mesh_mAP else 0.5
            BATCH_PROPOSAL_IDs = self.get_proposal_id(end_points, batch, mode='random', batch_sample_ids=batch_sample_ids, DUMP_CONF_THRESH=dump_threshold)

            # Skip propagate point clouds to box centers.
            
            # gather proposal features
            gather_ids = BATCH_PROPOSAL_IDs[...,0].unsqueeze(1).repeat(1, 128, 1).long()
            proposal_features = torch.gather(proposal_features, 2, gather_ids)

            # gather proposal centers
            gather_ids = BATCH_PROPOSAL_IDs[...,0].unsqueeze(-1).repeat(1,1,3).long()
            pred_centers = torch.gather(end_points['center'], 1, gather_ids)

            # gather proposal orientations
            pred_heading_class = torch.argmax(end_points['heading_scores'], -1)  # B,num_proposal
            heading_residuals = end_points['heading_residuals_normalized'] * (np.pi / self.dataset_config.num_heading_bin)  # Bxnum_proposalxnum_heading_bin
            pred_heading_residual = torch.gather(heading_residuals, 2, pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
            pred_heading_residual.squeeze_(2)
            heading_angles = self.dataset_config.class2angle_cuda(pred_heading_class, pred_heading_residual)
            heading_angles = torch.gather(heading_angles, 1, BATCH_PROPOSAL_IDs[...,0])
            
            # gather instance labels
            proposal_instance_labels = torch.gather(batch['object_instance_labels'], 1, BATCH_PROPOSAL_IDs[...,1])
            object_input_features, mask_loss = self.skip_propagation(pred_centers, heading_angles, proposal_features, inputs['point_clouds'], batch['point_instance_labels'], proposal_instance_labels)

            # Prepare input-output pairs for shape completion
            # proposal_to_gt_box_w_cls_list (B x N_Limit x 4): (bool_mask, proposal_id, gt_box_id, cls_id)
            input_points_for_completion, \
            input_points_occ_for_completion, \
            cls_codes_for_completion = self.prepare_data(batch, BATCH_PROPOSAL_IDs)
            
            export_shape = batch.get('export_shape', self.export_shape) # if output shape voxels.
            batch_size, feat_dim, N_proposals = object_input_features.size()
            object_input_features = object_input_features.transpose(1, 2) \
                                    .contiguous().view(batch_size * N_proposals, feat_dim)
            completion_loss, shape_example = self.completion.compute_loss(object_input_features,
                                                                          input_points_for_completion,
                                                                          input_points_occ_for_completion,
                                                                          cls_codes_for_completion, export_shape)
            
        else:
            BATCH_PROPOSAL_IDs = None
            completion_loss = torch.tensor(0.)
            mask_loss = torch.tensor(0.)
            shape_example = None
        # end if self.phase == 'completion'

        completion_loss = torch.cat([completion_loss.unsqueeze(0), mask_loss.unsqueeze(0)], dim = 0)

        #return end_points, completion_loss.unsqueeze(0), shape_example, BATCH_PROPOSAL_IDs
        


    def validation_step(self, batch, batch_idx):

        pass

    def configure_optimizers(self):

        pass