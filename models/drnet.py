import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch import optim

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

class DRNet(pl.LightningModule):

    def __init__(self, config = None):
        super().__init__()
        self.save_hyperparameters()
        # configs
        self.cfg = self.hparams.config.task
        self.dataset_config = ScannetConfig()
        
        # networks
        self.backbone = Pointnet2Backbone()
        self.voting = VotingModule()
        self.proposal = ProposalModule(self.cfg)
        if self.cfg.phase == 'completion':
            self.skip_propagation = SkipPropagation()
            self.completion = ONet(self.cfg)

        # losses
        self.detection_loss = DetectionLoss()
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
        
        # freeze
        if hasattr(self.cfg, 'freeze'):
            for m in self.cfg.freeze:
                module = getattr(self, m)
                module.freeze()

        # for test
        self.AP_IOU_THRESHOLDS = [0.5]
        self.ap_calculator_list = [
            APCalculator(iou_thresh, self.dataset_config.class2type, self.cfg.generation.evaluate_mesh_mAP) 
            for iou_thresh in self.AP_IOU_THRESHOLDS
        ]


    def forward(self, batch):
        with torch.no_grad():
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
            if_proposal_feature = (self.cfg.phase == 'completion')
            end_points, proposal_features = self.proposal(xyz, features, end_points, if_proposal_feature)

            eval_dict, parsed_predictions = parse_predictions(end_points, batch, self.eval_config)
            parsed_gts = parse_groundtruths(batch, self.eval_config)

            # completion
            evaluate_mesh_mAP = True if self.cfg.phase == 'completion' and self.cfg.generation.generate_mesh and self.cfg.generation.evaluate_mesh_mAP else False

            if self.cfg.phase == 'completion':
                # use 3D NMS to generate sample ids.
                batch_sample_ids = eval_dict['pred_mask']
                # dump_threshold = self.eval_config['conf_thresh'] if evaluate_mesh_mAP else self.cfg.config['generation']['dump_threshold']
                dump_threshold = 0.05 if evaluate_mesh_mAP else 0.5
                BATCH_PROPOSAL_IDs = self._get_proposal_id(end_points, batch, mode='random', batch_sample_ids=batch_sample_ids, DUMP_CONF_THRESH=dump_threshold)

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
                _ = self._select_data(batch, BATCH_PROPOSAL_IDs)

                batch_size, feat_dim, N_proposals = object_input_features.size()
                object_input_features = object_input_features.transpose(1, 2).contiguous().view(batch_size * N_proposals, feat_dim)

                gather_ids = BATCH_PROPOSAL_IDs[..., 0].unsqueeze(-1).repeat(1, 1, end_points['sem_cls_scores'].size(2))
                cls_codes_for_completion = torch.gather(end_points['sem_cls_scores'], 1, gather_ids)
                cls_codes_for_completion = (cls_codes_for_completion >= torch.max(cls_codes_for_completion, dim=2, keepdim=True)[0]).float()
                cls_codes_for_completion = cls_codes_for_completion.view(batch_size*N_proposals, -1)

                completion_loss, shape_example = self.completion.compute_loss(object_input_features, input_points_for_completion, input_points_occ_for_completion, cls_codes_for_completion, False)

                if shape_example is not None:
                    gt_voxels = batch['object_voxels'][0][BATCH_PROPOSAL_IDs[0,..., 1]]
                    ious = compute_iou(shape_example.detach().cpu().numpy(), gt_voxels.detach().cpu().numpy())
                    cls_labels = BATCH_PROPOSAL_IDs[0, ..., 2].detach().cpu().numpy()
                    iou_stats = {'cls': cls_labels, 'iou': ious}                    
                else:
                    iou_stats = None
                
                if self.cfg.generation.generate_mesh:
                    meshes = self.completion.generator.generate_mesh(object_input_features, cls_codes_for_completion)
                else:
                    meshes = None
                
            else:
                BATCH_PROPOSAL_IDs = None
                completion_loss = torch.tensor(0.)
                mask_loss = torch.tensor(0.)
                shape_example = None
                meshes = None
                iou_stats = None
            
            voxel_size = float(inputs['point_clouds'][0,:,2].max() - inputs['point_clouds'][0,:,2].min()) / 46

            '''fit mesh points to scans'''
            pred_mesh_dict = None
            if self.cfg.phase == 'completion' and self.cfg.generation.generate_mesh:
                pred_mesh_dict = {'meshes': meshes, 'proposal_ids': BATCH_PROPOSAL_IDs}
                parsed_predictions = self._fit_mesh_to_scan(pred_mesh_dict, parsed_predictions, eval_dict, inputs['point_clouds'], dump_threshold)
            pred_mesh_dict = pred_mesh_dict if evaluate_mesh_mAP else None
            eval_dict = assembly_pred_map_cls(eval_dict, parsed_predictions, self.eval_config, mesh_outputs=pred_mesh_dict, voxel_size=voxel_size)

            gt_mesh_dict = {'shapenet_catids': batch['shapenet_catids'],
                            'shapenet_ids': batch['shapenet_ids']} if evaluate_mesh_mAP else None
            eval_dict['batch_gt_map_cls'] = assembly_gt_map_cls(parsed_gts, mesh_outputs=gt_mesh_dict, voxel_size=voxel_size)

            completion_loss = torch.cat([completion_loss.unsqueeze(0), mask_loss.unsqueeze(0)], dim=0)
            #return end_points, completion_loss.unsqueeze(0), shape_example, BATCH_PROPOSAL_IDs, eval_dict, meshes, iou_stats, parsed_predictions
            
            # compute loss
            total_loss = self._compute_loss(
                (end_points, completion_loss.unsqueeze(0)), batch
            )
            end_points['losses'] = total_loss
            end_points['loss'] = total_loss['total']

            end_points['shape_example'] = shape_example
            end_points['BATCH_PROPOSAL_IDs'] = BATCH_PROPOSAL_IDs
            end_points['eval_dict'] = eval_dict
            end_points['parsed_predictions'] = parsed_predictions
            if meshes is not None:
                end_points['meshes'] = meshes
            if iou_stats is not None:
                end_points['iou_stats'] = iou_stats
            return end_points


    def _common_step(self, batch, batch_idx):
        r"""
        :return: 
        """
        from IPython import embed
        embed()
        inputs = {'point_clouds': batch['point_clouds']}
        x = torch.zeros(5)
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
        if_proposal_feature = (self.cfg.phase == 'completion')
        end_points, proposal_features = self.proposal(xyz, features, end_points, if_proposal_feature)

        eval_dict, parsed_predictions = parse_predictions(end_points, batch, self.eval_config)
        parsed_gts = parse_groundtruths(batch, self.eval_config)

        # completion
        evaluate_mesh_mAP = True if self.cfg.phase == 'completion' and self.cfg.generation.generate_mesh and self.cfg.generation.evaluate_mesh_mAP else False

        if self.cfg.phase == 'completion':
            # use 3D NMS to generate sample ids.
            batch_sample_ids = eval_dict['pred_mask']
            BATCH_PROPOSAL_IDs = self._get_proposal_id(end_points, batch, mode='objectness')

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
            cls_codes_for_completion = self._select_data(batch, BATCH_PROPOSAL_IDs)
            
            export_shape = batch.get('export_shape', self.cfg.generation.export_shape) # if output shape voxels.
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
        # end if self.cfg.phase == 'completion'

        completion_loss = torch.cat([completion_loss.unsqueeze(0), mask_loss.unsqueeze(0)], dim = 0)
        #return end_points, completion_loss.unsqueeze(0), shape_example, BATCH_PROPOSAL_IDs

        # compute loss
        total_loss = self._compute_loss(
            (end_points, completion_loss.unsqueeze(0)), batch
        )
        end_points['losses'] = total_loss
        end_points['loss'] = total_loss['total']
        return end_points
    

    def training_step(self, batch, batch_idx):
        out = self._common_step(batch, batch_idx)
        loss = out['losses']
        self.log('train_loss', out['loss'])
        if 'completion_loss' in loss:
            self.log('det_loss', loss['detection_loss'], prog_bar=True, on_step=True)
            self.log('train_comp_loss', loss['completion_loss'], prog_bar=True, on_step=True)
        return out


    def validation_step(self, batch, batch_idx):
        out = self._common_step(batch, batch_idx)
        loss = out['losses']
        self.log('val_loss', out['loss'], on_step=True)
        if 'completion_loss' in loss:
            self.log('det_loss', loss['detection_loss'], prog_bar=True, on_step=True)
            self.log('val_comp_loss', loss['completion_loss'], prog_bar=True, on_step=True)
        return out


    def test_step(self, batch, batch_idx):
        out = self(batch)
        loss = out['losses']
        self.log("test_loss", float(out['loss']), prog_bar=True, on_step=True)
        print(f'loss: {float(out["loss"])}', end=' ')
        if 'completion_loss' in loss:
            self.log('test_comp_loss', float(loss['completion_loss']), prog_bar=True, on_step=True)
            self.log('test_det_loss', float(loss['detection_loss']), prog_bar=True, on_step=True)
            print(f'det_loss: {float(out["detection_loss"])}', end=' ')
            print(f'comp_loss: {float(out["completion_loss"])}', end=' ')
                
        eval_metrics = {}
        if 'iou_stats' in out:
            cls_iou_stat = out['iou_stats']
            cls_iou_stat_out = {}
            for cls, iou in zip(cls_iou_stat['cls'], cls_iou_stat['iou']):
                if str(cls) + '_voxel_iou' not in cls_iou_stat_out:
                    cls_iou_stat_out[str(cls) + '_voxel_iou'] = []
                cls_iou_stat_out[str(cls) + '_voxel_iou'].append(iou)

            eval_metrics = {**eval_metrics, **cls_iou_stat_out}
            loss = {**loss, **eval_metrics}
        # end if
        eval_dict = out['eval_dict']
        for ap_calculator in self.ap_calculator_list:
            ap_calculator.step(eval_dict['batch_pred_map_cls'], eval_dict['batch_gt_map_cls'])
        # visualize intermediate results.
        if self.cfg.generation.dump_results:
            self._visualize_step(batch_idx, batch, out, eval_dict)


    def test_epoch_end(self, outputs):
         # Evaluate average precision
        for i, ap_calculator in enumerate(self.ap_calculator_list):
            # self.AP_IOU_THRESHOLDS[i])
            metrics_dict = ap_calculator.compute_metrics()
            for key in metrics_dict:
                print(f'eval {key}: {metrics_dict[key]}')
    

    def configure_optimizers(self):
        optimizers = schedulers = {}
        optim_cfg = self.cfg.optimizer

        if self.cfg.phase == 'detection' or \
            not hasattr(self.cfg, 'freeze') or \
            not self.cfg.freeze:
            optimizers['detection'] = optim.AdamW(
                list(self.backbone.parameters()) + 
                list(self.voting.parameters()) + 
                list(self.proposal.parameters()),
                lr=optim_cfg.detection.lr,
                betas=optim_cfg.detection.betas,
                eps=optim_cfg.detection.eps,
                weight_decay=optim_cfg.detection.weight_decay,
            )
        if self.cfg.phase == 'completion':
            optimizers['completion'] = optim.AdamW(
                list(self.skip_propagation.parameters()) + 
                list(self.completion.parameters()),
                lr=optim_cfg.completion.lr,
                betas=optim_cfg.completion.betas,
                eps=optim_cfg.completion.eps,
                weight_decay=optim_cfg.completion.weight_decay,
            )
        optimizers_list = [v for _, v in optimizers.items()]

        if self.cfg.phase == 'detection' or \
            not hasattr(self.cfg, 'freeze') or \
            not self.cfg.freeze:
            schedulers['detection'] = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizers['detection'],
                patience=optim_cfg.detection.patience,
                factor=optim_cfg.detection.factor,
                threshold=optim_cfg.detection.threshold,
            )
        if self.cfg.phase == 'completion':
            schedulers['completion'] = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizers['completion'],
                patience=optim_cfg.completion.patience,
                factor=optim_cfg.completion.factor,
                threshold=optim_cfg.completion.threshold,
            )
        schedulers_list = [ {'scheduler': v, 'monitor': 'val_loss'} for _, v in schedulers.items() ]
        return optimizers_list, schedulers_list


    def _compute_loss(self, est_data, gt_data):
        end_points, completion_loss = est_data[:2]
        detection_loss = self.detection_loss(end_points, gt_data, self.dataset_config, self.device)
        if self.cfg.phase == 'completion':
            completion_loss = self.completion_loss(completion_loss)
            total_loss = {**detection_loss,
                        'detection_loss': detection_loss['total'],
                        'completion_loss': completion_loss['completion_loss'], 
                        'mask_loss': completion_loss['mask_loss']}
            total_loss['total'] += completion_loss['total_loss']
        else:
            total_loss = detection_loss
        return total_loss


    def _get_proposal_id(self, end_points, data, mode='random', batch_sample_ids=None, DUMP_CONF_THRESH=-1.):
        '''
        Get the proposal ids for completion training for the limited GPU RAM.
        :param end_points: estimated data from votenet.
        :param data: data source which contains gt contents.
        :return:
        '''
        batch_size, MAX_NUM_OBJ = data['box_label_mask'].shape
        NUM_PROPOSALS = end_points['center'].size(1)
        object_limit_per_scene = 10
        proposal_id_list = []

        if mode == 'objectness' or batch_sample_ids is not None:
            objectness_probs = torch.softmax(end_points['objectness_scores'], dim=2)[..., 1]

        for batch_id in range(batch_size):
            box_mask = torch.nonzero(data['box_label_mask'][batch_id])
            gt_centroids = data['center_label'][batch_id, box_mask, 0:3].squeeze(1)
            dist1, object_assignment, _, _ = nn_distance(end_points['center'][batch_id].unsqueeze(0),
                                                         gt_centroids.unsqueeze(0))  # dist1: BxK, dist2: BxK2
            object_assignment = box_mask[object_assignment[0]].squeeze(-1)
            proposal_to_gt_box_w_cls = torch.cat(
                [torch.arange(0, NUM_PROPOSALS).unsqueeze(-1).long().to(self.device), object_assignment.unsqueeze(-1)],
                dim=-1)
            gt_classes = data['sem_cls_label'][batch_id][proposal_to_gt_box_w_cls[:, 1]]
            proposal_to_gt_box_w_cls = torch.cat([proposal_to_gt_box_w_cls, gt_classes.long().unsqueeze(-1)], dim=-1)

            if batch_sample_ids is None:
                if mode == 'random':
                    sample_ids = torch.multinomial(torch.ones(size=(NUM_PROPOSALS,)), object_limit_per_scene,
                                                   replacement=False)
                elif mode == 'nn':
                    sample_ids = torch.argsort(dist1[0])[:object_limit_per_scene]
                elif mode == 'objectness':
                    # sample_ids = torch.multinomial((objectness_probs[batch_id]>=self.eval_config['conf_thresh']).detach().cpu().float(), num_samples=object_limit_per_scene, replacement=True)
                    objectness_sort = torch.argsort(objectness_probs[batch_id], descending=True)
                    gt_ids = np.unique(proposal_to_gt_box_w_cls[objectness_sort, 1].detach().cpu().numpy(), return_index=True)[1]
                    gt_ids = np.hstack([gt_ids, np.setdiff1d(range(len(objectness_sort)), gt_ids, assume_unique=True)])[
                             :object_limit_per_scene]
                    sample_ids = objectness_sort[gt_ids]
                else:
                    raise NameError('Please specify a correct filtering mode.')
            else:
                sample_ids = (objectness_probs[batch_id] > DUMP_CONF_THRESH).detach().cpu().numpy()*batch_sample_ids[batch_id]

            proposal_to_gt_box_w_cls = proposal_to_gt_box_w_cls[sample_ids].long()
            proposal_id_list.append(proposal_to_gt_box_w_cls.unsqueeze(0))
        
        return torch.cat(proposal_id_list, dim=0)


    def _fit_mesh_to_scan(self, pred_mesh_dict, parsed_predictions, eval_dict, input_scan, dump_threshold):
        '''fit meshes to input scan'''
        pred_corners_3d_upright_camera = parsed_predictions['pred_corners_3d_upright_camera']
        pred_sem_cls = parsed_predictions['pred_sem_cls']
        bsize, N_proposals = pred_sem_cls.shape
        pred_mask = eval_dict['pred_mask']
        obj_prob = parsed_predictions['obj_prob']
        device = input_scan.device
        input_scan = input_scan.detach().cpu().numpy()
        transform_shapenet = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

        index_list = []
        box_params_list = []
        max_obj_points = 10000
        max_pc_in_box = 50000
        obj_points_list = []
        obj_points_mask_list = []
        pc_in_box_list = []
        pc_in_box_mask_list = []
        for i in range(bsize):
            for j in range(N_proposals):
                if not (pred_mask[i, j] == 1 and obj_prob[i, j] > dump_threshold):
                    continue
                # get mesh points
                mesh_data = pred_mesh_dict['meshes'][list(pred_mesh_dict['proposal_ids'][i,:,0]).index(j)]
                obj_points = mesh_data.vertices
                obj_points = obj_points - (obj_points.max(0) + obj_points.min(0)) / 2.
                obj_points = obj_points.dot(transform_shapenet.T)
                obj_points = obj_points / (obj_points.max(0) - obj_points.min(0))

                obj_points_matrix = np.zeros((max_obj_points, 3))
                obj_points_mask = np.zeros((max_obj_points,), dtype=np.uint8)
                obj_points_matrix[:obj_points.shape[0], :] = obj_points
                obj_points_mask[:obj_points.shape[0]] = 1

                # box corners
                box_corners_cam = pred_corners_3d_upright_camera[i, j]
                box_corners_depth = flip_axis_to_depth(box_corners_cam)
                # box vector form
                centroid = (np.max(box_corners_depth, axis=0) + np.min(box_corners_depth, axis=0)) / 2.
                forward_vector = box_corners_depth[1] - box_corners_depth[2]
                left_vector = box_corners_depth[0] - box_corners_depth[1]
                up_vector = box_corners_depth[6] - box_corners_depth[2]
                orientation = np.arctan2(forward_vector[1], forward_vector[0])
                sizes = np.linalg.norm([forward_vector, left_vector, up_vector], axis=1)
                box_params = np.array([*centroid, *sizes, orientation])

                # points in larger boxes (remove grounds)
                larger_box = flip_axis_to_depth(get_3d_box(1.2*sizes, -orientation, flip_axis_to_camera(centroid)))
                height = np.percentile(input_scan[i, :, 2], 5)
                scene_scan = input_scan[i, input_scan[i, :, 2] >= height, :3]
                pc_in_box, inds = extract_pc_in_box3d(scene_scan, larger_box)
                if len(pc_in_box) < 5:
                    continue

                pc_in_box = pc_in_box[:max_pc_in_box, :]
                pc_in_box_matrix = np.zeros((max_pc_in_box, 3))
                pc_in_box_mask = np.zeros((max_pc_in_box,), dtype=np.uint8)
                pc_in_box_matrix[:pc_in_box.shape[0], :] = pc_in_box
                pc_in_box_mask[:pc_in_box.shape[0]] = 1

                index_list.append((i, j))
                obj_points_list.append(obj_points_matrix)
                obj_points_mask_list.append(obj_points_mask)
                box_params_list.append(box_params)
                pc_in_box_list.append(pc_in_box_matrix)
                pc_in_box_mask_list.append(pc_in_box_mask)

        obj_points_list = np.array(obj_points_list)
        pc_in_box_list = np.array(pc_in_box_list)
        obj_points_mask_list = np.array(obj_points_mask_list)
        pc_in_box_mask_list = np.array(pc_in_box_mask_list)
        box_params_list = np.array(box_params_list)

        # scale to predicted sizes
        obj_points_list = obj_points_list * box_params_list[:, np.newaxis, 3:6]

        obj_points_list = torch.from_numpy(obj_points_list).to(device).float()
        pc_in_box_list = torch.from_numpy(pc_in_box_list).to(device).float()
        pc_in_box_mask_list = torch.from_numpy(pc_in_box_mask_list).to(device).float()
        '''optimize box center and orientation'''
        centroid_params = box_params_list[:, :3]
        orientation_params = box_params_list[:, 6]
        centroid_params = torch.from_numpy(centroid_params).to(device).float()
        orientation_params = torch.from_numpy(orientation_params).to(device).float()
        centroid_params.requires_grad = True
        orientation_params.requires_grad = True
        with torch.enable_grad():
            lr = 0.01
            iterations = 100
            optimizer = optim.Adam([centroid_params, orientation_params], lr=lr)
            centroid_params_cpu, orientation_params_cpu, best_loss = None, None, 1e6
            for iter in range(iterations):
                optimizer.zero_grad()
                loss = self._chamfer_dist(
                    obj_points_list, obj_points_mask_list, 
                    pc_in_box_list, pc_in_box_mask_list,
                    centroid_params, orientation_params
                )
                if loss < best_loss:
                    centroid_params_cpu = centroid_params.data.detach().cpu().numpy()
                    orientation_params_cpu = orientation_params.data.detach().cpu().numpy()
                    best_loss = loss
                loss.backward()
                optimizer.step()

        for idx in range(box_params_list.shape[0]):
            i, j = index_list[idx]
            best_box_corners_cam = get_3d_box(box_params_list[idx, 3:6], -orientation_params_cpu[idx], flip_axis_to_camera(centroid_params_cpu[idx]))
            pred_corners_3d_upright_camera[i, j] = best_box_corners_cam

        parsed_predictions['pred_corners_3d_upright_camera'] = pred_corners_3d_upright_camera
        return parsed_predictions
    
    def _chamfer_dist(self, obj_points, obj_points_masks, pc_in_box, pc_in_box_masks, centroid_params, orientation_params):
        b_s = obj_points.size(0)
        axis_rectified = torch.zeros(size=(b_s, 3, 3)).to(obj_points.device)
        axis_rectified[:, 2, 2] = 1
        axis_rectified[:, 0, 0] = torch.cos(orientation_params)
        axis_rectified[:, 0, 1] = torch.sin(orientation_params)
        axis_rectified[:, 1, 0] = -torch.sin(orientation_params)
        axis_rectified[:, 1, 1] = torch.cos(orientation_params)
        obj_points_after = torch.bmm(obj_points, axis_rectified) + centroid_params.unsqueeze(-2)
        dist1, dist2 = chamfer_func(obj_points_after, pc_in_box)
        return torch.mean(dist2 * pc_in_box_masks)*1e3

    def _visualize_step(self, batch_idx, gt_data, our_data, eval_dict, inference_switch=False):
        ''' Performs a visualization step.
        '''
        split_file = os.path.join('datasets/splits/fullscan', 'scannetv2_' + self.cfg.mode + '.json')
        scene_name = read_json_file(split_file)[gt_data['scan_idx']]['scan'].split('/')[3]
        dump_dir = os.path.join(self.cfg.generation.dump_path, '%s_%s_%s'%(self.cfg.phase, batch_idx, scene_name))
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        DUMP_CONF_THRESH = 0.5 # Dump boxes with obj prob larger than that.
        batch_id = 0

        '''Predict meshes'''
        pred_sem_cls = our_data['parsed_predictions']['pred_sem_cls'][batch_id].detach().cpu().numpy()
        if our_data['meshes'] is not None:
            meshes = our_data['meshes']
            BATCH_PROPOSAL_IDs = our_data['BATCH_PROPOSAL_IDs'][0].detach().detach().cpu().numpy()
            for mesh_data, map_data in zip(meshes, BATCH_PROPOSAL_IDs):
                str_nums = (map_data[0], map_data[1], pred_sem_cls[map_data[0]])
                object_mesh = os.path.join(dump_dir, 'proposal_%d_target_%d_class_%d_mesh.ply' % str_nums)
                mesh_data.export(object_mesh)
        else:
            BATCH_PROPOSAL_IDs = np.empty(0)

        '''Predict boxes'''
        est_data = our_data
        pred_corners_3d_upright_camera = our_data['parsed_predictions']['pred_corners_3d_upright_camera']
        objectness_prob = our_data['parsed_predictions']['obj_prob'][batch_id]

        # INPUT
        point_clouds = gt_data['point_clouds'].detach().cpu().numpy()

        # NETWORK OUTPUTS
        seed_xyz = est_data['seed_xyz'].detach().detach().cpu().numpy()  # (B,num_seed,3)
        if 'vote_xyz' in est_data:
            # aggregated_vote_xyz = est_data['aggregated_vote_xyz'].detach().detach().cpu().numpy()
            # vote_xyz = est_data['vote_xyz'].detach().detach().cpu().numpy()  # (B,num_seed,3)
            aggregated_vote_xyz = est_data['aggregated_vote_xyz'].detach().detach().cpu().numpy()

        box_corners_cam = pred_corners_3d_upright_camera[batch_id]
        box_corners_depth = flip_axis_to_depth(box_corners_cam)
        centroid = (np.max(box_corners_depth, axis=1) + np.min(box_corners_depth, axis=1)) / 2.

        forward_vector = box_corners_depth[:,1] - box_corners_depth[:,2]
        left_vector = box_corners_depth[:,0] - box_corners_depth[:,1]
        up_vector = box_corners_depth[:,6] - box_corners_depth[:,2]
        orientation = np.arctan2(forward_vector[:,1], forward_vector[:,0])
        forward_size = np.linalg.norm(forward_vector, axis=1)
        left_size = np.linalg.norm(left_vector, axis=1)
        up_size = np.linalg.norm(up_vector, axis=1)
        sizes = np.vstack([forward_size, left_size, up_size]).T

        box_params = np.hstack([centroid, sizes, orientation[:,np.newaxis]])

        # OTHERS
        pred_mask = eval_dict['pred_mask']  # B,num_proposal

        pc = point_clouds[batch_id, :, :]

        # Dump various point clouds
        pc_util.write_ply(pc, os.path.join(dump_dir, '%06d_pc.ply' % (batch_id)))
        pc_util.write_ply(seed_xyz[batch_id, :, :], os.path.join(dump_dir, '%06d_seed_pc.ply' % (batch_id)))
        if 'vote_xyz' in est_data:
            pc_util.write_ply(est_data['vote_xyz'][batch_id, :, :],
                              os.path.join(dump_dir, '%06d_vgen_pc.ply' % (batch_id)))
            pc_util.write_ply(aggregated_vote_xyz[batch_id, :, :],
                              os.path.join(dump_dir, '%06d_aggregated_vote_pc.ply' % (batch_id)))
        pc_util.write_ply(box_params[:, 0:3], os.path.join(dump_dir, '%06d_proposal_pc.ply' % (batch_id)))
        if np.sum(objectness_prob > DUMP_CONF_THRESH) > 0:
            pc_util.write_ply(box_params[objectness_prob > DUMP_CONF_THRESH, 0:3],
                              os.path.join(dump_dir, '%06d_confident_proposal_pc.ply' % (batch_id)))

        # Dump predicted bounding boxes
        if np.sum(objectness_prob > DUMP_CONF_THRESH) > 0:
            num_proposal = box_params.shape[0]
            if len(box_params) > 0:
                pc_util.write_oriented_bbox(box_params[objectness_prob > DUMP_CONF_THRESH, :],
                                            os.path.join(dump_dir, '%06d_pred_confident_bbox.ply' % (batch_id)))
                pc_util.write_oriented_bbox(
                    box_params[np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[batch_id, :] == 1), :],
                    os.path.join(dump_dir, '%06d_pred_confident_nms_bbox.ply' % (batch_id)))
                pc_util.write_oriented_bbox(box_params[pred_mask[batch_id, :] == 1, :],
                                            os.path.join(dump_dir, '%06d_pred_nms_bbox.ply' % (batch_id)))
                pc_util.write_oriented_bbox(box_params, os.path.join(dump_dir, '%06d_pred_bbox.ply' % (batch_id)))

                save_path = os.path.join(dump_dir, '%06d_pred_confident_nms_bbox.npz' % (batch_id))
                np.savez(
                    save_path, 
                    obbs=box_params[np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[batch_id, :] == 1), :],
                    proposal_map = BATCH_PROPOSAL_IDs
                )

        # Return if it is at inference time. No dumping of groundtruths
        if inference_switch:
            return

        objectness_loss, objectness_label, objectness_mask, object_assignment = \
            compute_objectness_loss(est_data, gt_data, self.device)

        # LABELS
        gt_center = gt_data['center_label'].detach().cpu().numpy()  # (B,MAX_NUM_OBJ,3)
        gt_mask = gt_data['box_label_mask'].detach().cpu().numpy()  # B,K2
        gt_heading_class = gt_data['heading_class_label'].detach().cpu().numpy()  # B,K2
        gt_heading_residual = gt_data['heading_residual_label'].detach().cpu().numpy()  # B,K2
        gt_size_class = gt_data['size_class_label'].detach().cpu().numpy()  # B,K2
        gt_size_residual = gt_data['size_residual_label'].detach().cpu().numpy()  # B,K2,3
        objectness_label = objectness_label.detach().detach().cpu().numpy()  # (B,K,)
        objectness_mask = objectness_mask.detach().detach().cpu().numpy()  # (B,K,)

        if np.sum(objectness_label[batch_id, :]) > 0:
            pc_util.write_ply(box_params[objectness_label[batch_id, :] > 0, 0:3],
                              os.path.join(dump_dir, '%06d_gt_positive_proposal_pc.ply' % (batch_id)))
        if np.sum(objectness_mask[batch_id, :]) > 0:
            pc_util.write_ply(box_params[objectness_mask[batch_id, :] > 0, 0:3],
                              os.path.join(dump_dir, '%06d_gt_mask_proposal_pc.ply' % (batch_id)))
        pc_util.write_ply(gt_center[batch_id, :, 0:3], os.path.join(dump_dir, '%06d_gt_centroid_pc.ply' % (batch_id)))
        pc_util.write_ply_color(box_params[:, 0:3], objectness_label[batch_id, :],
                                os.path.join(dump_dir, '%06d_proposal_pc_objectness_label.ply' % (batch_id)))

        # Dump GT bounding boxes
        obbs = []
        for j in range(gt_center.shape[1]):
            if gt_mask[batch_id, j] == 0: continue
            obb = self.dataset_config.param2obb(gt_center[batch_id, j, 0:3], gt_heading_class[batch_id, j], gt_heading_residual[batch_id, j],
                                   gt_size_class[batch_id, j], gt_size_residual[batch_id, j])
            obbs.append(obb)
        if len(obbs) > 0:
            obbs = np.vstack(tuple(obbs))  # (num_gt_objects, 7)
            pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%06d_gt_bbox.ply' % (batch_id)))

        # OPTIONALL, also dump prediction and gt details
        if 'batch_pred_map_cls' in eval_dict:
            fout = open(os.path.join(dump_dir, '%06d_pred_map_cls.txt' % (batch_id)), 'w')
            for t in eval_dict['batch_pred_map_cls'][batch_id]:
                fout.write(str(t[0]) + ' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write(' ' + str(t[2]))
                fout.write('\n')
            fout.close()
        if 'batch_gt_map_cls' in eval_dict:
            fout = open(os.path.join(dump_dir, '%06d_gt_map_cls.txt' % (batch_id)), 'w')
            for t in eval_dict['batch_gt_map_cls'][batch_id]:
                fout.write(str(t[0]) + ' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write('\n')
            fout.close()


    def _select_data(self, data, BATCH_PROPOSAL_IDs):
        '''
        Select those proposals that have a corresponding gt object shape (to the gt boxes.)
        :param data: data source which contains gt contents.
        :param BATCH_PROPOSAL_IDs: mapping list from proposal ids to gt box ids.
        :return:
        '''
        batch_size, n_objects, n_points, point_dim = data['object_points'].size()
        N_proposals = BATCH_PROPOSAL_IDs.size(1)
        object_ids = BATCH_PROPOSAL_IDs[:, :, 1].unsqueeze(-1).unsqueeze(-1).expand(batch_size, N_proposals,
                                                                                   n_points, point_dim)
        input_points_for_completion = torch.gather(data['object_points'], 1, object_ids)
        input_points_for_completion = input_points_for_completion.view(batch_size * N_proposals,
                                                                       n_points,
                                                                       point_dim)
        occ_ids = BATCH_PROPOSAL_IDs[:, :, 1].unsqueeze(-1).expand(batch_size, N_proposals, n_points)
        input_points_occ_for_completion = torch.gather(data['object_points_occ'], 1, occ_ids)
        input_points_occ_for_completion = input_points_occ_for_completion.view(batch_size * N_proposals,
                                                                               n_points)
        cls_codes_for_completion = []
        for batch_id in range(batch_size):
            # class encoding
            cls_codes = torch.zeros([N_proposals, self.dataset_config.num_class]).to(self.device)
            cls_codes[range(N_proposals), BATCH_PROPOSAL_IDs[batch_id, :, 2]] = 1

            cls_codes_for_completion.append(cls_codes)

        cls_codes_for_completion = torch.cat(cls_codes_for_completion, dim=0)

        return input_points_for_completion, \
               input_points_occ_for_completion, cls_codes_for_completion

