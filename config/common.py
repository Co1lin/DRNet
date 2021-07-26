from typing import Any, List
from hydra.core.config_store import ConfigStore
from dataclasses import MISSING, dataclass, field

@dataclass
class BasicConfig:
    phase: str = 'detection'
    # device
    gpus: list = field(default_factory=lambda: [1,2])
    num_workers: int = 16   # cpu
    
    # reproducibility
    seed: int = 100
    
    # resume or finetuning
    weight: str = ''
    weight_mode: str = ''  # 'resume' or 'finetune'
    
    # data
    dataset: str = 'datasets'
    batch_size: int = 4
    
    # optimizer
    lr: float = 1e-3
    betas: list = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    # scheduler
    patience: int = 20
    scheduler_factor: float = 0.1
    scheduler_threshold: float = 0.01

    # options
    num_point: int = 80000
    num_target: int = 256
    ap_iou_thresholds: list = field(default_factory=lambda: [0.5])
    
    cluster_sampling: str = 'vote_fps'
    remove_empty_box: bool = False
    use_3d_nms: bool = True
    nms_iou: float = 0.25
    use_old_type_nms: bool = False
    cls_nms: bool = True
    per_class_proposal: bool = True
    conf_thresh: float = 0.05
    use_cls_for_completion: bool = False
    dump_threshold: float = 0.5
    export_shape: bool = False
    evaluate_mesh_mAP: bool = False
    generate_mesh: bool = False

@dataclass
class TrainConfig(BasicConfig):
    mode: str = 'train'
    # plan
    epochs: int = 480
    
@dataclass
class TestConfig(BasicConfig):
    mode: str = 'test'
    # options
    evaluate_mesh_mAP: bool = False
    generate_mesh: bool = False
    dump_results: bool = False
    dump_path: str = 'visualization'

@dataclass
class TrainDetConfig(TrainConfig):
    phase: str = 'detection'

@dataclass
class TrainCompConfig(TrainConfig):
    phase: str = 'completion'

@dataclass
class TrainAllConfig(TrainConfig):
    phase: str = 'completion'

@dataclass
class TestDetConfig(TrainConfig):
    phase: str = 'detection'

@dataclass
class TestCompConfig(TrainConfig):
    phase: str = 'completion'

@dataclass
class Config:
    config: BasicConfig = MISSING

    defaults: list = field(default_factory=lambda: [
        { 'config': 'test_comp', },
    ])

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="config", name="train_det", node=TrainDetConfig)
cs.store(group="config", name="train_comp", node=TrainCompConfig)
cs.store(group="config", name="test_det", node=TestDetConfig)
cs.store(group="config", name="test_comp", node=TrainCompConfig)