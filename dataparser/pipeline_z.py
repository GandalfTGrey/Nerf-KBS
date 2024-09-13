from dataclasses import dataclass, field
from typing import Type, Literal

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig, VanillaPipeline


@dataclass
class myPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda:myPipeline)
    """target class to instantiate"""
    datamanager: SUDSDataManagerConfig = SUDSDataManagerConfig()
    """specifies the datamanager config"""
    model: SUDSModelConfig = SUDSModelConfig()
    """specifies the model config"""

class myPipeline(VanillaPipeline):
    config: myPipelineConfig

    def __init__(
            self,
            config: myPipelineConfig,
            device: str,
            test_mode: Literal['test', 'val', 'inference'] = 'val',
            world_size: int = 1,
            local_rank: int = 0):