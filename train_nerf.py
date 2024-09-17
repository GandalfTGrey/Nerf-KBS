from __future__ import annotations

import tyro
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import method_configs, descriptions
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.vanilla_nerf import VanillaModelConfig, NeRFModel
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.plugins.registry import discover_methods
from scripts.train import main

import sys
sys.path.append("..")
from Zchenghuan.data.vkitti_dataparser import vkittiDataParserConfig


def train_nerf():
    method_configs["vanilla-nerf"] = TrainerConfig(
        method_name="vanilla-nerf",
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                # dataparser=BlenderDataParserConfig(),
                # dataparser=NerfstudioDataParserConfig(
                # ),
                dataparser=vkittiDataParserConfig(),
            ),
            model=VanillaModelConfig(_target=NeRFModel),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": None,
            },
            "temporal_distortion": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 16),
        vis="tensorboard",

    )
    external_methods, external_descriptions = discover_methods()
    method_configs.update(external_methods)
    descriptions.update(external_descriptions)

    AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[
        # Don't show unparseable (fixed) arguments in helptext.
        tyro.conf.FlagConversionOff[
            tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
        ]
    ]

    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    train_nerf()

