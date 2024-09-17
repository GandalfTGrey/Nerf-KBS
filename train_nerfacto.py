from __future__ import annotations

import tyro
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import method_configs, descriptions
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.registry import discover_methods
from nerfstudio.scripts.train import main

import sys
sys.path.append("..")
from data.KITTI_MOT import kittiDataParserConfig
from data.vkitti_dataparser import vkittiDataParserConfig


def train_nerfacto():

       method_configs["nerfacto"] = TrainerConfig(
           method_name="nerfacto",
           steps_per_eval_batch=500,
           steps_per_save=2000,
           max_num_iterations=30000,
           steps_per_eval_image=500,
           mixed_precision=False,
           pipeline=VanillaPipelineConfig(
               datamanager=VanillaDataManagerConfig(
                   # dataparser=NerfstudioDataParserConfig(),
                   # dataparser=vkittiDataParserConfig(),
                   dataparser=kittiDataParserConfig(),
                   train_num_rays_per_batch=4096,
                   eval_num_rays_per_batch=4096,
                   camera_optimizer=CameraOptimizerConfig(
                       mode="off",
                   ),
                   # train_num_images_to_sample_from=10,
                   # train_num_times_to_repeat_images=100,
                   # eval_num_images_to_sample_from=10,
                   # eval_num_times_to_repeat_images=100,
               ),
               model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
           ),
           optimizers={
               "proposal_networks": {
                   "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                   "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.00001, max_steps=2000000),
               },
               "fields": {
                   "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                   "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.00001, max_steps=2000000),
               },
           },
           viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
           vis="viewer+tensorboard",
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
    train_nerfacto()