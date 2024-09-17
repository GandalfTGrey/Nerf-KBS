import faulthandler
import signal
from pathlib import Path

import tyro
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig, MachineConfig
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import descriptions, method_configs
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
# from nerfstudio.data.datamanagers.depth_datamanager import DepthDataManagerConfig
# from nerfstudio.data.datamanagers.semantic_datamanager import SemanticDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.nuscenes_dataparser import NuScenesDataParserConfig
from nerfstudio.data.dataparsers.sitcoms3d_dataparser import Sitcoms3DDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig,RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig, DepthNerfactoModel
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
# from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.vanilla_nerf import VanillaModelConfig, NeRFModel
# from nerfstudio.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.registry import discover_methods
# from .scripts.train import main
from nerfstudio.scripts.train import main

import sys
sys.path.append("..")
from data.vkitti_dataparser import vkittiDataParserConfig
# from Zchenghuan.data.kitti_odom import kittiDataParserConfig
from nerfacto import NerfactoModelConfig
from semantic_nerfw import SemanticNerfWModelConfig
from data.KITTI_MOT import kittiDataParserConfig
from data.semantic_datamanager import SemanticDataManagerConfig, SemanticDataManager

# from depth_flow_nerfacto import DepthFlowNerfactoModelConfig
# from process_data.kittiDataParserConfig import kittiDataParserConfig

'''
depth-nerfacto
--data
/home/chenghuan/suds/Zchenghuan/sequences/fill_img5
'''

def z_entrypoint():

    # faulthandler.register(signal.SIGUSR1)
    # method_configs["nerfacto-big"] = TrainerConfig(
    #     method_name="nerfacto",
    #     steps_per_eval_batch=500,
    #     steps_per_save=2000,
    #     max_num_iterations=100000,
    #     mixed_precision=True,
    #     pipeline=VanillaPipelineConfig(
    #         datamanager=VanillaDataManagerConfig(f
    #         #     dataparser=NerfstudioDataParserConfig(),
    #             dataparser=kittiDataParserConfig(),
    #             train_num_rays_per_batch=4096,#8192,
    #             eval_num_rays_per_batch=4096,
    #             camera_optimizer=CameraOptimizerConfig(mode="off",
    #                              optimizer=AdamOptimizerConfig(lr=1e-5, eps=1e-15),
    #                              scheduler=ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=5000),
    # ),
    #         ),
    #         model=NerfactoModelConfig(
    #             eval_num_rays_per_chunk=1 << 15,
    #             num_nerf_samples_per_ray=128,
    #             num_proposal_samples_per_ray=(512, 256),
    #             hidden_dim=128,
    #             hidden_dim_color=128,
    #             # appearance_embed_dim=128,
    #             max_res=4096,
    #             proposal_weights_anneal_max_num_iters=5000,
    #             log2_hashmap_size=21,
    #             # camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
    #         ),
    #     ),
    #     optimizers={
    #         "proposal_networks": {
    #             "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
    #             "scheduler": None,
    #         },
    #         "fields": {
    #             "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
    #             "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=50000),
    #         },
    
    #     },
    #     viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    #     vis="viewer+tensorboard",
    # )
    
    
    
    
    method_configs["test-nerfacto"] = TrainerConfig(
        machine=MachineConfig(num_devices=1, num_machines=1, machine_rank=0, device_type="cuda:2"),
        method_name="test-nerfacto",
        steps_per_eval_batch=5000,
        steps_per_save=2000,
        steps_per_eval_image=5000,
        steps_per_eval_all_images=10000,
        max_num_iterations=20000,#600000
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                # dataparser= vkittiDataParserConfig(
                #     first_frame=1,
                #     last_frame=140,
                #     use_depth=False,
                #     # use_semantic=True,
                #     # use_mask=True,
                #     split_setting= "nvs-75",
                # ),
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.75),
                camera_res_scale_factor = 1,
                train_num_rays_per_batch=4096,#4096,
                eval_num_rays_per_batch=4096,#4096,
                ),
            
            model=NerfactoModelConfig(
                                      # eval_num_rays_per_chunk=1 << 15,
                                      # num_nerf_samples_per_ray=128,
                                      # num_proposal_samples_per_ray=(512, 256),
                                      # hidden_dim=128,
                                      # hidden_dim_color=128,
                                      # # appearance_embed_dim=128,
                                      # max_res=4096,
                                      # proposal_weights_anneal_max_num_iters=5000,
                                      # log2_hashmap_size=21,
                                      )
            # model=SemanticNerfWModelConfig(eval_num_rays_per_chunk=1 << 16,
            #                                # num_nerf_samples_per_ray=97,
            #                                near_plane=0.001,
            #
            #                                ),
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
        viewer=ViewerConfig(num_rays_per_chunk=1 << 16),
        vis="viewer",
    )

    '''
    method_configs["depth-nerfacto"] = TrainerConfig(
        method_name="depth_flow_nerfacto",
        steps_per_eval_image=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
            datamanager=DepthDataManagerConfig(
                dataparser=kittiDataParserConfig(),
                dataparser=NerfstudioDataParserConfig(
                    depth_unit_scale_factor=1,
                    # auto_scale_poses=False
                    # orientation_method="none",
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(mode="off",),
            ),
            model=DepthFlowNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15,
                                               # far_plane=10.0,
                                               ),
            # model=DepthNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15,
            #                                #far_plane=100.0
            #    ,                           ),
            # model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    
            ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                'scheduler': None
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
        )
    )

    '''
    


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
    z_entrypoint()

