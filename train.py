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
from nerfstudio.scripts.train import main

from data.vkitti_dataparser import vkittiDataParserConfig
from semantic_nerfw import SemanticNerfWModelConfig
from data.KITTI_MOT import kittiDataParserConfig
from data.semantic_datamanager import SemanticDataManagerConfig
from nerfacto import NerfactoModelConfig
# from depth_flow_nerfacto import DepthFlowNerfactoModelConfig
# from process_data.kittiDataParserConfig import kittiDataParserConfig

'''
depth-nerfacto
--data
/home/chenghuan/suds/Zchenghuan/sequences/fill_img5
/media/data/chenghuan/mars/vkitti/Scene06/clone
'''
method_configs["semantic-nerfw"] = TrainerConfig(
    method_name="semantic-nerfw",
    machine=MachineConfig(num_devices=1, num_machines=1, machine_rank=0, device_type="cuda:2"),
    steps_per_eval_batch=500,
    steps_per_save=2000,
    steps_per_eval_image=500,
    steps_per_eval_all_images=10000,
    max_num_iterations=30000,#300000
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=SemanticDataManagerConfig(
            dataparser=kittiDataParserConfig(
                first_frame=5,
                last_frame=120,
                use_depth=False,
                use_semantic=True,
                use_mask=True,
                # scale_factor=0.1,
                split_setting= "nvs-75",
                image_height=375,
                image_width=1242,
            ),
            
            # dataparser = vkittiDataParserConfig(
            #     first_frame=0,
            #     last_frame=230,
            # )
            camera_res_scale_factor = 1,
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(mode="off"),
            # camera_optimizer=CameraOptimizerConfig(mode="off"),
            

        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 16),
        # model=SemanticNerfWModelConfig(eval_num_rays_per_chunk=1 << 16,
        #                                use_depth=False,
        #                                use_semantic=False,
        #                                 # num_nerf_samples_per_ray=97,
        #                                 # near_plane=0.001,
        #                                 ),
    ),
    optimizers={
        "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": None,

        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": None,
            # "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
            # "scheduler": None,
        },

    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 16),
    vis="viewer",
)

def z_entrypoint():
    faulthandler.register(signal.SIGUSR1)
    # method_configs["semantic-nerfw"] = TrainerConfig(
    #     method_name="semantic-nerfw",
    #     steps_per_eval_batch=500,
    #     steps_per_save=2000,
    #     steps_per_eval_image=500,
    #     steps_per_eval_all_images=29990,
    #     max_num_iterations=30000,#300000
    #     mixed_precision=True,
    #     pipeline=VanillaPipelineConfig(
    #         datamanager=SemanticDataManagerConfig(
    #         # datamanager=VanillaDataManagerConfig(
    #         #     dataparser=vkittiDataParserConfig(),
    #             # dataparser=kittiDataParserConfig()
    #             dataparser=kittiDataParserConfig(
    #                 first_frame=0,
    #                 last_frame=230,
    #                 use_depth=True,
    #                 use_semantic=True,
    #                 use_mask=True,
    #                 # scale_factor=0.1,
    #                 split_setting= "nvs-75"
    #             ),
    #             train_num_rays_per_batch=4096,
    #             eval_num_rays_per_batch=4096,
    #             camera_optimizer=CameraOptimizerConfig(mode="off"),

    #         ),
    #         # model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 16),
    #         model=SemanticNerfWModelConfig(eval_num_rays_per_chunk=1 << 16,
    #                                        # num_nerf_samples_per_ray=97,
    #                                        # near_plane=0.001,
    #                                        use_depth=True,
    #                                        ),
    #     ),
    #     optimizers={
    #         "proposal_networks": {
    #              "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
    #               "scheduler": None,

    #         },
    #         "fields": {
    #             "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
    #             "scheduler": None,
    #             # "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
    #             # "scheduler": None,
    #         },

    #     },
    #     viewer=ViewerConfig(num_rays_per_chunk=1 << 16),
    #     vis="viewer+tensorboard",
    # )

    # method_configs["depth-nerfacto"] = TrainerConfig(
    #     method_name="depth_flow_nerfacto",
    #     steps_per_eval_image=500,
    #     steps_per_save=2000,
    #     max_num_iterations=30000,
    #     mixed_precision=True,
    #     pipeline=VanillaPipelineConfig(
    #         datamanager=VanillaDataManagerConfig(
    #         datamanager=DepthDataManagerConfig(
    #             dataparser=kittiDataParserConfig(),
    #             dataparser=NerfstudioDataParserConfig(
    #                 depth_unit_scale_factor=1,
    #                 # auto_scale_poses=False
    #                 # orientation_method="none",
    #             ),
    #             train_num_rays_per_batch=4096,
    #             eval_num_rays_per_batch=4096,
    #             camera_optimizer=CameraOptimizerConfig(mode="off",),
    #         ),
    #         model=DepthFlowNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15,
    #                                            # far_plane=10.0,
    #                                            ),
    #         # model=DepthNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15,
    #         #                                #far_plane=100.0
    #         #                                ),
    #         # model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    
    #     ),
    #     optimizers={
    #         "proposal_networks": {
    #             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
    #             'scheduler': None
    #         },
    #         "fields": {
    #             "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
    #             "scheduler": None,
    #         },
    #     },
    #     # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    #     vis="tensorboard",
    # )
    # )

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

