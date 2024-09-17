# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from nerfstudio.utils.math import normalized_depth_scale_and_shift
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss, ScaleAndShiftInvariantLoss,
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
    SemanticRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps

# from Zchenghuan.data.KITTI_MOT import kitti, kittiDataParserConfig
from local_tensorfs import LocalTensorfs
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from loss import get_pred_flow, get_fwd_bwd_cam2cams, decode_flow

from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs, Semantics,
)

@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    # _target: Type = field(default_factory=lambda: NerfactoModel)
    # near_plane: float = 0.001#0.01
    # """How far along the ray to start sampling."""
    # far_plane: float = 1000.0#1000.0
    # """How far along the ray to stop sampling."""
    # background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    # """Whether to randomize the background color."""
    # hidden_dim: int = 64
    # """Dimension of hidden layers"""
    # hidden_dim_color: int = 64
    # """Dimension of hidden layers for color network"""
    # hidden_dim_transient: int = 64
    # """Dimension of hidden layers for transient network"""
    # num_levels: int = 16
    # """Number of levels of the hashmap for the base mlp."""
    # max_res: int = 2048
    # """Maximum resolution of the hashmap for the base mlp."""
    # log2_hashmap_size: int = 19
    # """Size of the hashmap for the base mlp"""
    # num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)#(256, 96)
    # """Number of samples per ray for each proposal network."""
    # num_nerf_samples_per_ray: int = 48#48,97
    # """Number of samples per ray for the nerf network."""
    # proposal_update_every: int = 5
    # """Sample every n steps after the warmup"""
    # proposal_warmup: int = 5000
    # """Scales n from 1 to proposal_update_every over this many steps"""
    # num_proposal_iterations: int = 2
    # """Number of proposal network iterations."""
    # use_same_proposal_network: bool = False
    # """Use the same proposal network. Otherwise use different ones."""
    # proposal_net_args_list: List[Dict] = field(
    #     default_factory=lambda: [
    #         {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
    #         {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
    #     ]
    # )
    # """Arguments for the proposal density fields."""
    # proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    # """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    # flow_loss_mult: float = 0.001
    # interlevel_loss_mult: float = 1
    # """Proposal loss multiplier."""
    # distortion_loss_mult: float = 0.002
    # """Distortion loss multiplier."""
    # orientation_loss_mult: float = 0.0001
    # """Orientation loss multiplier on computed normals."""
    # pred_normal_loss_mult: float = 0.001
    # """Predicted normal loss multiplier."""
    # use_proposal_weight_anneal: bool = True
    # """Whether to use proposal weight annealing."""
    # use_average_appearance_embedding: bool = True
    # """Whether to use average appearance embedding or zeros for inference."""
    # proposal_weights_anneal_slope: float = 10.0
    # """Slope of the annealing function for the proposal weights."""
    # proposal_weights_anneal_max_num_iters: int = 1000
    # """Max num iterations for the annealing function."""
    # use_single_jitter: bool = True
    # """Whether use single jitter or not for the proposal networks."""
    # predict_normals: bool = False
    # """Whether to predict normals or not."""
    # disable_scene_contraction: bool = False
    # """Whether to disable scene contraction or not."""
    # semantic_loss_weight: float = 0.001
    # mono_depth_loss_mult: float = 0.001
    # is_euclidean_depth: bool = True
    # pass_semantic_gradients: bool = False
    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.001
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    pass_semantic_gradients: bool = False
    mono_depth_loss_mult:float = 0.01
    is_euclidean_depth:bool = False
    use_depth:bool = False
    use_semantic:bool = False
    use_mask:bool = False




class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig
    def __init__(
        self,
        config: NerfactoModelConfig,
        metadata: Dict,
        **kwargs,
    ) -> None:
        self.field = None
        self.use_mask = config.use_mask
        self.use_depth = config.use_depth
        self.use_semantic = config.use_semantic
        super().__init__(config=config, **kwargs)

    # def __init__(self, config: NerfactoModelConfig, metadata: Dict, **kwargs) -> None:


        # self.dataparser_outputs = kitti(kittiDataParserConfig()).get_dataparser_outputs()
        # self.metadata = self.dataparser_outputs.metadata


        if self.use_semantic:
            assert "semantics" in metadata.keys() and isinstance(metadata["semantics"], Semantics)
            self.semantics = metadata["semantics"]
            self.semantic_num = len(self.semantics.classes)
            super().__init__(config=config, **kwargs)
            self.colormap = self.semantics.colors.clone().detach().to(self.device)
            self.color2label = {tuple(color.tolist()): i for i, color in enumerate(self.colormap)}
            # map Van to Car
            # for i, sem_class in enumerate(self.semantics.classes):
            #     if sem_class == "Car":
            #         self.color2label[(0, 139, 139)] = i
            # self.str2semantic = {label: i for i, label in enumerate(self.semantics.classes)}
    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            use_semantics=self.use_semantic,
            num_semantic_classes=len(self.semantics.classes) if self.use_semantic else None,
            pass_semantic_gradients=self.config.pass_semantic_gradients,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()
        self.renderer_semantics = SemanticRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()
        if self.use_semantic:
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=self.semantic_num)
        self.mono_depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)


        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "directions": ray_bundle.directions,
            "origins" : ray_bundle.origins
        }

        if self.use_semantic:
            # semantics
            semantic_weights = weights
            if not self.config.pass_semantic_gradients:
                semantic_weights = semantic_weights.detach()
            outputs["semantics"] = self.renderer_semantics(
                field_outputs[FieldHeadNames.SEMANTICS], weights=semantic_weights
            )
        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)

        print("PSNR: {}".format(metrics_dict["psnr"]))
        # if self.training:
        #     metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        # return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        ############
        # dataparser_outputs= DataparserOutputs
        #### self.flow_image是所有光流的缓存 每次从所有图片中随机采样像素




        # coords = coords.reshape(coords.shape[:2] + (1,) * len(camera_indices.shape[:-1]) + (2,))  # (h, w, 1..., 2)
        # coords = coords.expand(coords.shape[:2] + camera_indices.shape[:-1] + (2,))  # (h, w, num_rays, 2)
        # i, j = self._ids2pixel(width[0], height[0], camera_indices)
        # ij = torch.cat([i, j], dim=-1)

        ###############################################
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )

            # loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * distortion_loss(outputs["weights_list"],
                                                                                              outputs["ray_samples_list"])


            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )

        #########depth&semantic loss
        # semantic loss
        if self.use_semantic:
            semantics_gt = batch["semantics"][..., 0]  # torch.Size([4096, 3])

            semantics_label = [
                self.color2label.get(tuple(pixel.tolist()), self.semantic_num)
                for pixel in semantics_gt
            ]  # list 4096
            semantics_label = torch.tensor(semantics_label, device=semantics_gt.device)  # torch.Size([4096])
            loss_dict["semantics_loss"] = self.config.semantic_loss_weight * self.cross_entropy_loss(
                outputs["semantics"], semantics_label
            )
        if self.training and self.config.use_depth:
            assert "depth_image" in batch
            depth_gt = batch["depth_image"].to(self.device).float()

            if not self.config.is_euclidean_depth:
                depth_gt = depth_gt * outputs["directions_norm"]
            predicted_depth = outputs["depth"].float()

            mask = torch.ones_like(depth_gt).reshape(1, 32, -1).bool()
            mono_depth_loss = self.mono_depth_loss(predicted_depth.reshape(1, 32, -1), depth_gt.reshape(1, 32, -1),
                                                   mask)
            loss_dict["depth_loss"] = self.config.mono_depth_loss_mult * mono_depth_loss



        # if self.use_flow:
        #     flow_image = batch["flow_image"].to(self.device)
        #     cam2world = self.dataparser_outputs.cameras.camera_to_worlds.to(self.device)  # (n,3,4)
        #     fx = self.dataparser_outputs.cameras.fx.to(self.device)
        #     width = self.dataparser_outputs.cameras.width.to(self.device)
        #     height = self.dataparser_outputs.cameras.height.to(self.device)
        #     w_h = torch.cat([width[0], height[0]])
        #     ray_indices = batch["indices"]  # pixel batch indices
        #     camera_indices = ray_indices[:, 0]
        #     fwd_camera_indices = torch.clamp(camera_indices + 1, 0, camera_indices.max())
        #     bwd_camera_indices = torch.clamp(camera_indices - 1, 0, camera_indices.max())
        #     y = ray_indices[:, 1]  # row indices
        #     x = ray_indices[:, 2]
        #
        #     fwd_mask = batch["fwd_mask"].to(self.device)
        #     fwd_mask[camera_indices == len(cam2world) - 1] = 0
        #
        #     true_indices = [camera_indices[..., i] for i in range(camera_indices.shape[-1])]
        #     c2w = cam2world[true_indices]
        #     fwd_c2w = cam2world[fwd_camera_indices]
        #     bwd_c2w = cam2world[bwd_camera_indices]
        #     # starting_frame_id = dataparser.first_frame
        #     pts = outputs["origins"] + outputs["directions"] * outputs["depth"]  # [num_rays, 3]
        #     coords: torch.Tensor = self.dataparser_outputs.cameras.get_image_coords()  # index=tuple(index)# (h, w, 2)
        #     coords = coords[y, x].to(self.device)
        #     #############
        #     fwd_cam2cams, bwd_cam2cams = get_fwd_bwd_cam2cams(c2w, fwd_c2w, bwd_c2w)  # - starting_frame_id)  # c2w w2c
        #     pred_fwd_flow = get_pred_flow(pts, coords, fwd_cam2cams, fx[0], w_h)
        #     flow_loss_arr = torch.sum(torch.abs(pred_fwd_flow - flow_image), dim=-1) * fwd_mask
        #     flow_loss_arr[flow_loss_arr > torch.quantile(flow_loss_arr, 0.9)] = 0
        #     # flow_loss = (flow_loss_arr).mean() * self.config.flow_loss_mul * reg_loss_weight / ((W + H) / 2)
        #     ##############
        #     loss_dict["flow_loss"] = self.config.flow_loss_mult * torch.mean(flow_loss_arr) / (width[0] + height[0]) / 2
        # #########
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)

        rgb = outputs["rgb"]
        rgb = torch.clamp(rgb, min=0, max=1)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        # print("outputs_depth:{}---{}".format(outputs["depth"].min(),outputs["depth"].max()))
        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]
        
        if self.use_mask:
            motion_mask_bin = batch["mask"].to(self.device)
            motion_mask = torch.cat([motion_mask_bin] * 3, dim=2)
            motion_mask = torch.moveaxis(motion_mask, -1, 0)[None, ...]
        # psnr = self.psnr(torch.mul(image, motion_mask.int()), torch.mul(rgb, motion_mask.int()))
        # ssim = self.ssim(torch.mul(image, motion_mask.int()), torch.mul(rgb, motion_mask.int()))
        # lpips = self.lpips(torch.mul(image, motion_mask.int()), torch.mul(rgb, motion_mask.int()))

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)
        print("PSNR: {}".format(metrics_dict["psnr"]))
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        if self.use_depth:
            depth_gt = batch["depth_image"].to(self.device)
            print("输入的depth：{0}---{1}".format(depth_gt.min(), depth_gt.max()))
            if not self.config.is_euclidean_depth:
                depth_gt = depth_gt * outputs["directions_norm"]
            # depth_gt.max() = depth_gt.max()
            depth_gt_colormap = colormaps.apply_depth_colormap(depth_gt)

            scale, shift = normalized_depth_scale_and_shift(outputs["depth"], depth_gt[None, ...], depth_gt[None, ...] > 0.0)
            print("outdepth：{0}---{1}".format(outputs["depth"].min(), outputs["depth"].max()))
            print("scale:{0}, shift{1}".format(scale,shift))

            pre_depth = outputs["depth"] * scale + shift
            print("predicted_depth：{0}---{1}".format(pre_depth.min(), pre_depth.max()))
            # predicted_depth[predicted_depth > depth_gt.max()] = depth_gt.max()
            predicted_depth_colormap = colormaps.apply_depth_colormap(outputs["depth"])

            combined_depth = torch.cat([depth_gt_colormap, predicted_depth_colormap], dim=1)
            images_dict["depth"] = combined_depth
            depth_mask = depth_gt > 0
            metrics_dict["depth_mse"] = torch.nn.functional.mse_loss(
                pre_depth[depth_mask], depth_gt[depth_mask]
            )

        return metrics_dict, images_dict





