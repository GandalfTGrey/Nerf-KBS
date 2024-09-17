import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Type, List
import pandas as pd

import numpy as np
import pyquaternion
import torch
from nerfstudio.cameras import camera_utils
from nuscenes.nuscenes import NuScenes as NuScenesDatabase
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs, Semantics,
)
from nerfstudio.data.scene_box import SceneBox


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def load_poses(data_dir, sequence):
    """Load ground truth poses (T_w_cam0) from file."""
    pose_file = os.path.join(data_dir, sequence + '.txt')

    # Read and parse the poses
    poses = []
    try:
        with open(pose_file, 'r') as f:
            lines = f.readlines()
            # if self.frames is not None:
            #     lines = [lines[i] for i in self.frames]

            for line in lines:
                T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                T_w_cam0 = T_w_cam0.reshape(3, 4)
                T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                poses.append(T_w_cam0)

    except FileNotFoundError:
        print('Ground truth poses are not avaialble for sequence ' +
              sequence + '.')

    return poses

@dataclass
class kittiDataParserConfig(DataParserConfig):
    """
    kitti_odom dataset config.

    """

    _target: Type = field(default_factory=lambda: kitti)
    """target class to instantiate"""

    data_dir: Path = Path("/home/chenghuan/suds/Zchenghuan/sequences/kitti")
    """Path to kitti dataset."""

    mask_dir: Optional[Path] = None
    """Path to masks of dynamic objects."""

    semantics_dir: Optional[Path] = Path("/home/chenghuan/suds/Zchenghuan/sequences/kitti/semantics")
    """whether or not to include loading of semantics data"""

    # depth_dir: Optional[Path] = None
    # """Path to mono-depth."""

    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""

    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""

    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""

    train_split_fraction: float = 0.9
    """The percent of images to use for training. The remaining images are for eval."""

    verbose: bool = False
    """Load dataset with verbose messaging"""

    scale_factor: float = 1.0
    """How much to scale the camera origins by."""

    depth_unit_scale_factor: float = 1e-3 #1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""

    mask_classes: List[str] = field(default_factory=lambda: [])
    """classes to mask out from training for all modalities"""



    use_depth: bool=True
    sequence: str = "00"

    first_frame:    int = 244
    last_frame: int = 294
    """frame = [first_frame, last_frame)"""


@dataclass
class kitti(DataParser):
    """kitti_odom DatasetParser"""

    config: kittiDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        # pylint: disable=too-many-statements
        transform1 = np.array(
            [
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        transform2 = np.array(
            [
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        # Load the calibration file
        image_filepath = os.path.join(self.config.data_dir, self.config.sequence)
        depth_filepath = os.path.join(self.config.data_dir, 'depth')
        calib_filepath = os.path.join(self.config.data_dir, 'calib.txt')
        filedata = read_calib_file(calib_filepath)
        T2 = np.eye(4)
        T2[0, 3] =np.reshape(filedata['P2'], (3, 4))[0, 3] / np.reshape(filedata['P2'], (3, 4))[0, 0]


        cam_intrinsics = np.reshape(filedata['P2'], (3, 4))[0:3, 0:3]
        w2c = load_poses(self.config.data_dir, self.config.sequence)

        idx = [idx for idx in range(self.config.first_frame, self.config.last_frame )]
        cam_intrinsics = np.tile(cam_intrinsics,(len(idx),1,1))
        poses=[]
        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        semantics_filenames = []
        mask_dir = self.config.mask_dir if self.config.mask_dir is not None else Path("")
        semantics_dir = self.config.semantics_dir if self.config.semantics_dir is not None else Path("")
        depth_dir = Path(self.config.data_dir / "depth" )
        for i in idx:

            # rotate to opencv frame
            pose = transform1 @ (w2c[i] @ T2)

            # convert from opencv camera to nerfstudio camera
            pose[0:3, 1:3] *= -1
            pose = pose[np.array([1, 0, 2, 3]), :]
            pose[2, :] *= -1

            # rotate to z-up in viewer
            # pose = transform2 @ pose
            poses.append(pose)

            image_filenames.append(os.path.join(image_filepath, f"{i:06}.png"))
            # depth_filenames.append(os.path.join(depth_filepath, f"{i:06}.npy"))
            mask_filenames.append(mask_dir / "masks")
            semantics_filenames.append(semantics_dir / f"{i:06}.png")
            depth_filenames.append(depth_dir / f"{i:06}.npy")


        poses = torch.from_numpy(np.stack(poses).astype(np.float32)) #torch.Size([len(idx), 4, 4])
        intrinsics = torch.from_numpy(np.array(cam_intrinsics).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )
        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # filter image_filenames and poses based on train/eval split percentage
        num_snapshots = len(idx)
        num_train_snapshots = math.ceil(num_snapshots * self.config.train_split_fraction)
        num_eval_snapshots = num_snapshots - num_train_snapshots
        i_all = np.arange(num_snapshots)

        i_train = np.linspace(
            0, num_snapshots - 1, num_train_snapshots, dtype=int
        )  # 0-xxx equally spaced training snapshots starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        assert len(i_eval) == num_eval_snapshots
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices]

        depth_filenames = [depth_filenames[i] for i in indices]
        intrinsics = intrinsics[indices]
        poses = poses[indices]

        if self.config.semantics_dir is not None:
            semantics = pd.read_csv(
                os.path.join(self.config.data_dir,"semantics_list.txt"),
                sep=",",
                index_col=False,
            )
            semantics = Semantics(
                filenames=[],
                classes=semantics["Category"].tolist(),
                colors=torch.tensor(semantics.iloc[:, 1:].values) / 255.0,
                mask_classes=self.config.mask_classes,
            )
            semantics.filenames = [semantics_filenames[i] for i in indices]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = 1.0
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )



        cameras = Cameras(
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            distortion_params=torch.Tensor([0, 0, 0, 0, 0, 0]),
            height=376,
            width=1241,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if self.config.mask_dir is not None else None,
            dataparser_transform=transform_matrix,
            dataparser_scale=scale_factor,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                "semantics": semantics if self.config.semantics_dir is not None else None,
            },

        )
        return dataparser_outputs

