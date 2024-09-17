import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path,  PosixPath
from typing import Tuple, Optional, Dict

import numpy as np
import tyro
from nerfstudio.configs.config_utils import CONSOLE
from nerfstudio.process_data import process_data_utils, colmap_utils
from nerfstudio.process_data.process_data_utils import list_images
from rich.console import Console

CONSOLE = Console(width=120)
DEFAULT_COLMAP_PATH = Path("0")
'''
--data
/home/chenghuan/suds/Zchenghuan/sequences/fill_img/0001
--output-dir
/home/chenghuan/suds/Zchenghuan/sequences/fill_img
'''
@dataclass
class ProcessImages:
    """Process images into a nerfstudio dataset.

    This script does the following:

    1. Scales images to a specified size.
    2. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.
    """

    data: Path
    """Path the data, either a video file or a directory of images."""
    output_dir: Path
    """Path to the output directory."""
    colmap_model_path: Path = DEFAULT_COLMAP_PATH

    skip_image_processing: bool = True
    """If --use-sfm-depth and this flag is True, also export debug images showing SfM overlaid upon input images."""

    crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """Portion of the image to crop. All values should be in [0,1]. (top, bottom, left, right)"""

    num_downscales: int = 2
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""

    verbose: bool = False
    """If True, print extra logging."""

    use_sfm_depth: bool = True
    """If True, export and use depth maps induced from SfM points."""

    include_depth_debug: bool = False
    """If --use-sfm-depth and this flag is True, also export debug images showing SfM overlaid upon input images."""

    # first_frame: int = 245
    # last_frame: int = 345

    def main(self, image_rename_map=None, require_cameras_exist=None, image_id_to_depth_path=None) -> None:

        colmap_dir = Path(os.path.join(self.output_dir, "colmap", "0"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        image_ = Path(os.path.join(self.output_dir, "images2"))
        # image_file = sorted(os.listdir(image_))
        # for i in range(self.first_frame, self.last_frame + 1):
        #     print(image_file[i-1])

        image_dir = self.output_dir / 'images'
        image_dir.mkdir(parents=True, exist_ok=True)
        summary_log = []
        # Copy and downscale images

        image_paths = list_images(image_ )
        num_frames = len(image_paths)
        # id = [idx for idx in range(self.first_frame, self.last_frame + 1)]
        # image_filenames = list(np.array(image_paths)[id])

        # if not self.skip_image_processing:
        #     # Copy images to output directory
        #     image_rename_map_paths = process_data_utils.copy_images(
        #         image_paths, image_dir=image_dir, crop_factor=self.crop_factor, verbose=self.verbose
        #     )
        #     image_rename_map = dict((a.name, b.name) for a, b in image_rename_map_paths.items())
        #     num_frames = len(image_rename_map)
        #     summary_log.append(f"Starting with {num_frames} images")
        #
        #     # Downscale images
        #     summary_log.append(
        #         process_data_utils.downscale_images(image_dir, self.num_downscales, verbose=self.verbose)
        #     )
        # else:
        #     num_frames = len(process_data_utils.list_images(self.data))
        #     if num_frames == 0:
        #         CONSOLE.log("[bold red]:skull: No usable images in the data folder.")
        #         sys.exit(1)
        #     summary_log.append(f"Starting with {num_frames} images")



        # Export depth maps
        if self.use_sfm_depth:
            depth_dir = self.output_dir / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
            image_id_to_depth_path = colmap_utils.create_sfm_depth(
                recon_dir=colmap_dir,
                output_dir=depth_dir,
                include_depth_debug=self.include_depth_debug,
                input_images_dir=image_dir,
                verbose=self.verbose,
            )
            summary_log.append(
                process_data_utils.downscale_images(
                    depth_dir, self.num_downscales, folder_name="depths", nearest_neighbor=True,
                    verbose=self.verbose
                )
            )
        else:
            image_id_to_depth_path = None

        colmap_model_path = colmap_dir
        # mask 路径 记得换
        # mask_path = '/home/chenghuan/suds/Zchenghuan/sequences/motion/0003'
        # mask_path_list = glob.glob(os.path.join(mask_path,'*.png'))
        # mask_path_list = [PosixPath(path) for path in mask_path_list]


        if (colmap_model_path / "cameras.bin").exists():
            with CONSOLE.status("[bold yellow]Saving results to transforms.json", spinner="balloon"):
                num_matched_frames = colmap_utils.colmap_to_json(
                    recon_dir=colmap_model_path,
                    output_dir=self.output_dir,
                    image_id_to_depth_path=image_id_to_depth_path,
                    image_rename_map=image_rename_map,
                    # first_frame=self.first_frame,
                    # last_frame=self.last_frame,
                )
                summary_log.append(f"Colmap matched {num_matched_frames} images")
            summary_log.append(colmap_utils.get_matching_summary(num_frames, num_matched_frames))
        elif require_cameras_exist:
            CONSOLE.log(f"[bold red]Could not find existing COLMAP results ({colmap_model_path / 'cameras.bin'}).")
            sys.exit(1)
        else:
            CONSOLE.log(
                "[bold yellow]Warning: could not find existing COLMAP results. Not generating transforms.json")

        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ProcessImages).main()

if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # type: ignore