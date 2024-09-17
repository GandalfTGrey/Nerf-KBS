import os
import shutil
import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from nerfstudio.process_data.process_data_utils import list_images
from rich.console import Console
from typing_extensions import Literal, OrderedDict

from nerfstudio.utils.rich_utils import status
from nerfstudio.utils.scripts import run_command

CONSOLE = Console(width=120)
POLYCAM_UPSCALING_TIMES = 2

def split_image_filenames(directory: Path, first_frame: int=245,
    last_frame: int=345,) -> Tuple[List[Path], int]:
    """Returns a list of image filenames in a directory.

    Args:
        dir: Path to the directory.
        max_num_images: The maximum number of images to return. -1 means no limit.
    Returns:
        A tuple of A list of image filenames, number of original image paths.
    """
    image_paths = list_images(directory)
    num_orig_images = len(image_paths)

    idx = [f"{idx}" for idx in range(first_frame, last_frame)]
    image_filenames = list(np.array(image_paths)[idx])

    return image_filenames