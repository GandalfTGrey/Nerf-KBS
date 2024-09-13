import os.path
from argparse import Namespace
from typing import Dict

import configargparse
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_text, qvec2rotmat, read_images_text
from tqdm import tqdm

from image_metadata import *
from metadata_utils import *

'''
--output_path
/home/chenghuan/suds/Zchenghuan/sequences/metadata1.json
--kitti_seq
0003
--kitti_root
/home/chenghuan/suds/Zchenghuan/sequences
'''

# From https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/devkits/convertOxtsPose/python/utils.py
def _lat_lon_to_mercator(lat: float, lon: float, scale: float) -> Tuple[float, float]:
    ''' converts lat/lon coordinates to mercator coordinates using mercator scale '''
    er = 6378137.  # average earth radius at the equator

    mx = scale * lon * np.pi * er / 180
    my = scale * er * np.log(np.tan((90 + lat) * np.pi / 360))

    return mx, my


# From https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/devkits/convertOxtsPose/python/utils.py
def _lat_to_scale(lat: float) -> float:
    ''' compute mercator scale from latitude '''
    scale = np.cos(lat * np.pi / 180.0)
    return scale

def colmap_to_pose(colmap_pose_path:str):
    'colmap 计算填补后image pose'

    cam_id_to_camera = read_cameras_text(os.path.join(colmap_pose_path, "cameras.txt"))
    im_id_to_image = read_images_text(os.path.join(colmap_pose_path, "images.txt"))
    frames = []
    CAMERA_ID = 1
    W = cam_id_to_camera[CAMERA_ID].width
    H = cam_id_to_camera[CAMERA_ID].height

    for im_id, im_data in im_id_to_image.items():
        rotation = qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c) #（4，4）
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL) （y,z）方向方向相反
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1
        # c2w.tolist()
    return c2w, W, H

'''
--output_path
/home/chenghuan/suds/data/kitti/metadata
--kitti_sequence
0006
--kitti_root
/home/chenghuan/suds/data/kitti
'''

def get_kitti_items(kitti_root: str,
                    kitti_seq: str,
                    frame_ranges: Optional[List[Tuple[int]]],
                    train_every: Optional[int],
                    test_every: Optional[int]) -> \
        Tuple[List[ImageMetadata], List[str], torch.Tensor, float, torch.Tensor]:

    '''获取内参'''
    calib: Dict[str, torch.Tensor] = {}
    # data_path = os.path.join(kitti_root)
    with open('{}/calib/{}.txt'.format(kitti_root, kitti_seq), 'r') as f:
        for line in f:
            tokens = line.strip().split()
            calib[tokens[0]] = torch.DoubleTensor([float(x) for x in tokens[1:]])
    P2 = calib['P2:'].view(3, 4)
    intrinsics = P2[:, :3]

    num_frames = 0

    val_frames = get_val_frames(num_frames, test_every, train_every)
    metadata_items: List[ImageMetadata] = []
    item_frame_ranges: List[Tuple[int]] = []
    static_masks = []
    min_bounds = None
    max_bounds = None
    use_masks = True

    colmap_pose_path = os.path.join(kitti_root, 'colmap', kitti_seq)
    # c2w, W, H = colmap_to_pose(colmap_pose_path)

    '''导入RGB depth flow de path'''
    '''如何加入 c2w depth flow'''
    cam_id_to_camera = read_cameras_text(os.path.join(colmap_pose_path, "cameras.txt"))
    im_id_to_image = read_images_text(os.path.join(colmap_pose_path, "images.txt"))

    frames = []
    # metadata_items: List[ImageMetadata] = []
    CAMERA_ID = 1
    W = cam_id_to_camera[CAMERA_ID].width
    H = cam_id_to_camera[CAMERA_ID].height

    min_frame = None
    max_frame = None
    for im_id, im_data in tqdm(im_id_to_image.items()):
        rotation = qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)  # （4，4）
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL) （y,z）方向方向相反
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        frame_range = get_frame_range(frame_ranges, im_id) if frame_ranges is not None else None
        min_frame = min(im_id, min_frame) if min_frame is not None else im_id
        max_frame = max(im_id, max_frame) if max_frame is not None else im_id

        image_index = len(metadata_items)
        is_val = image_index // 2 in val_frames

        if is_val:
            backward_neighbor = image_index - 2
            forward_neighbor = image_index + 2
        else:
            backward_neighbor = get_neighbor(image_index, val_frames, -2)
            forward_neighbor = get_neighbor(image_index, val_frames, 2)



        backward_flow_path = '{0}/motion/{1}/flow_bwd/{2:06d}.png'.format(kitti_root, kitti_seq, im_id-1 - (image_index - backward_neighbor) // 2)
        forward_flow_path = '{0}/motion/{1}/flow_fwd/{2:06d}.png'.format(kitti_root, kitti_seq, im_id-1)

        depth = '{0}/motion/{1}/depth_npy/{2:06d}_disp.npy'.format(kitti_root, kitti_seq, im_id-1)
        motion_mask = '{0}/motion/{1}/motion_mask/{2:06d}.png'.format(kitti_root, kitti_seq, im_id-1)
        fill_image_path = '{0}/fill_img/{1}/{2:06d}_fill.png'.format(kitti_root, kitti_seq, im_id-1)
        # fill_image = image_from_stream(fill_image_path)

        item = ImageMetadata(
            fill_image_path,
            torch.DoubleTensor(c2w[:3]),
            W,
            H,
            torch.DoubleTensor([intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]),
            image_index,
            im_id,  # frame,
            0,
            depth,
            motion_mask,
            None,
            None,
            backward_flow_path,
            forward_flow_path,
            backward_neighbor,
            forward_neighbor,
            None,
            1,
            None

        )

        metadata_items.append(item)
        # frames.append(item)
        item_frame_ranges.append(frame_range)

        min_bounds, max_bounds = get_bounds_from_depth(item, min_bounds, max_bounds)
    for item in metadata_items:
        normalize_timestamp(item, min_frame, max_frame)

    for item in metadata_items:
        if item.backward_neighbor_index < 0 \
                or item_frame_ranges[item.image_index] != item_frame_ranges[item.backward_neighbor_index]:
            item.backward_flow_path = None
            item.backward_neighbor_index = None

        if item.forward_neighbor_index >= len(metadata_items) \
                or item_frame_ranges[item.image_index] != item_frame_ranges[item.forward_neighbor_index]:
            item.forward_flow_path = None
            item.forward_neighbor_index = None

    origin, pose_scale_factor, scene_bounds = scale_bounds(metadata_items, min_bounds, max_bounds)

    return metadata_items, static_masks, origin, pose_scale_factor, scene_bounds

def _get_opts() -> Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    # parser.add_argument('--config_file', is_config_file=True)

    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--kitti_root', type=str, required=True)
    parser.add_argument('--kitti_seq', type=str, required=True)
    parser.add_argument('--frame_ranges', type=int, nargs='+', default=[0, 100])
    parser.add_argument('--train_every', type=int, default=1)
    parser.add_argument('--test_every', type=int, default=None)

    return parser.parse_args()


def main(hparams: Namespace) -> None:
    assert hparams.train_every is not None or hparams.test_every is not None, \
        'Exactly one of train_every or test_every must be specified'

    assert hparams.train_every is None or hparams.test_every is None, \
        'Only one of train_every or test_every must be specified'

    if hparams.frame_ranges is not None:
        frame_ranges = []
        for i in range(0, len(hparams.frame_ranges), 2):
            frame_ranges.append([hparams.frame_ranges[i], hparams.frame_ranges[i + 1]])
    else:
        frame_ranges = None

    metadata_items, static_masks, origin, pose_scale_factor, scene_bounds = get_kitti_items(hparams.kitti_root,
                                                                                            hparams.kitti_seq,
                                                                                            frame_ranges,
                                                                                            hparams.train_every,
                                                                                            hparams.test_every)

    write_metadata(hparams.output_path, metadata_items, static_masks, origin, pose_scale_factor, scene_bounds)


if __name__ == '__main__':
    main(_get_opts())
