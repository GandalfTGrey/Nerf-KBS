import argparse
import glob
import os

import PIL
import numpy as np
import skimage
import torchvision
from PIL import Image
import torch
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import zoom
import numpy as np
import pyquaternion
import torch
import cv2
from nuscenes.nuscenes import NuScenes as NuScenesDatabase
from tqdm import tqdm
import PIL.Image as pil
from torchvision import transforms

from process_data.monodepth2 import networks
import sys
from process_data.RAFT.core.raft import RAFT
from process_data.monodepth2.layers import disp_to_depth

epsylon = 1e-5

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--kitti_path', type=str,
                       )
    parser.add_argument('--kitti_flow_path', type=str,
                        default='/home/chenghuan/cc-master/kitti/kitti2015/data_scene_flow')
    parser.add_argument('--model_name', type=str, default="mono_640x192",
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--post_process",
                             help="if set will perform the flipping post processing "
                                  "from the original monodepth paper",
                             action="store_true")
    parser.add_argument("--min_depth",
                             type=float,
                             help="minimum depth",
                             default=0.1)
    parser.add_argument("--max_depth",
                             type=float,
                             help="maximum depth",
                             default=100.0)
    parser.add_argument('--optical_flow_model', default='RAFT/models/raft-kitti.pth',
                        help="restore checkpoint")

    parser.add_argument('--output_path',
                        help="dataset for output")
    parser.add_argument('--small',
                        action='store_true',
                        help='use small model')
    parser.add_argument('--mixed_precision',
                        action='store_true',
                        help='use mixed precision')
    parser.add_argument('--alternate_corr',
                        action='store_true',
                        help='use efficent correlation implementation')
    parser.add_argument("--camera",
                        choices=['zed', 'self_collected_I', 'kitti_odom', 'kitti_raw','tum', 'kitti_flow_mask', 'vkitti'],
                        help="camera type",
                        default='self_collected_I')
    parser.add_argument("--kitti_odom_seq",
                        help="kitti sequence",
                        default='10')
    parser.add_argument("--kitti_seq",
                        help="kitti sequence",
                        default='0006')
    parser.add_argument("--save_ins_image",
                        help="Whether to save instance mask",
                        default='False')


    return parser.parse_args()


def load_image(imfile,device):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def check_files(folder_path):

    if not os.listdir(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' is empty.")

def read_raw_calib_file(filepath):
    # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
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

def np_cosine_distance(a, b):
    """Measure cosine distance between a and b
        :param a: tensor
        :param b: tensor
        :return cosine similarity
    """
    normalize_a = a / (np.linalg.norm(a, axis=0) + epsylon)
    normalize_b = b / (np.linalg.norm(b, axis=0) + epsylon)
    cos_similarity = np.sum(
        np.multiply(normalize_a, normalize_b), axis=0
    )
    return (1.0 - cos_similarity)/2.0

def semantic_flow_combine(instance_m, motion_m, m_th): # 0.8
    final_mask = np.zeros((motion_m.shape)).astype('uint8')
    instance_n = np.unique(instance_m)
    global ins_mask
    for n in instance_n:
        if n > 0:
            ins_mask = instance_m.copy()
            ins_mask[ins_mask != n] = 0
            ins_mask[ins_mask == n] = 1
            if motion_m.shape[0] != ins_mask.shape[0]:
                g_shape = motion_m.shape
                r_shape = ins_mask.shape
                ins_mask = zoom(ins_mask, (g_shape[0] / r_shape[0], g_shape[1] / r_shape[1]), order=0)#矩阵缩放调整mask大小

    mov_ratio = np.sum(motion_m * ins_mask.astype('float32')) / np.sum(ins_mask.astype('float32'))
    if mov_ratio > m_th:
        final_mask += ins_mask.astype('uint8')
            # elif 0.5 < mov_ratio <= args.m_th:
    elif mov_ratio <= m_th: # 0.6
        final_mask += ins_mask.astype('uint8') * motion_m.astype('uint8')

    return 1-final_mask


def semantic_flow_combine2(instance_m, motion_m, m_th): # 0.8
    #motion_m: 0 for static, 1 for moving
    final_mask = np.zeros((motion_m.shape)).astype('uint8')
    ins_mask = instance_m.copy()
    ins_mask[ins_mask < 0.5] = 0
    ins_mask[ins_mask > 0.5] = 1
    ins_mask = 1- ins_mask
    # ins_mask: 0 for background(static), 1 for object(tend to moving)
    
    if motion_m.shape[0] != ins_mask.shape[0]:
        g_shape = motion_m.shape
        r_shape = ins_mask.shape
        ins_mask = zoom(ins_mask, (g_shape[0] / r_shape[0], g_shape[1] / r_shape[1]), order=0)#矩阵缩放调整mask大小

    mov_ratio = np.sum(motion_m * ins_mask.astype('float32')) / np.sum(ins_mask.astype('float32'))
    if mov_ratio > m_th:
        final_mask += ins_mask.astype('uint8')
            # elif 0.5 < mov_ratio <= args.m_th:
    elif mov_ratio <= m_th: # 0.6
        final_mask = ins_mask * motion_m

    return final_mask, ins_mask




def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def rotation_translation_to_pose(r_quat, t_vec):
    """Convert quaternion rotation and translation vectors to 4x4 matrix"""

    pose = np.eye(4)

    # NB: Nuscenes recommends pyquaternion, which uses scalar-first format (w x y z)
    # https://github.com/nutonomy/nuscenes-devkit/issues/545#issuecomment-766509242
    # https://github.com/KieranWynn/pyquaternion/blob/99025c17bab1c55265d61add13375433b35251af/pyquaternion/quaternion.py#L299
    # https://fzheng.me/2017/11/12/quaternion_conventions_en/
    pose[:3, :3] = pyquaternion.Quaternion(r_quat).rotation_matrix

    pose[:3, 3] = t_vec
    return pose

def colorize(image, cmap="gray"):
    h, w, c = image.shape
    print(h, w, c)
    if c == 1:  # depth
        image = image.squeeze()
        image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
        cmap = plt.get_cmap(cmap)
        image_colorized = cmap(image_normalized)[:, :, :3]
        return np.uint8(image_colorized * 255)
    else:
        return np.uint8(image * 255)

def annotation_to_panoptical(annotation_path, output_path,save_image=True):
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [70, 130, 180]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [250, 170, 30]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]#Car
    colormap[14] = [0, 0, 70]#Truck
    colormap[15] = [0, 60, 100]#bus
    colormap[16] = [0, 80, 100]#person
    colormap[17] = [0, 0, 230]#motorcycle
    colormap[18] = [119, 11, 32]#bicycle
    colormap[255] = [0, 0, 0]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dynamic_category = [13, 14, 15, 16, 18]
    for filename in tqdm(os.listdir(annotation_path)):
        file_path = os.path.join(annotation_path, filename)
        img = cv2.imread(file_path)#.to(device)
        modified_img = img.copy()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pixel = img[i, j]
                r, g, b = pixel[0], pixel[1], pixel[2]
                last_value = b
                if b in dynamic_category:
                    modified_img[i, j] = colormap[255]
                else:
                    modified_img[i, j] = [1, 1, 1]

        if save_image:
            modified_file_path = os.path.join(output_path, filename)
            modified_img=cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(modified_file_path, (modified_img * 255))



def load_depth_model(args, device):

    # MIN_DEPTH = 1e-3
    # MAX_DEPTH = 80.0

    # download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("monodepth2/models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    # depth network
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)  # 18
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    return feed_width,feed_height,encoder,depth_decoder


def load_flow_model(args,device):
    flow_model = torch.nn.DataParallel(RAFT(args))
    flow_model.load_state_dict(torch.load(args.model))

    flow_model = flow_model.module
    flow_model.to(device)
    flow_model.eval()

    return flow_model

def load_pose_model(args):
    model_path = os.path.join("monodepth2/models", args.model_name)
    pose_encoder_path = os.path.join(model_path, "pose_encoder.pth")
    pose_decoder_path = os.path.join(model_path, "pose.pth")
    pose_encoder = networks.ResnetEncoder(18, False, 2)  ##18
    pose_encoder.load_state_dict(torch.load(pose_encoder_path))
    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))
    pose_encoder.cuda()
    pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()

    return pose_encoder,pose_decoder

def run_maskrcnn(model, img_path, intWidth, intHeight, threshold):


    o_image = PIL.Image.open(img_path)
    image = o_image.resize((intWidth, intHeight), PIL.Image.LANCZOS)

    image_tensor = torchvision.transforms.functional.to_tensor(image).cuda()

    tenHumans = torch.FloatTensor(intHeight, intWidth).fill_(1.0).cuda()

    objPredictions = model([image_tensor])[0]

    for intMask in range(objPredictions['masks'].size(0)):
        if objPredictions['scores'][intMask].item() > threshold:
            if objPredictions['labels'][intMask].item() == 1: # person
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 4: # motorcycle
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 2: # bicycle
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 3: # car
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 8: # truck
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            if objPredictions['labels'][intMask].item() == 28: # umbrella
                tenHumans[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

    npyMask = skimage.morphology.erosion(tenHumans.cpu().numpy(),
                                         skimage.morphology.disk(10))
    npyMask = ((npyMask < 1e-3) * 255.0).clip(0.0, 255.0).astype(np.uint8)
    return npyMask

def get_instance_mask(basedir, W, H, threshold):
    # RUN SEMANTIC SEGMENTATION
    img_dir = os.path.join(basedir, 'images_2')
    img_path_list = sorted(glob.glob(os.path.join(img_dir, '*.jpeg'))) \
                    + sorted(glob.glob(os.path.join(img_dir, '*.png'))) \
                    + sorted(glob.glob(os.path.join(img_dir, '*.jpg'))) \

    semantic_mask_dir = os.path.join(basedir, 'semantic_mask')
    netMaskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda().eval()
    os.makedirs(semantic_mask_dir, exist_ok=True)

    for i in tqdm(range(0, len(img_path_list))):
        img_path = img_path_list[i]
        img_name = img_path.split('/')[-1]
        semantic_mask = run_maskrcnn(netMaskrcnn, img_path, W, H, threshold)

        cv2.imwrite(os.path.join(semantic_mask_dir,
                                 img_name),
                    255 - semantic_mask)
