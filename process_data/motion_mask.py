# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import importlib
import os
import sys

from matplotlib import pyplot as plt

from process_data.inverse_warp import pose2flow, flow_to_image
from process_data.motion_mask_util import parse_args, load_image, batch_post_process_disparity, \
    np_cosine_distance, read_raw_calib_file, semantic_flow_combine, get_instance_mask, semantic_flow_combine2

sys.path.append('core')

import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import cv2
import math
from torch.autograd import Variable
from scipy.ndimage.interpolation import zoom

import torch
from torch.nn import functional as F
from torchvision import transforms, datasets
from PIL import Image

from process_data.monodepth2 import networks
from process_data.monodepth2.layers import disp_to_depth
from process_data.monodepth2.utils import download_model_if_doesnt_exist
from process_data.monodepth2.layers import transformation_from_parameters



from process_data.RAFT.core.raft import RAFT
from process_data.RAFT.core.utils import flow_viz
from process_data.RAFT.core.utils.utils import InputPadder

# import pykitti

def cal_opt_cam_flow_res(optical_flow:torch.Tensor,
                            cam_flow:torch.Tensor)->torch.Tensor:
    # higher score of moving_probability means more likely to be non-rigid
    optical_flow_np = (optical_flow[0] / optical_flow.max()).numpy()
    rigid_flow_np = (cam_flow.numpy()[0] / cam_flow.max()).numpy()
    normalized_optical_flow = np.linalg.norm(optical_flow_np, axis=0, keepdims=True)
    normalized_rigid_flow = np.linalg.norm(rigid_flow_np, axis=0, keepdims=True)

    distance_direct = normalized_optical_flow - normalized_rigid_flow
    distance_direct = (distance_direct- distance_direct.min()) / (distance_direct.max() - distance_direct.min()) # ranging from 0 to 1
    # distance_direct -= 0.5
    # distance_direct = 1 / (1 + np.exp(-distance_direct))
    return distance_direct
            
def render_mask_on_image(image, mask, color=(255, 255, 255), mask_weight=0.7):
    """
    将掩码渲染到图像上
    :param image: 原始图像
    :param mask: 掩码，0为背景，1为前景, of shape (H, W)
    :param color: 掩码颜色
    :return: 渲染后的图像
    """
    mask = np.uint8(mask)
    mask_color = np.zeros_like(image)
    # mask_color[mask_3d==1] = color
    # mask_3d = np.stack([mask, mask, mask], axis=0)
    
    mask_color[0, mask == 1] = color[0]  # 赋值给红色通道
    mask_color[1, mask == 1] = color[1]  # 赋值给绿色通道
    mask_color[2, mask == 1] = color[2]  # 赋值给蓝色通道
    result = cv2.addWeighted(image, 1-mask_weight, mask_color, mask_weight, 0)
    return result

"""
--image_path
/home/liu/data16t/Projects/NeRF_GS/Zchenghuan/sequences/self_collected_I/images
--output_path
/home/liu/data16t/Projects/NeRF_GS/Zchenghuan/sequences/self_collected_I/fill_output
--camera
I
--post_process
--model_name
mono+stereo_1024x320
--ext
jpeg
"""

def test_simple(args):
    """Function to predict for a single image or folder of images
    params:
        - args.image_path : path to the image or folder of images
        - args.output_path : path to save resulting depth maps
        - args.camera : camera model for which the network was trained, 
            self_collected_I: self collected camera index I (facing front)
            vkitti: for vKITTI dataset
            kitti_odom: for KITTI MOT dataset? TODO: change name or fix this
            kitti_flow_mask: 
        - args.model_name : name of the model to use
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80 #80.0
    if args.camera == 'self_collected_I':
        # I
        intrinsics = np.array([[1569.6758515689437 /2, 0.0, 960.0/2],
                               [0.0, 1569.6758515689437/2, 540.0/2],
                               [0.0, 0.0, 1.0]], dtype=np.float32)
        
        intrinsics_inv = np.linalg.inv(intrinsics)
        

    elif args.camera == 'vkitti':
        intrinsics = np.array([[725.0087, 0.0, 620.5],
                               [0.0, 725.0087, 187],
                               [0.0, 0.0, 1.0]])
        
    elif args.camera == 'kitti_odom':
        # kitti_odom
        # args.kitti_path = '/home/wangshuo/Datasets/KITTI/KITTI_full/odometry/data_odometry_full'
        # kitti_mask_path = '/home/wangshuo/CLionProjects/Semantic-Flow-Guided-SLAM/Mask/semantic/kitti'
        # odom = odometry(args.kitti_path, args.kitti_odom_seq)
        # calib = odom.calib
        # intrinsics = calib.K_cam2
        # T_cam02_velo_np = calib.T_cam2_velo  # gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)
        # T_cam03_velo_np = calib.T_cam3_velo
        # K_cam2 = calib.K_cam2  # 3x3
        # K_cam3 = calib.K_cam3
        # T_cam02_velo = torch.from_numpy(T_cam02_velo_np)
        # T_cam03_velo = torch.from_numpy(T_cam03_velo_np)
        calib_path = "/home/liu/data16t/Projects/NeRF_GS/Zchenghuan/MOT_sequences/kitti-MOT/training/calib/0006.txt"
        calib = {}
        with open(calib_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                calib[tokens[0]] = torch.DoubleTensor([float(x) for x in tokens[1:]])
        P2 = calib['P2:'].view(3, 4)
        intrinsics = P2[:, :3].to(torch.float32)
        intrinsics_inv = torch.inverse(P2[:, :3]).to(torch.float32)

    elif args.camera == 'kitti_flow_mask':
        calib_path = '/home/chenghuan/cc-master/kitti/kitti2015/data_scene_flow_calib/training/calib_cam_to_cam'
        calib_list = os.listdir(calib_path)
        calib_list.sort()
        intrinsics_list = []
        for calib in calib_list:
            data = read_raw_calib_file(os.path.join(calib_path, calib))
            intrinsics = np.reshape(data['P_rect_02'], (3, 4))[:, :3].astype(np.float32)
            intrinsics_list.append(intrinsics)

    # if not args.camera == 'kitti_flow_mask':
    #     intrinsics_inv = np.linalg.inv(intrinsics)
    #     intrinsics = torch.from_numpy(intrinsics.astype(np.float32))
    #     intrinsics_inv = torch.from_numpy(intrinsics_inv.astype(np.float32))

    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see READ ME.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join( os.path.dirname(__file__), "monodepth2/models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    # depth network
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
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

    # pose network
    pose_encoder_path = os.path.join(model_path, "pose_encoder.pth")
    pose_decoder_path = os.path.join(model_path, "pose.pth")
    pose_encoder = networks.ResnetEncoder(18, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path))
    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))
    pose_encoder.to(device)
    pose_encoder.eval()
    pose_decoder.to(device)
    pose_decoder.eval()
    pred_poses = []
    
    # flow network
    flow_model = torch.nn.DataParallel(RAFT(args))
    raft_model_path = os.path.join(os.path.dirname(__file__), args.optical_flow_model)
    flow_model.load_state_dict(torch.load(raft_model_path))

    flow_model = flow_model.module
    flow_model.to(DEVICE)
    flow_model.eval()



    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
        
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        if args.camera == 'kitti_odom':
            # paths = glob.glob(os.path.join(args.kitti_path, 'sequences', args.kitti_odom_seq,
            #                                'image_2/*.{}'.format(args.ext)))
            paths = "/home/chenghuan/mars/data/kitti-MOT/training/image_02/0006"
            paths = glob.glob(os.path.join(paths, '*.{}'.format(args.ext)))
            paths.sort()
            # path_ins = glob.glob(os.path.join(kitti_mask_path, args.kitti_odom_seq,
            #                                'instance/*.{}'.format(args.ext)))
            path_ins = "/home/chenghuan/mars/data/kitti-MOT/panoptic_maps/ins_mask/0006"
            path_ins = glob.glob(os.path.join(path_ins, '*.{}'.format(args.ext)))
            path_ins.sort()
            output_directory = args.output_path
            
        elif args.camera == 'tum':
            paths = glob.glob(os.path.join(args.tum_path, args.tum_seq,
                                           'rgb/*.{}'.format(args.ext)))
            paths.sort()
            output_directory = os.path.join(args.tum_path, args.tum_seq, 'output')


        elif args.camera == 'kitti_flow_mask':
            paths = glob.glob(os.path.join(args.kitti_flow_path, 'training',
                                           'image_2/*.{}'.format(args.ext)))
            paths.sort()
            output_directory = os.path.join(args.kitti_flow_path, 'output')
            paths_f = paths[1::2]
            paths_b = paths[::2]
            
        elif args.camera == 'self_collected_I':
            paths = glob.glob(os.path.join(args.image_path, 'images_2','*.{}'.format(args.ext)))
            paths.sort()
            path_ins = glob.glob(os.path.join(args.image_path, 'semantic_mask', '*.{}'.format('jpeg')))
            path_ins.sort()
            output_directory = args.output_path
            
        else:
            raise NotImplementedError("Please specify a valid dataset")

        paths_f = paths[1:]
        paths_b = paths[:-1]



    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    pre_fix_thresh = str(int(th_r*10))
    if not os.path.exists(os.path.join(output_directory, pre_fix_thresh +'moving_prob_from_obj_flow')):
        os.makedirs(os.path.join(output_directory, pre_fix_thresh +'moving_prob_from_obj_flow'))
    if not os.path.exists(os.path.join(output_directory, pre_fix_thresh + '_moving_prob_hard_mask_from_obj_flow')):
        os.makedirs(os.path.join(output_directory,  pre_fix_thresh + 'moving_prob_hard_mask_from_obj_flow'))
    if not os.path.exists(os.path.join(output_directory,  pre_fix_thresh + 'combined_motion_mask_SenmanticAndObjFlow')):
        os.makedirs(os.path.join(output_directory,  pre_fix_thresh + 'combined_motion_mask_SenmanticAndObjFlow'))
    if not os.path.exists(os.path.join(output_directory, pre_fix_thresh +  'mask_rendered')):
        os.makedirs(os.path.join(output_directory,  pre_fix_thresh + 'mask_rendered'))
    
    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            # depth prediction
            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            if args.camera == 'kitti_flow_mask':
                intrinsics = intrinsics_list[idx]
                intrinsics_inv = np.linalg.inv(intrinsics)
                intrinsics = torch.from_numpy(intrinsics.astype(np.float32))
                intrinsics_inv = torch.from_numpy(intrinsics_inv.astype(np.float32))

            # Load image and preprocess
            input_image_np = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image_np.size
            input_image_np = input_image_np.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image_np).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)

            if args.post_process:
                # Post-processed results require each image to have two forward passes
                input_image = torch.cat((input_image, torch.flip(input_image, [3])), 0)

            features = encoder(input_image)
            disp = depth_decoder(features)[("disp", 0)]


            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            pred_disp, _ = disp_to_depth(disp_resized, args.min_depth, args.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            if args.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_depth = 1 / pred_disp
            print("pred_depth:  [{0},{1}]".format(pred_depth.min(), pred_depth.max()))
            if args.camera == 'KITTI':
                pred_depth *= 1.2
            elif args.camera == 'kitti':
                pred_depth *= 5.37
            elif args.camera == 'tum':
                pred_depth *= 0.747
            # pred_depth *= 5.4
            # pred_depth *= 0.747
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            pred_depth_image = pred_depth.transpose(1,2,0).squeeze()
            pred_depth_image_normalized = (pred_depth_image - np.min(pred_depth_image)) / (np.max(pred_depth_image) - np.min(pred_depth_image))
            cmap = plt.get_cmap("gray")
            pred_depth_image_colorized = cmap(pred_depth_image_normalized)[:, :, :3]
            pred_depth_image = np.uint8(pred_depth_image_colorized * 255)

            # # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            # name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            # scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            # np.save(name_dest_npy, scaled_disp.cpu().numpy())
            #
            # # Saving colormapped depth image
            # disp_resized_np = pred_disp[0]
            # vmax = np.percentile(disp_resized_np, 95)
            # normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            # colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            # im = pil.fromarray(colormapped_im)
            image_name = image_path.split("/")[-1]
            if not os.path.exists(os.path.join(output_directory, 'depth_show')):
                os.makedirs(os.path.join(output_directory, 'depth_show'))
            name_dest = os.path.join(output_directory, 'depth_show', image_name)
            # pred_depth_image.save(name_dest)
            plt.imsave(name_dest, pred_depth_image)

            # if not os.path.exists(os.path.join(output_directory, 'depth')):
            #     os.makedirs(os.path.join(output_directory, 'depth'))
            # name_dest = os.path.join(output_directory, "depth/{}_depth.npy".format(output_name))
            # np.save(name_dest, pred_depth[0])
            # pred_depth.save(name_dest_im)

            # pose_prediction
            # Load image and preprocess
            input_image2 = pil.open(paths_f[idx]).convert('RGB')
            input_image2 = input_image2.resize((feed_width, feed_height), pil.LANCZOS)
            input_image2 = transforms.ToTensor()(input_image2).unsqueeze(0)

            # PREDICTION
            input_image2 = input_image2.to(device)
            all_color_aug = torch.cat((input_image[:1], input_image2), 1)

            features = [pose_encoder(all_color_aug)]
            axisangle, translation = pose_decoder(features)
            pose_vec = torch.cat((translation[:, 0, 0], axisangle[:, 0, 0]), 1)

            pose_T = transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy()
            pred_poses.append(pose_T)

            pred_depth = torch.from_numpy(pred_depth.astype(np.float32))
            flow_cam = pose2flow(pred_depth, pose_vec.detach().cpu(), torch.from_numpy(intrinsics).unsqueeze(0), torch.from_numpy(intrinsics_inv).unsqueeze(0))
            flow_cam_np = flow_cam[0].permute(1, 2, 0).cpu().detach().numpy()#cam flow  rigid flow

            if not os.path.exists(os.path.join(output_directory, 'cam_flow')):
                os.makedirs(os.path.join(output_directory, 'cam_flow'))
            rig_flow_im = flow_to_image(flow_cam_np)
            png_name = os.path.join(output_directory, 'cam_flow/{:010d}_flow.png'.format(idx))
            cv2.imwrite(png_name, rig_flow_im[:, :, ::-1])

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest))

            # flow prediction
            image1 = load_image(paths_b[idx], device)
            image2 = load_image(paths_f[idx], device)

            padder = InputPadder(image1.shape)#, mode='kitti')
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = flow_model(image1, image2, iters=20, test_mode=True)
            # viz(image1, flow_up)

            flow_up = padder.unpad(flow_up[0])
            flow_np = flow_up.permute(1, 2, 0).cpu().detach().numpy()

            flow_im = flow_to_image(flow_np)
            if not os.path.exists(os.path.join(output_directory, 'flow_show')):
                os.makedirs(os.path.join(output_directory, 'flow_show'))
            png_name = os.path.join(output_directory, 'flow_show/{:010d}_flow.png'.format(idx))
            cv2.imwrite(png_name, flow_im[:, :, ::-1])

            optical_flow = flow_up.unsqueeze(0).detach().cpu()
            cam_flow = flow_cam[0].unsqueeze(0)
            
            


            ###################################################################
            ## using cosine and ratio distance to calculate moving probability
            ################################################################
            ## higher score of moving_probability means more likely to be non-rigid
            # optical_flow_np = (optical_flow[0] / optical_flow.max()).numpy()
            # rigid_flow_np = (cam_flow.numpy()[0] / cam_flow.max()).numpy()
            # cosine_distance = np_cosine_distance(optical_flow_np, rigid_flow_np)
            # normalized_optical_flow = np.linalg.norm(optical_flow_np, axis=0, keepdims=True)
            # normalized_rigid_flow = np.linalg.norm(rigid_flow_np, axis=0, keepdims=True)
            # ratio = (
            #      np.minimum(normalized_optical_flow, normalized_rigid_flow)
            # ) / (epsylon + np.maximum(normalized_optical_flow, normalized_rigid_flow))
            # ratio_distance = 1.0 - ratio
            # moving_probability = np.maximum(cosine_distance, ratio_distance)
            # moving_probability = moving_probability[0]
        
            ###################################################################
            ## using direct flow residual to calculate moving probability
            ################################################################
            moving_probability = cal_opt_cam_flow_res(optical_flow, cam_flow)[0]

            
            png_name = os.path.join(output_directory, pre_fix_thresh +'moving_prob_from_obj_flow/{:010d}_mask.png'.format(idx))
            
            cv2.imwrite(png_name, moving_probability*255)


            ####################################################################
            ## hard moving_prob_mask_from_obj_flow, 0 means static, 1 means dynamic
            ####################################################################
            moving_prob_hard = moving_probability.copy()
            moving_prob_hard[moving_prob_hard > th_r] = 1
            moving_prob_hard[moving_prob_hard <= th_r] = 0

            png_name = os.path.join(output_directory,  pre_fix_thresh + 'moving_prob_hard_mask_from_obj_flow/{:010d}_mask.png'.format(idx))
            cv2.imwrite(png_name, moving_prob_hard * 255)


            ####################################################################
            ## combine with instance mask , 0 means static, 1 means dynamic
            ####################################################################
            ins_path = path_ins[idx]
            ins_im = cv2.imread(ins_path)
            pred_np_combined, ins_mask = semantic_flow_combine2(np.array(ins_im/255)[:, :, 0], moving_prob_hard, m_th)
            ## pred_np: 0 for static, 1 for dynamic
            ## ins_im: 0 for dynamic, 1 for static
            ## pred_np_combined 0 for static, 1 for dynamic
            

            png_name = os.path.join(output_directory,  pre_fix_thresh + 'combined_motion_mask_SenmanticAndObjFlow/{:010d}_mask.png'.format(idx))
            cv2.imwrite(png_name, pred_np_combined * 255)
            
            

            ####################################################################
            ## render the three masks on the image respectively,
            ## [ins_mask, moving_prob_hard_mask_from_obj_flow, combined_motion_mask_SenmanticAndObjFlow, original_image]
            ####################################################################

            png_name = os.path.join(output_directory,  pre_fix_thresh + 'mask_rendered/{:010d}_mask.png'.format(idx))


            original_image = np.transpose(
                np.array(input_image_np.resize(
                    (original_width, original_height ), pil.LANCZOS)),
                (2, 0, 1))
            images = []
            for mask in [ins_mask, moving_prob_hard, pred_np_combined]:
                rendered_image = render_mask_on_image(original_image.copy(), mask, color=(255, 255, 255), mask_weight=0.6)
                images.append(np.transpose(rendered_image, (1, 2, 0)))
            images.append(np.transpose(original_image, (1, 2, 0)))
            stitched_image = np.vstack(images)
            cv2.imwrite(png_name, np.transpose(stitched_image, (1, 2, 0)))
            cv2.imwrite(png_name,stitched_image)
            print(f"Saved stitched image to: {png_name}")


            
        pred_poses = np.concatenate(pred_poses)
    save_path = os.path.join(output_directory, "poses.npy")
    np.save(save_path, pred_poses)
    print("-> Predictions saved to", save_path)

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    epsylon = 1e-7
    DEVICE = 'cuda:0'
    th_r = 0.3  # 0.8
    m_th = 0.6

    args.save_ins_image = False  # debug TODO: delete
    if args.save_ins_image:
        get_instance_mask(args.image_path,
                          W=1242, H=375, threshold=0.5) # # RUN SEMANTIC SEGMENTATION
        print(" Completed instance mask")

    test_simple(args, )