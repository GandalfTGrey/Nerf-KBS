from __future__ import absolute_import, division, print_function

import argparse
import importlib
import os
import sys
sys.path.append("E2FGVI")
from E2FGVI.core.utils import to_tensors
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
'''
--image_path
/home/chenghuan/suds/Zchenghuan/sequences
--ckpt
/home/chenghuan/suds/Zchenghuan/process_data/models/E2FGVI-HQ-CVPR22.pth
'''

def load_image(path):
    frames = []
    lst = os.listdir(path)
    lst.sort()
    fr_lst = [path + '/' + name for name in lst if name.endswith('jpeg') or name.endswith('jpg') or name.endswith('png')]
    fr_lst.pop()
    for fr in fr_lst:
        image = cv2.imread(fr)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(image)
    return frames

def load_mask(mpath, size):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for mp in mnames:
        m = Image.open(os.path.join(mpath, mp))
        m = m.resize(size, Image.NEAREST)
        m = np.array(m.convert('L'))
        m = 1 - np.array(m > 0).astype(np.uint8)#黑色为感兴趣区域
        m = cv2.dilate(m,
                       cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10)),
                       iterations=4)
        masks.append(Image.fromarray(m * 255))
    return masks

# resize frames
def resize_frames(frames, size=None):
    if size is not None:
        frames = [f.resize(size) for f in frames]
    else:
        size = frames[0].size
    return frames, size

def inpaint():

    ###### LOAD MODEL
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    if args.model == "e2fgvi":
        size = (432, 240)
    elif args.set_size:
        size = (960, 540)
    else:
        size = None

    # net = 'E2FGVI.model.e2fgvi_hq'
    # net = importlib.import_module(net)
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data)
    print(f'Loading model from: {args.ckpt}')
    model.eval()

    ##### LOAD DATA
    image_path = args.image_path
    # frame_path = os.path.join(image_path,args.kitti_seq)
    frame_path = os.path.join(image_path)
    frames = load_image(frame_path)
    frames, size = resize_frames(frames, size)
    # size = (args.width, args.height)

    video_length = len(frames)
    imgs = to_tensors()(frames).unsqueeze(0) * 2 - 1
    h, w = imgs.shape[3], imgs.shape[4]
    frames = [np.array(f).astype(np.uint8) for f in frames]

    # mpath = os.path.join(image_path, 'fill_output', "motion_mask2")
    # mpath = "/home/chenghuan/cc-master/kitti/kitti2015/exp3/output/motion_mask3"
    mpath = "/home/chenghuan/mars/vkitti/Scene06/clone/frames/ins_mask/Camera_0"
    masks = load_mask(args.mpath, size)

    binary_masks = [
        np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks
    ]
    masks = to_tensors()(masks).unsqueeze(0)  # torch.Size([1, n, 1, h, w])
    imgs, masks = imgs.to(device), masks.to(device)
    comp_frames = [None] * video_length

    torch.cuda.empty_cache()
    print(f'-> Start inpaint...')
    for f in tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                             min(video_length, f + neighbor_stride + 1))
        ]
        ref_ids = []
        selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
        with torch.no_grad():
            masked_imgs = selected_imgs * (1 - selected_masks)
            mod_size_h = 60
            mod_size_w = 108
            h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
            w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [3])],
                3)[:, :, :, :h + h_pad, :]
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [4])],
                4)[:, :, :, :, :w + w_pad]
            pred_imgs, _ = model(masked_imgs, len(neighbor_ids))
            pred_imgs = pred_imgs[:, :, :h, :w]
            pred_imgs = (pred_imgs + 1) / 2
            pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_imgs[i]).astype(
                    np.uint8) * binary_masks[idx] + frames[idx] * (
                              1 - binary_masks[idx])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32) * 0.5 + img.astype(np.float32) * 0.5

                comp = cv2.cvtColor(comp_frames[idx], cv2.COLOR_BGR2RGB)  # (376, 1241, 3)
                if not os.path.exists(os.path.join(image_path, 'fill_img')):
                    os.makedirs((os.path.join(image_path, 'fill_img')))

                png_name = os.path.join(image_path, 'fill_img/rgb_{0:05d}.jpg'.format(idx))
                cv2.imwrite(png_name, comp)
    print('Finish inpaint! The result fill_image is saved in: {}'.format(os.path.join(image_path, 'fill_img')))

def parse_args():
    parser = argparse.ArgumentParser(description="inpainting")
    parser.add_argument("--image_path", type=str, required=True, default="/home/chenghuan/cc-master/kitti/kitti2015/exp3/image/left")
    parser.add_argument("--kitti_seq", help="kitti sequence")
    parser.add_argument("--mpath", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True,default="/home/chenghuan/suds/Zchenghuan/process_data/models/E2FGVI-HQ-CVPR22.pth")
    # parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=['e2fgvi', 'e2fgvi_hq'], default='e2fgvi_hq')
    parser.add_argument("--neighbor_stride", type=int, default=4)#4
    parser.add_argument("--step", type=int, default=5)#5

    # args for e2fgvi_hq (which can handle videos with arbitrary resolution)
    parser.add_argument("--set_size", action='store_true', default=False)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    ref_length = args.step  # ref_step
    # num_ref = args.num_ref
    neighbor_stride = args.neighbor_stride
    inpaint()
