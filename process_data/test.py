from process_data.motion_mask_util import annotation_to_panoptical

annotation_path = "/home/chenghuan/mars/data/kitti-MOT/kitti-step/panoptic_maps/val/0006"
output_path = "/home/chenghuan/mars/data/kitti-MOT/kitti-step/panoptic_maps/test"

INS = annotation_to_panoptical(annotation_path, output_path,save_image=True)