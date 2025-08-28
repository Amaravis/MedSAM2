from glob import glob
from tqdm import tqdm
import os
from os.path import join, basename
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import OrderedDict
import pandas as pd
import numpy as np
import argparse
from scipy.ndimage import label, center_of_mass

from PIL import Image
import torch
import torch.multiprocessing as mp
from sam2.build_sam import build_sam2_video_predictor_npz
from skimage import measure, morphology
import cv2
import SimpleITK as sitk
import nibabel as nib
from collections import defaultdict

import argparse
import sys

def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def dice_multi_class(preds, targets):
    smooth = 1.0
    assert preds.shape == targets.shape
    labels = np.unique(targets)[1:]
    dices = []
    for label in labels:
        pred = preds == label
        target = targets == label
        intersection = (pred * target).sum()
        dices.append((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
    return np.mean(dices)

def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, edgecolor='blue'):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     

def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    
    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.
    
    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    h, w, d = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_pil = Image.fromarray(array[:, :, i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array
    
    return resized_array

def mask2D_to_bbox(gt2D, max_shift=20):
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = gt2D.shape
    bbox_shift = np.random.randint(0, max_shift + 1, 1)[0]
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W-1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H-1, y_max + bbox_shift)
    boxes = np.array([x_min, y_min, x_max, y_max])
    return boxes

def mask3D_to_bbox(gt3D, max_shift=20):
    z_indices, y_indices, x_indices = np.where(gt3D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    z_min, z_max = np.min(z_indices), np.max(z_indices)
    D, H, W = gt3D.shape
    bbox_shift = np.random.randint(0, max_shift + 1, 1)[0]
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W-1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H-1, y_max + bbox_shift)
    z_min = max(0, z_min)
    z_max = min(D-1, z_max)
    boxes3d = np.array([x_min, y_min, z_min, x_max, y_max, z_max])
    return boxes3d

def normalize_by_slices(volume):
    """
    volume: numpy.ndarray, shape: (H, W, D)
    return: normalized volume by every slices
    """
    norm_volume = np.zeros_like(volume, dtype=np.float32)
    for i in range(volume.shape[2]):
        slice_ = volume[:,:,i]
        slice_min = slice_.min()
        slice_ptp = np.ptp(slice_) + 1e-8
        norm_volume[:,:,i] = (slice_ - slice_min) / slice_ptp
    return norm_volume

# def apply_mask_to_volume(segs_3D, out_mask_logits, z):
#     mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
#     segs_3D[mask.nonzero()[0], mask.nonzero()[1], z] = 1

def apply_mask_to_volume(segs_3D, out_mask_logits, z, out_obj_ids):
    """
    Parameters:
        segs_3D (np.ndarray): (H, W, D)。
        out_mask_logits (torch.Tensor): [N, 1, H_mask, W_mask]
        z (int): 
        out_obj_ids: id list
    """
    for i, obj_id_val in enumerate(out_obj_ids):
        single_obj_mask_logits = out_mask_logits[i]
        predicted_mask_2d = (single_obj_mask_logits > 0.0).cpu().numpy()
        segs_3D[predicted_mask_2d.nonzero()[1], predicted_mask_2d.nonzero()[2], z] = 1


def extract_cmb_centers(mask, min_voxels=1, max_voxels=10):
    struct = np.array([[[1,1,1],
                        [1,1,1],
                        [1,1,1]],
                       
                       [[1,1,1],
                        [1,1,1],
                        [1,1,1]],
                       
                       [[1,1,1],
                        [1,1,1],
                        [1,1,1]]], dtype=bool)
    
    labeled, num = label(mask, structure=struct)
    centers = []

    for i in range(1, num + 1):
        component = (labeled == i)
        n_voxels = np.count_nonzero(component)
        #if n_voxels < min_voxels or n_voxels > max_voxels:
        #    continue
        com = center_of_mass(component)
        centers.append(tuple(map(int, com)))

    return centers

def parse_args():
    parser = argparse.ArgumentParser(description="region growth base on MedSAM2 model")
    parser.add_argument("--volume", type=str, default="swi_data/volume", help="volume saved dir")
    parser.add_argument("--mask", type=str, default="swi_data/mask", help="mask saved dir")
    parser.add_argument("--pred", type=str, default="swi_data/pred", help="pred saved dir")
    return parser.parse_args()

def main_labels(args):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_cfg = os.path.join("configs", "sam2.1_hiera_t512.yaml")
    checkpoint = os.path.join(BASE_DIR, "checkpoints", "MedSAM2_CTLesion.pt")
    predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)

    volume_dir_path = args.volume
    label_dir_path = args.mask
    pred_dir_path = args.pred

    for filename in os.listdir(volume_dir_path):
        print(filename)
        basename = filename.replace('.nii.gz', '')
        volume_path = os.path.join(volume_dir_path, filename)
        label_path = os.path.join(label_dir_path, basename + '.txt')
        volume = nib.load(volume_path)
        nii_image_data_pre = volume.get_fdata()
        segs_3D = np.zeros(nii_image_data_pre.shape, dtype=np.uint8)
        centers = defaultdict(list)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = parts = line.strip().split()
                    if len(parts) >= 3:
                        z = int(float(parts[1])*nii_image_data_pre.shape[2])
                        x = int(float(parts[2])*nii_image_data_pre.shape[1])
                        y = int(float(parts[3])*nii_image_data_pre.shape[0])
                        centers[z].append([x, y])
            centers = dict(centers)


        nii_norm = normalize_by_slices(nii_image_data_pre)
        img_3D_ori = nii_norm * 255.0

        assert np.max(img_3D_ori) < 256, f'input data should be in range [0, 255], but got {np.unique(img_3D_ori)}'

        img_h = img_3D_ori.shape[0]
        img_w = img_3D_ori.shape[1]
        img_d = img_3D_ori.shape[2]
        img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512)
        img_resized = img_resized / 255.0
        img_resized = torch.from_numpy(img_resized)
        img_mean=(0.485, 0.456, 0.406)
        img_std=(0.229, 0.224, 0.225)
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        img_resized -= img_mean
        img_resized /= img_std
        z_mids = []
        # point_coords = torch.tensor([[y, x]], dtype=torch.float)
        # point_labels = torch.tensor([1], dtype=torch.int) 
        inference_state = predictor.init_state(img_resized, img_h, img_w)
        # print(inference_state["num_frames"])
        obj_id = 1

        for z, pts in centers.items():
            for pt in pts:
                y, x = pt
                point_coord = torch.tensor([[y, x]], dtype=torch.float)
                point_label = torch.tensor([1], dtype=torch.int) 
                box_size_1 = 7
                y_min_1 = max(y - box_size_1, 0)
                y_max_1 = min(y + box_size_1, nii_image_data_pre.shape[0])
                x_min_1 = max(x - box_size_1, 0)
                x_max_1 = min(x + box_size_1, nii_image_data_pre.shape[1])
                bbox_1 = np.array([y_min_1, x_min_1, y_max_1, x_max_1], dtype=np.float32)
                box_size_2 = 7
                y_min_2 = max(y - box_size_2, 0)
                y_max_2 = min(y + box_size_2, nii_image_data_pre.shape[0])
                x_min_2 = max(x - box_size_2, 0)
                x_max_2 = min(x + box_size_2, nii_image_data_pre.shape[1])
                bbox_2 = np.array([y_min_2, x_min_2, y_max_2, x_max_2], dtype=np.float32)
                _, out_obj_ids_init, out_mask_logits = predictor.add_new_points_or_box(
                                                    inference_state=inference_state,
                                                    frame_idx=z,
                                                    obj_id=obj_id,
                                                    points=point_coord,
                                                    labels=point_label,
                                                    box=bbox_1,
                                                )
                apply_mask_to_volume(segs_3D,out_mask_logits,z=z, out_obj_ids=out_obj_ids_init)
                _, out_obj_ids_down, out_mask_logits = predictor.add_new_points_or_box(
                                                    inference_state=inference_state,
                                                    frame_idx=z-1,
                                                    obj_id=obj_id,
                                                    points=point_coord,
                                                    labels=point_label,
                                                    box=bbox_2,
                                                    # clear_old_points = False
                                                )
                apply_mask_to_volume(segs_3D,out_mask_logits,z=z-1, out_obj_ids=out_obj_ids_down)
                _, out_obj_ids_up, out_mask_logits = predictor.add_new_points_or_box(
                                                    inference_state=inference_state,
                                                    frame_idx=z+1,
                                                    obj_id=obj_id,
                                                    points=point_coord,
                                                    labels=point_label,
                                                    box=bbox_2,
                                                    # clear_old_points = False
                                                )
                apply_mask_to_volume(segs_3D,out_mask_logits,z=z+1, out_obj_ids=out_obj_ids_up)
                obj_id += 1
        # for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        #     apply_mask_to_volume(segs_3D,out_mask_logits,out_frame_idx, out_obj_ids)
        # for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
        #     apply_mask_to_volume(segs_3D,out_mask_logits,out_frame_idx, out_obj_ids)
        # pred_mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
        affine = np.eye(4)
        seg_nifti = nib.Nifti1Image(segs_3D.astype(np.uint8), affine)
        nib.save(seg_nifti, os.path.join(pred_dir_path, filename))

    print("FINISH")



        
        


def main(args):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_cfg = os.path.join("configs", "sam2.1_hiera_t512.yaml")
    checkpoint = os.path.join(BASE_DIR, "checkpoints", "trained_checkpoint.pt")
    #model_cfg = os.path.join(BASE_DIR, "sam2", "configs", "sam2.1_hiera_tiny_finetune512.yaml")
    #checkpoint = os.path.join(BASE_DIR, "checkpoints", "trained_checkpoint.pt")
    predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)

    volume_dir_path = args.volume
    mask_dir_path = args.mask
    pred_dir_path = args.pred
    for filename in tqdm(os.listdir(volume_dir_path), desc="Processing files"):
        volume_path = os.path.join(volume_dir_path, filename)
        mask_path = os.path.join(mask_dir_path, filename)
        # volume_path =  "real_sample/239_T2_MRI_SWI_BFC_50mm_HM_volume.nii.gz"
        # mask_path =  "real_sample/239_T2_MRI_SWI_BFC_50mm_HM_mask.nii.gz"

        volume = nib.load(volume_path)
        nii_image_data_pre = volume.get_fdata()
        segs_3D = np.zeros(nii_image_data_pre.shape, dtype=np.uint8)

        if os.path.exists(mask_path):
            mask = nib.load(mask_path)
            mask = mask.get_fdata()
        else:
            mask = np.zeros(nii_image_data_pre.shape, dtype=np.uint8)

        # Two ways to extract cmb centers from gt mask.
        # label_position = np.argwhere(np.isclose(mask, 1.0))
        label_positions = extract_cmb_centers(mask)
        print(len(label_positions))
        z_to_points = defaultdict(list)
        for x, y, z in label_positions:
            z_to_points[z].append([y, x])
        z_to_points = dict(z_to_points)

        # normalized whole image together or slice by slice.
        # nii_norm = (nii_image_data_pre - nii_image_data_pre.min())/(np.ptp(nii_image_data_pre) + 1e-8)
        nii_norm = normalize_by_slices(nii_image_data_pre)
        img_3D_ori = nii_norm*255.0

        assert np.max(img_3D_ori) < 256, f'input data should be in range [0, 255], but got {np.unique(img_3D_ori)}'

        img_h = img_3D_ori.shape[0]
        img_w = img_3D_ori.shape[1]
        img_d = img_3D_ori.shape[2]
        img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512)
        img_resized = img_resized / 255.0
        img_resized = torch.from_numpy(img_resized)
        img_mean=(0.485, 0.456, 0.406)
        img_std=(0.229, 0.224, 0.225)
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        img_resized -= img_mean
        img_resized /= img_std
        z_mids = []
        # point_coords = torch.tensor([[y, x]], dtype=torch.float)
        # point_labels = torch.tensor([1], dtype=torch.int) 
        inference_state = predictor.init_state(img_resized, img_h, img_w)
        # print(inference_state["num_frames"])
        obj_id = 1
        for z, pts in z_to_points.items():
            # point_coords = torch.tensor([pts], dtype=torch.float32)  # shape: [1, N, 2]
            # point_labels = torch.tensor([[1] * len(pts)], dtype=torch.int32)  # shape: [1, N]
            for pt in pts:
                y, x = pt
                point_coord = torch.tensor([[y, x]], dtype=torch.float)
                point_label = torch.tensor([1], dtype=torch.int) 
                box_size_1 = 3
                y_min_1 = max(y - box_size_1, 0)
                y_max_1 = min(y + box_size_1, 176)
                x_min_1 = max(x - box_size_1, 0)
                x_max_1 = min(x + box_size_1, 256)
                bbox_1 = np.array([y_min_1, x_min_1, y_max_1, x_max_1], dtype=np.float32)
                box_size_2 = 3
                y_min_2 = max(y - box_size_2, 0)
                y_max_2 = min(y + box_size_2, 176)
                x_min_2 = max(x - box_size_2, 0)
                x_max_2 = min(x + box_size_2, 256)
                bbox_2 = np.array([y_min_2, x_min_2, y_max_2, x_max_2], dtype=np.float32)
                _, out_obj_ids_init, out_mask_logits = predictor.add_new_points_or_box(
                                                    inference_state=inference_state,
                                                    frame_idx=z,
                                                    obj_id=obj_id,
                                                    points=point_coord,
                                                    labels=point_label,
                                                    box=bbox_1,
                                                )
                
                apply_mask_to_volume(segs_3D,out_mask_logits,z=z, out_obj_ids=out_obj_ids_init)
                _, out_obj_ids_down, out_mask_logits = predictor.add_new_points_or_box(
                                                    inference_state=inference_state,
                                                    frame_idx=z-1,
                                                    obj_id=obj_id,
                                                    points=point_coord,
                                                    labels=point_label,
                                                    box=bbox_2,
                                                    # clear_old_points = False
                                                )
                apply_mask_to_volume(segs_3D,out_mask_logits,z=z-1, out_obj_ids=out_obj_ids_down)
                if z+1 < nii_image_data_pre.shape[2]:
                    _, out_obj_ids_up, out_mask_logits = predictor.add_new_points_or_box(
                                                        inference_state=inference_state,
                                                        frame_idx=z+1,
                                                        obj_id=obj_id,
                                                        points=point_coord,
                                                        labels=point_label,
                                                        box=bbox_2,
                                                        # clear_old_points = False
                                                    )
                    apply_mask_to_volume(segs_3D,out_mask_logits,z=z+1, out_obj_ids=out_obj_ids_up)
                if z+2 < nii_image_data_pre.shape[2]:
                    _, out_obj_ids_up, out_mask_logits = predictor.add_new_points_or_box(
                                                        inference_state=inference_state,
                                                        frame_idx=z+2,
                                                        obj_id=obj_id,
                                                        points=point_coord,
                                                        labels=point_label,
                                                        box=bbox_2,
                                                        # clear_old_points = False
                                                    )
                    apply_mask_to_volume(segs_3D,out_mask_logits,z=z+2, out_obj_ids=out_obj_ids_up)
                _, out_obj_ids_up, out_mask_logits = predictor.add_new_points_or_box(
                                                    inference_state=inference_state,
                                                    frame_idx=z-2,
                                                    obj_id=obj_id,
                                                    points=point_coord,
                                                    labels=point_label,
                                                    box=bbox_2,
                                                    # clear_old_points = False
                                                )
                apply_mask_to_volume(segs_3D,out_mask_logits,z=z-2, out_obj_ids=out_obj_ids_up)
                obj_id += 1
        # for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        #     apply_mask_to_volume(segs_3D,out_mask_logits,out_frame_idx, out_obj_ids)
        # for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
        #     apply_mask_to_volume(segs_3D,out_mask_logits,out_frame_idx, out_obj_ids)
        # pred_mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
        affine = np.eye(4)
        seg_nifti = nib.Nifti1Image(segs_3D.astype(np.uint8), affine)
        nib.save(seg_nifti, os.path.join(pred_dir_path, filename))

    print("FINISH")

if __name__ == '__main__':
    args = parse_args()
    #main(args)
    main_labels(args)