import numpy as np
from scipy.ndimage import label, center_of_mass
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
import torch
import torch.distributed as dist
import argparse
from tqdm import tqdm

def extract_clusters(mask):
    """Extract connected clusters from a binary mask."""
    struct = np.ones((3,3,3), dtype=bool)  # 26-connectivity
    labeled, num = label(mask, structure=struct)
    #print(labeled.shape)
    clusters = []
    for i in range(1, num + 1):
        component = (labeled == i)
        #print(component.shape)
        com = center_of_mass(component)
        clusters.append({
            "id": i,
            "mask": component,
            "center": tuple(map(int, com))
        })
    return clusters

def compare_clusters(gt, pred):
 
    gt_clusters = extract_clusters(gt)
    pred_clusters = extract_clusters(pred)

    TP, FP, FN = 0 , 0 , 0
    matched_pred_ids = set()

    # Check each GT cluster
    for g in gt_clusters:
        found_match = False
        for p in pred_clusters:
            if np.any(g["mask"] & p["mask"]):  # overlap exists
                TP+=1
                matched_pred_ids.add(p["id"])
                found_match = True
                break
        if not found_match:
            FN+=1

    # Remaining unmatched pred clusters → FP
    for p in pred_clusters:
        if p["id"] not in matched_pred_ids:
            FP+=1

    return TP, FP, FN

def metrics(TP,FP,FN, D):
    tp_count = TP
    fp_count = FP
    fn_count = FN

    TPR = tp_count / (tp_count + fn_count)

    precision = tp_count / (tp_count + fp_count)

    avg_fp = fp_count / D

    return TPR, precision, avg_fp

def parse_args():
    parser = argparse.ArgumentParser(description="region growth base on MedSAM2 model")
    parser.add_argument("--pred_dir", type=str, default="pred-result", help="test dir")
    parser.add_argument("--gt_dir", type=str, default="valdo-gt", help="test dir")
    return parser.parse_args()

def main(args):
    MASK_DIR = os.path.join(args.gt_dir)
    total_dice_score = 0
    count = 0
    tp_list = 0
    fp_list = 0
    fn_list = 0
    for filename in tqdm(os.listdir(MASK_DIR), desc="Processing files"):
        print(filename)
        count+=1
        gt_path = os.path.join(args.gt_dir, filename)
        pred_path = os.path.join(args.pred_dir, filename)

        img2 = nib.load(pred_path)
        pred_mask = img2.get_fdata()  

        gt_img = nib.load(gt_path)
        gt_mask = gt_img.get_fdata()

        TP, FP, FN = compare_clusters(gt_mask,pred_mask)

        tp_list+=TP
        fp_list+=FP
        fn_list+=FN

    tpr, precision, avg_fp = metrics(tp_list,fp_list,fn_list,count)
    print(f'Test precision: {precision}')
    print(f'Test recall: {tpr}')
    print(f'Test avg fp: {avg_fp}')

if __name__ == '__main__':
    args = parse_args()
    main(args)
        