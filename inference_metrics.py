import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os
import torch
import torch.distributed as dist
import argparse
from tqdm import tqdm


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

def compute_per_channel_dice(pred_mask, gt_mask, epsilon=1e-6, weight=None):
    pred_tensor = torch.from_numpy(pred_mask).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    gt_tensor = torch.from_numpy(gt_mask).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

    assert gt_tensor.size() == pred_tensor.size(), "'input' and 'target' must have the same shape"

    input_flat = flatten(pred_tensor)
    target_flat = flatten(gt_tensor)
    
    intersect = (input_flat * target_flat).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (input_flat * input_flat).sum(-1) + (target_flat * target_flat).sum(-1)
    
    return 2 * (intersect / denominator.clamp(min=epsilon))

def parse_args():
    parser = argparse.ArgumentParser(description="region growth base on MedSAM2 model")
    parser.add_argument("--pred_dir", type=str, default="pred-result", help="test dir")
    parser.add_argument("--gt_dir", type=str, default="valdo-gt", help="test dir")
    return parser.parse_args()

def main(args):
    MASK_DIR = os.path.join(args.gt_dir)
    total_dice_score = 0
    count = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    tp_count = 0
    fp_count = 0
    fn_count = 0
    for filename in tqdm(os.listdir(MASK_DIR), desc="Processing files"):
        print(filename)

        gt_path = os.path.join(args.gt_dir, filename)
        pred_path = os.path.join(args.pred_dir, filename)

        img2 = nib.load(pred_path)
        pred_mask = img2.get_fdata()  

        gt_img = nib.load(gt_path)
        gt_mask = gt_img.get_fdata()

        dice_score = compute_per_channel_dice(pred_mask, gt_mask)
        total_dice_score += dice_score.item()
        pred_mask = pred_mask.astype(np.uint8)
        gt_mask = gt_mask.astype(np.uint8)
        true_positives = (pred_mask & gt_mask).astype(float).sum()
        false_positives = (pred_mask & ~gt_mask).astype(float).sum()
        false_negatives = (~pred_mask & gt_mask).astype(float).sum()
        if true_positives + false_negatives < 1e-6:
            total_true_positives += 1
            total_false_negatives += 0
        else:
            total_true_positives += (true_positives/(true_positives + false_negatives))
            total_false_negatives += (false_negatives/(true_positives + false_negatives))
        if true_positives + false_positives < 1e-6:
            total_false_positives += 1
        else:
            total_false_positives += (true_positives/(false_positives + true_positives))
            

        tp_count += true_positives
        fp_count += false_positives
        fn_count += false_negatives

        count += 1
    recall = total_true_positives/count
    precision = total_false_positives/count
    fnr = total_false_negatives/count
    average_tp = tp_count/count
    average_fp = fp_count/count
    average_fn = fn_count/count
    print(f"dice coefficient: {total_dice_score / count:.4f}")
    print(f'Test True Positive: {average_tp}')
    print(f'Test False Positive: {average_fp}')
    print(f'Test False Negative: {average_fn}')
    print(f'Test precision: {precision}')
    print(f'Test recall: {recall}')
    print(f'Test fnr: {fnr}')
    print("FINISH")

if __name__ == '__main__':
    args = parse_args()
    main(args)