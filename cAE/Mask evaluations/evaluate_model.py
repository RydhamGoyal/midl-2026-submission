#!/usr/bin/env python3
"""
Mask-based CBF Evaluation for UNet Baseline Model

This script:
1. Loads the trained UNet model
2. Runs inference on all test patients (middle axial slice)
3. Applies MCA masks (middle slice) to compute CBF % change
4. Generates per-patient and cohort summary plots
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.networks.nets import UNet

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).parent.parent  # cAE folder
ROOT_DIR = BASE_DIR.parent  # /data/rydham
EVAL_DIR = BASE_DIR / "Mask evaluations"
GLOBAL_EVAL_DIR = ROOT_DIR / "Mask evaluations"  # Global ground truth

PRE_SCANS_DIR = ROOT_DIR / "pre_scans"
POST_SCANS_DIR = ROOT_DIR / "post_scans"
MASKS_DIR = ROOT_DIR / "Masks"
MODEL_PATH = BASE_DIR / "runs_axial_post_from_pre" / "best_unet_axial_postfrompre.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = (128, 128)

# Patients to exclude (corrupted data)
EXCLUDE_PATIENTS = ["2023_035"]


# ============================================================
# Utility Functions
# ============================================================
def load_nifti(path):
    """Load a NIfTI file and return the data array."""
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32)


def load_middle_axial_slice(nii_path):
    """Load middle axial slice from a NIfTI file."""
    data = load_nifti(nii_path)
    if data.ndim == 4:
        data = data[..., 0]
    H, W, D = data.shape
    mid = D // 2
    return data[:, :, mid], mid, data.shape


def minmax_norm(x, eps=1e-6):
    """Normalize array to [0,1]."""
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def load_mask_middle_slice(mask_side, masks_dir, slice_idx, vol_shape):
    """
    Load the appropriate MCA mask and extract the middle slice.
    Resize to match the volume shape if needed.
    """
    if mask_side == 'left':
        mask_path = masks_dir / 'MNI_left_MCA_2mm.nii.gz'
        mask_data = load_nifti(mask_path)
    elif mask_side == 'right':
        mask_path = masks_dir / 'MNI_right_MCA_2mm.nii.gz'
        mask_data = load_nifti(mask_path)
    elif mask_side == 'both':
        left_data = load_nifti(masks_dir / 'MNI_left_MCA_2mm.nii.gz')
        right_data = load_nifti(masks_dir / 'MNI_right_MCA_2mm.nii.gz')
        mask_data = (left_data > 0) | (right_data > 0)
        mask_data = mask_data.astype(np.float32)
    else:
        raise ValueError(f"Invalid mask_side: {mask_side}")
    
    # Get the middle slice from mask
    mask_H, mask_W, mask_D = mask_data.shape
    mask_mid = mask_D // 2
    mask_slice = mask_data[:, :, mask_mid]
    
    # Resize mask to match target volume shape (H, W)
    mask_slice_t = torch.from_numpy(mask_slice).unsqueeze(0).unsqueeze(0).float()
    mask_slice_t = F.interpolate(mask_slice_t, size=(vol_shape[0], vol_shape[1]), 
                                  mode='nearest')
    mask_slice = mask_slice_t.squeeze().numpy()
    
    return mask_slice > 0


def make_unet_2d(ch=(16, 32, 64, 128)):
    """Create the UNet model (same architecture as training)."""
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=list(ch),
        strides=(2, 2, 2),
        num_res_units=2,
        act=("LeakyReLU", {"inplace": True}),
        norm="INSTANCE",
        dropout=0.0,
    )
    return model


def compute_percentage_change_2d(pre_slice, post_slice, mask, is_unhealthy=True):
    """
    Compute percentage change in CBF for a 2D slice.
    
    Note: Since the model normalizes inputs to [0,1], we compute change
    on the normalized values, which represents relative intensity change.
    """
    if is_unhealthy:
        region_mask = mask
    else:
        # Healthy = brain tissue not in mask
        brain_mask = pre_slice > 0
        region_mask = brain_mask & ~mask
    
    pre_values = pre_slice[region_mask]
    post_values = post_slice[region_mask]
    
    # Avoid division by zero
    valid_idx = pre_values > 1e-6
    if not np.any(valid_idx):
        return np.nan, 0
    
    pre_valid = pre_values[valid_idx]
    post_valid = post_values[valid_idx]
    
    # Percentage change per pixel
    pct_change = ((post_valid - pre_valid) / pre_valid) * 100
    
    return np.mean(pct_change), len(pre_valid)


def parse_patient_file(filepath):
    """Parse the test set patients file with mask assignments."""
    patients = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                patient_id = f"{parts[0]}_{parts[1]}"
                if patient_id not in EXCLUDE_PATIENTS:
                    patients.append({
                        'year': parts[0],
                        'id': parts[1],
                        'patient_id': patient_id,
                        'pre_filename': parts[2],
                        'mask_side': parts[3]
                    })
    return patients


def run_inference(model, pre_slice_norm, original_shape):
    """Run model inference on a normalized pre-scan slice."""
    # Convert to tensor and resize to model input size
    pre_t = torch.from_numpy(pre_slice_norm).unsqueeze(0).unsqueeze(0).float()
    pre_t = F.interpolate(pre_t, size=TARGET_SIZE, mode='bilinear', align_corners=False)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        pre_t = pre_t.to(DEVICE)
        out_t = model(pre_t)
    
    # Resize back to original shape
    out_t = F.interpolate(out_t, size=(original_shape[0], original_shape[1]), 
                          mode='bilinear', align_corners=False)
    out_np = out_t.cpu().squeeze().numpy()
    
    return out_np


def create_patient_plot(patient_id, mask_side, gt_data, pred_data, save_path):
    """Create a bar plot for a single patient comparing ground truth vs prediction."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(2)
    width = 0.35
    
    unhealthy_values = [gt_data['unhealthy_pct_change'], pred_data['unhealthy_pct_change']]
    healthy_values = [gt_data['healthy_pct_change'], pred_data['healthy_pct_change']]
    
    bars1 = ax.bar(x - width/2, unhealthy_values, width, label='Unhealthy Region', 
                   color='#3498db', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, healthy_values, width, label='Healthy Region', 
                   color='#2c3e50', edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('CBF % Change', fontsize=12, fontweight='bold')
    ax.set_title(f'Patient {patient_id} ({mask_side.upper()} disease)\nCBF Change: Ground Truth vs UNet Prediction', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Ground Truth\n(Pre → Post)', 'Prediction\n(Pre → Predicted)'], fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_cohort_summary_plot(results_by_side, save_path, title_suffix=""):
    """Create cohort summary plot with error bars."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sides = ['left', 'right', 'both']
    side_titles = ['LEFT Disease', 'RIGHT Disease', 'BILATERAL Disease']
    
    for ax, side, side_title in zip(axes, sides, side_titles):
        if side not in results_by_side or len(results_by_side[side]) == 0:
            ax.text(0.5, 0.5, f'No patients with {side} disease', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(side_title)
            continue
        
        data = results_by_side[side]
        
        # Ground truth values
        gt_unhealthy = [d['gt_unhealthy'] for d in data if d['gt_unhealthy'] is not None]
        gt_healthy = [d['gt_healthy'] for d in data if d['gt_healthy'] is not None]
        
        # Prediction values
        pred_unhealthy = [d['pred_unhealthy'] for d in data if d['pred_unhealthy'] is not None]
        pred_healthy = [d['pred_healthy'] for d in data if d['pred_healthy'] is not None]
        
        x = np.arange(2)
        width = 0.35
        
        # Means and standard errors
        gt_means = [np.mean(gt_unhealthy) if gt_unhealthy else 0, 
                    np.mean(gt_healthy) if gt_healthy else 0]
        gt_errors = [np.std(gt_unhealthy)/np.sqrt(len(gt_unhealthy)) if len(gt_unhealthy) > 1 else 0,
                     np.std(gt_healthy)/np.sqrt(len(gt_healthy)) if len(gt_healthy) > 1 else 0]
        
        pred_means = [np.mean(pred_unhealthy) if pred_unhealthy else 0,
                      np.mean(pred_healthy) if pred_healthy else 0]
        pred_errors = [np.std(pred_unhealthy)/np.sqrt(len(pred_unhealthy)) if len(pred_unhealthy) > 1 else 0,
                       np.std(pred_healthy)/np.sqrt(len(pred_healthy)) if len(pred_healthy) > 1 else 0]
        
        # Plot bars with error bars
        bars1 = ax.bar(x - width/2, [gt_means[0], pred_means[0]], width, 
                       yerr=[gt_errors[0], pred_errors[0]], capsize=5,
                       label='Unhealthy Region', color='#3498db', 
                       edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, [gt_means[1], pred_means[1]], width,
                       yerr=[gt_errors[1], pred_errors[1]], capsize=5,
                       label='Healthy Region', color='#2c3e50',
                       edgecolor='black', linewidth=1.2)
        
        ax.set_ylabel('Mean CBF % Change', fontsize=11)
        ax.set_title(f'{side_title}\n(n={len(data)} patients)', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Ground Truth\n(Pre→Post)', 'Prediction\n(Pre→Pred)'], fontsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.suptitle(f'Conditional Autoencoder (cAE): Cohort Summary{title_suffix}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("UNet Baseline - Mask-based CBF Evaluation")
    print("=" * 60)
    
    # Create output directory
    EVAL_DIR.mkdir(exist_ok=True)
    
    # Load patient list with mask assignments
    patient_file = GLOBAL_EVAL_DIR / 'test_set_patients_moss.txt'
    patients = parse_patient_file(patient_file)
    print(f"\nLoaded {len(patients)} test patients (excluding {EXCLUDE_PATIENTS})")
    
    # Load ground truth evaluation
    gt_eval_path = GLOBAL_EVAL_DIR / 'ground_truth_evaluation.json'
    with open(gt_eval_path, 'r') as f:
        gt_eval = json.load(f)
    gt_by_patient = {p['patient_id']: p for p in gt_eval['patients']}
    print(f"Loaded ground truth evaluation for {len(gt_by_patient)} patients")
    
    # Load the trained model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = make_unet_2d(ch=(32, 64, 128, 256)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully!")
    
    # Process each patient
    results = []
    results_by_side = {'left': [], 'right': [], 'both': []}
    
    print(f"\nProcessing patients...")
    for i, patient in enumerate(patients):
        patient_id = patient['patient_id']
        mask_side = patient['mask_side']
        
        print(f"  [{i+1}/{len(patients)}] {patient_id} ({mask_side})...", end=" ")
        
        # Load pre and post scans (middle slice)
        pre_path = PRE_SCANS_DIR / patient['pre_filename']
        post_filename = patient['pre_filename'].replace('pre_', 'post_')
        post_path = POST_SCANS_DIR / post_filename
        
        if not pre_path.exists() or not post_path.exists():
            print("SKIP (missing files)")
            continue
        
        try:
            pre_slice, slice_idx, vol_shape = load_middle_axial_slice(str(pre_path))
            post_slice, _, _ = load_middle_axial_slice(str(post_path))
        except Exception as e:
            print(f"SKIP (error loading: {e})")
            continue
        
        # Normalize slices
        pre_slice_norm = minmax_norm(pre_slice)
        post_slice_norm = minmax_norm(post_slice)
        
        # Run inference to get prediction
        pred_slice_norm = run_inference(model, pre_slice_norm, vol_shape[:2])
        
        # Load mask (middle slice, resized to match)
        unhealthy_mask = load_mask_middle_slice(mask_side, MASKS_DIR, slice_idx, vol_shape)
        
        # Compute CBF % change for prediction
        pred_unhealthy_pct, pred_unhealthy_count = compute_percentage_change_2d(
            pre_slice_norm, pred_slice_norm, unhealthy_mask, is_unhealthy=True
        )
        pred_healthy_pct, pred_healthy_count = compute_percentage_change_2d(
            pre_slice_norm, pred_slice_norm, unhealthy_mask, is_unhealthy=False
        )
        
        # Get ground truth from pre-computed evaluation
        gt_data = gt_by_patient.get(patient_id, {})
        gt_unhealthy_pct = gt_data.get('unhealthy_pct_change')
        gt_healthy_pct = gt_data.get('healthy_pct_change')
        
        # Store result
        result = {
            'patient_id': patient_id,
            'mask_side': mask_side,
            'gt_unhealthy': gt_unhealthy_pct,
            'gt_healthy': gt_healthy_pct,
            'pred_unhealthy': float(pred_unhealthy_pct) if not np.isnan(pred_unhealthy_pct) else None,
            'pred_healthy': float(pred_healthy_pct) if not np.isnan(pred_healthy_pct) else None,
            'pred_unhealthy_pixel_count': int(pred_unhealthy_count),
            'pred_healthy_pixel_count': int(pred_healthy_count)
        }
        results.append(result)
        results_by_side[mask_side].append(result)
        
        # Create per-patient plot
        patient_plot_path = EVAL_DIR / f"patient_{patient_id}_{mask_side}.png"
        gt_plot_data = {
            'unhealthy_pct_change': gt_unhealthy_pct if gt_unhealthy_pct else 0,
            'healthy_pct_change': gt_healthy_pct if gt_healthy_pct else 0
        }
        pred_plot_data = {
            'unhealthy_pct_change': pred_unhealthy_pct if not np.isnan(pred_unhealthy_pct) else 0,
            'healthy_pct_change': pred_healthy_pct if not np.isnan(pred_healthy_pct) else 0
        }
        create_patient_plot(patient_id, mask_side, gt_plot_data, pred_plot_data, patient_plot_path)
        
        print("OK")
    
    # Save detailed results
    output = {
        'model': 'cAE',
        'description': 'CBF % change comparison: Ground Truth vs UNet Prediction',
        'note': 'Evaluation on middle axial slice only (model limitation)',
        'patients': results
    }
    
    results_path = EVAL_DIR / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Create cohort summary plots
    print("\nGenerating cohort summary plot...")
    summary_plot_path = EVAL_DIR / 'cohort_summary.png'
    create_cohort_summary_plot(results_by_side, summary_plot_path)
    print(f"Cohort summary saved to: {summary_plot_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    for side in ['left', 'right', 'both']:
        data = results_by_side[side]
        if not data:
            continue
        
        gt_unhealthy = [d['gt_unhealthy'] for d in data if d['gt_unhealthy'] is not None]
        gt_healthy = [d['gt_healthy'] for d in data if d['gt_healthy'] is not None]
        pred_unhealthy = [d['pred_unhealthy'] for d in data if d['pred_unhealthy'] is not None]
        pred_healthy = [d['pred_healthy'] for d in data if d['pred_healthy'] is not None]
        
        print(f"\n{side.upper()} Disease (n={len(data)}):")
        print(f"  Ground Truth:")
        print(f"    Unhealthy: {np.mean(gt_unhealthy):.2f}% ± {np.std(gt_unhealthy):.2f}%")
        print(f"    Healthy:   {np.mean(gt_healthy):.2f}% ± {np.std(gt_healthy):.2f}%")
        print(f"  UNet Prediction:")
        print(f"    Unhealthy: {np.mean(pred_unhealthy):.2f}% ± {np.std(pred_unhealthy):.2f}%")
        print(f"    Healthy:   {np.mean(pred_healthy):.2f}% ± {np.std(pred_healthy):.2f}%")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
