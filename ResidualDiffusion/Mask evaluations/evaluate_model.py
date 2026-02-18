#!/usr/bin/env python3
"""
Mask-based CBF Evaluation for Residual Diffusion Model

This script uses the SAVED predictions (out_*.npy files).
The test set uses a 25% split with seed=1337.
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# ============================================================
# Configuration
# ============================================================
SEED = 1337
np.random.seed(SEED)

BASE_DIR = Path(__file__).parent.parent  # ResidualDiffusion folder
ROOT_DIR = BASE_DIR.parent  # /data/rydham
EVAL_DIR = BASE_DIR / "Mask evaluations"
GLOBAL_EVAL_DIR = ROOT_DIR / "Mask evaluations"  # Global ground truth

PRE_SCANS_DIR = ROOT_DIR / "pre_scans"
POST_SCANS_DIR = ROOT_DIR / "post_scans"
MASKS_DIR = ROOT_DIR / "Masks"
PRED_DIR = BASE_DIR / "runs_residual_diffusion" / "pred_samples"

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


def get_test_patient_order():
    """
    Reproduce the exact test set order from the training code.
    Residual Diffusion uses 25% test split.
    """
    import re
    ID_RE = re.compile(r"(pre)_(\d{4})_(\d{3})\.nii\.gz$", re.IGNORECASE)
    
    def id_from_pre_path(p):
        m = ID_RE.search(os.path.basename(p))
        if not m:
            return None
        return m.group(2), m.group(3)
    
    def pre_to_post(pre_path):
        ids = id_from_pre_path(pre_path)
        if not ids:
            return None
        y, n = ids
        return str(POST_SCANS_DIR / f"post_{y}_{n}.nii.gz")
    
    # Get all paired scans
    all_pre = sorted(glob(str(PRE_SCANS_DIR / "pre_*.nii.gz")))
    paired = []
    for p in all_pre:
        q = pre_to_post(p)
        if q and os.path.exists(q):
            paired.append(p)
    
    # Same split as Residual Diffusion training: 25% test
    trainval_pre, test_pre = train_test_split(
        paired, test_size=0.25, random_state=SEED, shuffle=True
    )
    
    # Extract patient IDs from test set
    test_patients = []
    for pre_path in test_pre:
        ids = id_from_pre_path(pre_path)
        if ids:
            year, num = ids
            test_patients.append({
                'patient_id': f"{year}_{num}",
                'year': year,
                'id': num,
                'pre_path': pre_path
            })
    
    return test_patients


def load_mask_mapping():
    """Load the mask assignments from the global evaluation file."""
    patient_file = GLOBAL_EVAL_DIR / 'test_set_patients_moss.txt'
    mapping = {}
    with open(patient_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                patient_id = f"{parts[0]}_{parts[1]}"
                mapping[patient_id] = parts[3]
    return mapping


def load_mask_for_2d(mask_side, masks_dir, target_shape):
    """Load the appropriate MCA mask middle slice and resize."""
    def load_mask_middle_slice(mask_path):
        data = load_nifti(mask_path)
        D = data.shape[2]
        return data[:, :, D // 2]
    
    if mask_side == 'left':
        mask_slice = load_mask_middle_slice(masks_dir / 'MNI_left_MCA_2mm.nii.gz')
    elif mask_side == 'right':
        mask_slice = load_mask_middle_slice(masks_dir / 'MNI_right_MCA_2mm.nii.gz')
    elif mask_side == 'both':
        left = load_mask_middle_slice(masks_dir / 'MNI_left_MCA_2mm.nii.gz')
        right = load_mask_middle_slice(masks_dir / 'MNI_right_MCA_2mm.nii.gz')
        mask_slice = ((left > 0) | (right > 0)).astype(np.float32)
    else:
        raise ValueError(f"Invalid mask_side: {mask_side}")
    
    mask_t = torch.from_numpy(mask_slice).unsqueeze(0).unsqueeze(0).float()
    mask_t = F.interpolate(mask_t, size=target_shape, mode='nearest')
    return mask_t.squeeze().numpy() > 0


def compute_percentage_change_2d(pre_slice, post_slice, mask, is_unhealthy=True):
    """Compute percentage change in CBF for a 2D slice."""
    if is_unhealthy:
        region_mask = mask
    else:
        brain_mask = pre_slice > 1e-6
        region_mask = brain_mask & ~mask
    
    pre_values = pre_slice[region_mask]
    post_values = post_slice[region_mask]
    
    valid_idx = pre_values > 1e-6
    if not np.any(valid_idx):
        return np.nan, 0
    
    pre_valid = pre_values[valid_idx]
    post_valid = post_values[valid_idx]
    
    pct_change = ((post_valid - pre_valid) / pre_valid) * 100
    return np.mean(pct_change), len(pre_valid)


def create_patient_plot(patient_id, mask_side, gt_data, pred_data, save_path):
    """Create a bar plot for a single patient."""
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
    ax.set_title(f'Patient {patient_id} ({mask_side.upper()} disease)\nCBF Change: Ground Truth vs Residual Diffusion', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Ground Truth\n(Pre → Post)', 'Prediction\n(Pre → Predicted)'], fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
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


def create_cohort_summary_plot(results_by_side, save_path):
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
        
        gt_unhealthy = [d['gt_unhealthy'] for d in data if d['gt_unhealthy'] is not None]
        gt_healthy = [d['gt_healthy'] for d in data if d['gt_healthy'] is not None]
        pred_unhealthy = [d['pred_unhealthy'] for d in data if d['pred_unhealthy'] is not None]
        pred_healthy = [d['pred_healthy'] for d in data if d['pred_healthy'] is not None]
        
        x = np.arange(2)
        width = 0.35
        
        gt_means = [np.mean(gt_unhealthy) if gt_unhealthy else 0, 
                    np.mean(gt_healthy) if gt_healthy else 0]
        gt_errors = [np.std(gt_unhealthy)/np.sqrt(len(gt_unhealthy)) if len(gt_unhealthy) > 1 else 0,
                     np.std(gt_healthy)/np.sqrt(len(gt_healthy)) if len(gt_healthy) > 1 else 0]
        
        pred_means = [np.mean(pred_unhealthy) if pred_unhealthy else 0,
                      np.mean(pred_healthy) if pred_healthy else 0]
        pred_errors = [np.std(pred_unhealthy)/np.sqrt(len(pred_unhealthy)) if len(pred_unhealthy) > 1 else 0,
                       np.std(pred_healthy)/np.sqrt(len(pred_healthy)) if len(pred_healthy) > 1 else 0]
        
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
    
    plt.suptitle('Residual Diffusion: Cohort Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("Residual Diffusion - Mask-based CBF Evaluation")
    print("=" * 60)
    
    EVAL_DIR.mkdir(exist_ok=True)
    
    # Get the test patient order
    test_patients = get_test_patient_order()
    print(f"\nFound {len(test_patients)} test patients from 25% split")
    
    # Load mask assignments  
    mask_mapping = load_mask_mapping()
    print(f"Loaded mask assignments for {len(mask_mapping)} patients")
    
    # Load ground truth evaluation
    gt_eval_path = GLOBAL_EVAL_DIR / 'ground_truth_evaluation.json'
    with open(gt_eval_path, 'r') as f:
        gt_eval = json.load(f)
    gt_by_patient = {p['patient_id']: p for p in gt_eval['patients']}
    
    # Process each patient
    results = []
    results_by_side = {'left': [], 'right': [], 'both': []}
    
    print(f"\nProcessing patients...")
    for idx, patient in enumerate(test_patients):
        patient_id = patient['patient_id']
        
        if patient_id in EXCLUDE_PATIENTS:
            print(f"  [{idx+1}/{len(test_patients)}] {patient_id}... SKIP (excluded)")
            continue
        
        mask_side = mask_mapping.get(patient_id)
        if not mask_side:
            print(f"  [{idx+1}/{len(test_patients)}] {patient_id}... SKIP (no mask assignment)")
            continue
        
        print(f"  [{idx+1}/{len(test_patients)}] {patient_id} ({mask_side})...", end=" ")
        
        # Load saved predictions
        in_path = PRED_DIR / f"in_{idx:04d}.npy"
        out_path = PRED_DIR / f"out_{idx:04d}.npy"
        
        if not in_path.exists() or not out_path.exists():
            print("SKIP (no saved prediction)")
            continue
        
        pre_slice = np.load(in_path)
        pred_slice = np.load(out_path)
        
        # Load mask
        unhealthy_mask = load_mask_for_2d(mask_side, MASKS_DIR, pre_slice.shape)
        
        # Compute CBF % change for prediction
        pred_unhealthy_pct, pred_unhealthy_count = compute_percentage_change_2d(
            pre_slice, pred_slice, unhealthy_mask, is_unhealthy=True
        )
        pred_healthy_pct, pred_healthy_count = compute_percentage_change_2d(
            pre_slice, pred_slice, unhealthy_mask, is_unhealthy=False
        )
        
        # Get ground truth
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
    
    # Save results
    output = {
        'model': 'ResidualDiffusion',
        'description': 'CBF % change comparison: Ground Truth vs Residual Diffusion',
        'note': 'Evaluation on middle axial slice (uses saved predictions from 25% test split)',
        'patients': results
    }
    
    results_path = EVAL_DIR / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Create cohort summary plot
    print("\nGenerating cohort summary plot...")
    summary_plot_path = EVAL_DIR / 'cohort_summary.png'
    create_cohort_summary_plot(results_by_side, summary_plot_path)
    print(f"Cohort summary saved to: {summary_plot_path}")
    
    # Print summary
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
        if gt_unhealthy:
            print(f"    Unhealthy: {np.mean(gt_unhealthy):.2f}% ± {np.std(gt_unhealthy):.2f}%")
        if gt_healthy:
            print(f"    Healthy:   {np.mean(gt_healthy):.2f}% ± {np.std(gt_healthy):.2f}%")
        print(f"  Residual Diffusion Prediction:")
        if pred_unhealthy:
            print(f"    Unhealthy: {np.mean(pred_unhealthy):.2f}% ± {np.std(pred_unhealthy):.2f}%")
        if pred_healthy:
            print(f"    Healthy:   {np.mean(pred_healthy):.2f}% ± {np.std(pred_healthy):.2f}%")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
