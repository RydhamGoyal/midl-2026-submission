#!/usr/bin/env python3
"""
Unified Cohort Summary Analysis

This script generates cohort summary plots for ALL models using pixel-by-pixel
percentage change calculation with ±200% capping.

Formula:
    For each pixel: pct = (post - pre) / pre * 100
    Cap at ±200% to remove physiologically implausible outliers
    Only include pixels where pre > 0.05 (5% threshold)
    Return mean of capped values

Output:
- cohort_summary_{model}.png for each model
- cohort_statistics.json with mean/std/variance for all models
"""

import os
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

import torch
import torch.nn.functional as F
from monai.networks.nets import UNet

# ============================================================
# Configuration
# ============================================================
ROOT_DIR = Path("/data/rydham")
EVAL_DIR = ROOT_DIR / "Mask evaluations"
OUTPUT_DIR = EVAL_DIR / "Cohort Summary"
OUTPUT_DIR.mkdir(exist_ok=True)

PRE_SCANS_DIR = ROOT_DIR / "pre_scans"
POST_SCANS_DIR = ROOT_DIR / "post_scans"
MASKS_DIR = ROOT_DIR / "Masks"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = (128, 128)

# Models to analyze
MODELS = {
    "cAE": {
        "pred_dir": ROOT_DIR / "cAE" / "runs_axial_post_from_pre" / "pred_samples",
        "display_name": "cAE"
    },
    "DDPM": {
        "pred_dir": ROOT_DIR / "DDPM" / "runs_diffusion_post_from_pre" / "pred_samples",
        "display_name": "DDPM"
    },
    "ColdDiffusion": {
        "pred_dir": ROOT_DIR / "ColdDiffusion" / "runs_cold_diffusion" / "pred_samples",
        "display_name": "Cold Diffusion"
    },
    "ResidualDiffusion": {
        "pred_dir": ROOT_DIR / "ResidualDiffusion" / "runs_residual_diffusion" / "pred_samples",
        "display_name": "Residual Diffusion"
    }
}

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


def load_mask_middle_slice(mask_side, masks_dir, vol_shape):
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


def compute_percentage_change_pixel_capped(pre_slice, post_slice, mask, is_unhealthy=True,
                                            min_threshold=0.05, cap_value=200.0):
    """
    Compute percentage change in CBF using PIXEL-BY-PIXEL calculation with capping.
    
    Formula: 
        For each pixel: pct = (post - pre) / pre * 100
        Cap at ±cap_value (default ±200%)
        Return mean of capped values
    
    This preserves biological signal while removing extreme outliers.
    
    Args:
        pre_slice: Pre-scan normalized slice
        post_slice: Post-scan (or prediction) normalized slice
        mask: Binary mask for unhealthy region
        is_unhealthy: If True, use mask region; if False, use inverse (healthy)
        min_threshold: Minimum pre value to include (avoids division by ~0)
        cap_value: Maximum absolute percentage change to allow (±cap_value)
    """
    if is_unhealthy:
        region_mask = mask
    else:
        # Healthy = brain tissue not in mask
        brain_mask = pre_slice > min_threshold
        region_mask = brain_mask & ~mask
    
    # Also require minimum threshold for unhealthy region
    if is_unhealthy:
        region_mask = region_mask & (pre_slice > min_threshold)
    
    pre_values = pre_slice[region_mask]
    post_values = post_slice[region_mask]
    
    if len(pre_values) == 0:
        return np.nan, 0
    
    # Compute pixel-by-pixel percentage change
    # pre_values already filtered to be > min_threshold
    pct_change = ((post_values - pre_values) / pre_values) * 100
    
    # Cap at ±cap_value to remove physiologically implausible outliers
    pct_change_capped = np.clip(pct_change, -cap_value, cap_value)
    
    return np.mean(pct_change_capped), len(pre_values)


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


def load_prediction_slice(pred_dir, patient_idx):
    """Load a prediction slice from the model's pred_samples directory."""
    out_path = pred_dir / f"out_{patient_idx:04d}.npy"
    if out_path.exists():
        return np.load(out_path)
    return None


def create_cohort_summary_plot(results_by_side, model_name, save_path):
    """Create cohort summary plot with error bars for a single model."""
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
        gt_unhealthy = [d['gt_unhealthy'] for d in data if d['gt_unhealthy'] is not None and not np.isnan(d['gt_unhealthy'])]
        gt_healthy = [d['gt_healthy'] for d in data if d['gt_healthy'] is not None and not np.isnan(d['gt_healthy'])]
        
        # Prediction values
        pred_unhealthy = [d['pred_unhealthy'] for d in data if d['pred_unhealthy'] is not None and not np.isnan(d['pred_unhealthy'])]
        pred_healthy = [d['pred_healthy'] for d in data if d['pred_healthy'] is not None and not np.isnan(d['pred_healthy'])]
        
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
                       label='Unhealthy Region', color='#e74c3c', 
                       edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, [gt_means[1], pred_means[1]], width,
                       yerr=[gt_errors[1], pred_errors[1]], capsize=5,
                       label='Healthy Region', color='#27ae60',
                       edgecolor='black', linewidth=1.2)
        
        ax.set_ylabel('Mean CBF % Change', fontsize=11)
        ax.set_title(f'{side_title}\n(n={len(data)} patients)', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Ground Truth\n(Pre→Post)', 'Prediction\n(Pre→Pred)'], fontsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height) and abs(height) > 0.1:
                    ax.annotate(f'{height:.1f}%',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3 if height >= 0 else -12), 
                               textcoords="offset points",
                               ha='center', va='bottom' if height >= 0 else 'top', 
                               fontsize=8)
    
    plt.suptitle(f'{model_name}: Cohort Summary (Pixel-by-Pixel, ±200% Cap)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_model(model_key, model_config, patients, gt_results):
    """Evaluate a single model's predictions."""
    pred_dir = model_config["pred_dir"]
    display_name = model_config["display_name"]
    
    if not pred_dir.exists():
        print(f"  ⚠️  Prediction directory not found: {pred_dir}")
        return None
    
    results = []
    results_by_side = {'left': [], 'right': [], 'both': []}
    
    for i, patient in enumerate(patients):
        patient_id = patient['patient_id']
        mask_side = patient['mask_side']
        
        # Load ground truth from pre-computed results
        gt_data = gt_results.get(patient_id, {})
        
        # Load pre and post scans (middle slice)
        pre_path = PRE_SCANS_DIR / patient['pre_filename']
        post_filename = patient['pre_filename'].replace('pre_', 'post_')
        post_path = POST_SCANS_DIR / post_filename
        
        if not pre_path.exists() or not post_path.exists():
            continue
        
        try:
            pre_slice, slice_idx, vol_shape = load_middle_axial_slice(str(pre_path))
            post_slice, _, _ = load_middle_axial_slice(str(post_path))
        except Exception as e:
            continue
        
        # Normalize slices
        pre_slice_norm = minmax_norm(pre_slice)
        post_slice_norm = minmax_norm(post_slice)
        
        # Load prediction
        pred_slice = load_prediction_slice(pred_dir, i)
        if pred_slice is None:
            continue
        
        # Resize prediction to match original shape if needed
        if pred_slice.shape != pre_slice_norm.shape:
            pred_t = torch.from_numpy(pred_slice).unsqueeze(0).unsqueeze(0).float()
            pred_t = F.interpolate(pred_t, size=pre_slice_norm.shape, mode='bilinear', align_corners=False)
            pred_slice = pred_t.squeeze().numpy()
        
        # Load mask
        unhealthy_mask = load_mask_middle_slice(mask_side, MASKS_DIR, vol_shape)
        
        # Resize mask to match normalized slice shape
        if unhealthy_mask.shape != pre_slice_norm.shape:
            mask_t = torch.from_numpy(unhealthy_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            mask_t = F.interpolate(mask_t, size=pre_slice_norm.shape, mode='nearest')
            unhealthy_mask = mask_t.squeeze().numpy() > 0
        
        # Compute ground truth CBF % change (pixel-by-pixel with ±200% cap)
        gt_unhealthy_pct, _ = compute_percentage_change_pixel_capped(
            pre_slice_norm, post_slice_norm, unhealthy_mask, is_unhealthy=True
        )
        gt_healthy_pct, _ = compute_percentage_change_pixel_capped(
            pre_slice_norm, post_slice_norm, unhealthy_mask, is_unhealthy=False
        )
        
        # Compute prediction CBF % change (pixel-by-pixel with ±200% cap)
        pred_unhealthy_pct, _ = compute_percentage_change_pixel_capped(
            pre_slice_norm, pred_slice, unhealthy_mask, is_unhealthy=True
        )
        pred_healthy_pct, _ = compute_percentage_change_pixel_capped(
            pre_slice_norm, pred_slice, unhealthy_mask, is_unhealthy=False
        )
        
        result = {
            'patient_id': patient_id,
            'mask_side': mask_side,
            'gt_unhealthy': float(gt_unhealthy_pct) if not np.isnan(gt_unhealthy_pct) else None,
            'gt_healthy': float(gt_healthy_pct) if not np.isnan(gt_healthy_pct) else None,
            'pred_unhealthy': float(pred_unhealthy_pct) if not np.isnan(pred_unhealthy_pct) else None,
            'pred_healthy': float(pred_healthy_pct) if not np.isnan(pred_healthy_pct) else None,
        }
        results.append(result)
        results_by_side[mask_side].append(result)
    
    return {
        'model_key': model_key,
        'display_name': display_name,
        'results': results,
        'results_by_side': results_by_side
    }


def compute_statistics(results_by_side):
    """Compute summary statistics for a model."""
    stats = {}
    for side in ['left', 'right', 'both']:
        data = results_by_side.get(side, [])
        if not data:
            continue
        
        gt_unhealthy = [d['gt_unhealthy'] for d in data if d['gt_unhealthy'] is not None]
        gt_healthy = [d['gt_healthy'] for d in data if d['gt_healthy'] is not None]
        pred_unhealthy = [d['pred_unhealthy'] for d in data if d['pred_unhealthy'] is not None]
        pred_healthy = [d['pred_healthy'] for d in data if d['pred_healthy'] is not None]
        
        stats[side] = {
            'n_patients': len(data),
            'ground_truth': {
                'unhealthy': {
                    'mean': float(np.mean(gt_unhealthy)) if gt_unhealthy else None,
                    'std': float(np.std(gt_unhealthy)) if gt_unhealthy else None,
                    'variance': float(np.var(gt_unhealthy)) if gt_unhealthy else None,
                },
                'healthy': {
                    'mean': float(np.mean(gt_healthy)) if gt_healthy else None,
                    'std': float(np.std(gt_healthy)) if gt_healthy else None,
                    'variance': float(np.var(gt_healthy)) if gt_healthy else None,
                }
            },
            'prediction': {
                'unhealthy': {
                    'mean': float(np.mean(pred_unhealthy)) if pred_unhealthy else None,
                    'std': float(np.std(pred_unhealthy)) if pred_unhealthy else None,
                    'variance': float(np.var(pred_unhealthy)) if pred_unhealthy else None,
                },
                'healthy': {
                    'mean': float(np.mean(pred_healthy)) if pred_healthy else None,
                    'std': float(np.std(pred_healthy)) if pred_healthy else None,
                    'variance': float(np.var(pred_healthy)) if pred_healthy else None,
                }
            }
        }
    return stats


def main():
    print("=" * 70)
    print("Unified Cohort Summary Analysis (Pixel-by-Pixel, ±200% Cap)")
    print("=" * 70)
    
    # Load patient list
    patient_file = EVAL_DIR / 'test_set_patients_moss.txt'
    if not patient_file.exists():
        print(f"❌ Patient file not found: {patient_file}")
        return
    
    patients = parse_patient_file(patient_file)
    print(f"\nLoaded {len(patients)} test patients")
    
    # Pre-compute ground truth (will be recalculated with Option D for each model)
    gt_results = {}  # We'll calculate fresh for each model
    
    all_model_stats = {}
    
    # Evaluate each model
    for model_key, model_config in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_config['display_name']}")
        print(f"{'='*50}")
        
        model_results = evaluate_model(model_key, model_config, patients, gt_results)
        
        if model_results is None:
            print(f"  Skipping {model_key} - no predictions found")
            continue
        
        # Generate cohort summary plot
        plot_path = OUTPUT_DIR / f"cohort_summary_{model_key}.png"
        create_cohort_summary_plot(
            model_results['results_by_side'],
            model_results['display_name'],
            plot_path
        )
        print(f"  ✅ Saved: {plot_path.name}")
        
        # Compute statistics
        stats = compute_statistics(model_results['results_by_side'])
        all_model_stats[model_key] = {
            'display_name': model_results['display_name'],
            'statistics': stats,
            'n_patients_evaluated': len(model_results['results'])
        }
        
        # Print summary
        for side in ['left', 'right', 'both']:
            if side in stats:
                s = stats[side]
                print(f"\n  {side.upper()} Disease (n={s['n_patients']}):")
                gt = s['ground_truth']
                pred = s['prediction']
                if gt['unhealthy']['mean'] is not None:
                    print(f"    Ground Truth - Unhealthy: {gt['unhealthy']['mean']:.2f}% ± {gt['unhealthy']['std']:.2f}%")
                    print(f"    Ground Truth - Healthy:   {gt['healthy']['mean']:.2f}% ± {gt['healthy']['std']:.2f}%")
                if pred['unhealthy']['mean'] is not None:
                    print(f"    Prediction   - Unhealthy: {pred['unhealthy']['mean']:.2f}% ± {pred['unhealthy']['std']:.2f}%")
                    print(f"    Prediction   - Healthy:   {pred['healthy']['mean']:.2f}% ± {pred['healthy']['std']:.2f}%")
    
    # Save all statistics to JSON
    stats_path = OUTPUT_DIR / "cohort_statistics.json"
    output = {
        'description': 'Cohort summary statistics using pixel-by-pixel % change with ±200% cap',
        'formula': 'For each pixel: pct = (post - pre) / pre * 100, capped at ±200%, threshold pre > 0.05',
        'models': all_model_stats
    }
    with open(stats_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Statistics saved to: {stats_path}")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
