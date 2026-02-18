#!/usr/bin/env python3
"""
Ground Truth Mask Evaluation Script

Computes percentage change in CBF between pre-scan (input) and post-scan (ground truth)
for healthy and unhealthy brain regions, using MCA masks.

Output: JSON file with per-patient metrics and summary statistics.
"""

import os
import json
import numpy as np
import nibabel as nib
from pathlib import Path


def load_nifti(path):
    """Load a NIfTI file and return the data array."""
    img = nib.load(path)
    return img.get_fdata()


def load_mask(mask_path):
    """Load a mask file and return binary mask (1 = masked region)."""
    mask_data = load_nifti(mask_path)
    return mask_data > 0


def get_combined_mask(mask_side, masks_dir):
    """
    Get the combined unhealthy region mask based on disease side.
    
    Args:
        mask_side: 'left', 'right', or 'both'
        masks_dir: Path to masks directory
        
    Returns:
        Binary mask where True = unhealthy region
    """
    left_mask_path = os.path.join(masks_dir, 'MNI_left_MCA_2mm.nii.gz')
    right_mask_path = os.path.join(masks_dir, 'MNI_right_MCA_2mm.nii.gz')
    
    if mask_side == 'left':
        return load_mask(left_mask_path)
    elif mask_side == 'right':
        return load_mask(right_mask_path)
    elif mask_side == 'both':
        left_mask = load_mask(left_mask_path)
        right_mask = load_mask(right_mask_path)
        return left_mask | right_mask
    else:
        raise ValueError(f"Invalid mask_side: {mask_side}")


def compute_percentage_change(pre_data, post_data, mask, is_unhealthy=True):
    """
    Compute the percentage change in CBF for a specific region.
    
    Args:
        pre_data: Pre-scan 3D array
        post_data: Post-scan 3D array
        mask: Binary mask for unhealthy region
        is_unhealthy: If True, compute for masked (unhealthy) region; 
                      if False, compute for non-masked (healthy) region
                      
    Returns:
        Mean percentage change across all pixels in the region
    """
    if is_unhealthy:
        region_mask = mask
    else:
        # Healthy region = brain tissue NOT in the unhealthy mask
        # We need to also exclude background (zero values in pre-scan)
        brain_mask = pre_data > 0
        region_mask = brain_mask & ~mask
    
    # Get pixel values in the region
    pre_values = pre_data[region_mask]
    post_values = post_data[region_mask]
    
    # Avoid division by zero - only consider pixels with positive pre values
    valid_idx = pre_values > 0
    if not np.any(valid_idx):
        return np.nan, 0
    
    pre_valid = pre_values[valid_idx]
    post_valid = post_values[valid_idx]
    
    # Compute percentage change per pixel: (post - pre) / pre * 100
    pct_change = ((post_valid - pre_valid) / pre_valid) * 100
    
    return np.mean(pct_change), len(pre_valid)


def parse_patient_file(filepath):
    """
    Parse the test set patients file.
    
    Returns:
        List of dicts with keys: year, id, pre_filename, mask_side
    """
    patients = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                patients.append({
                    'year': parts[0],
                    'id': parts[1],
                    'pre_filename': parts[2],
                    'mask_side': parts[3]
                })
    return patients


def evaluate_patient(patient, pre_scans_dir, post_scans_dir, masks_dir):
    """
    Evaluate a single patient's ground truth CBF changes.
    
    Returns:
        Dict with patient metrics
    """
    # Construct file paths
    pre_path = os.path.join(pre_scans_dir, patient['pre_filename'])
    post_filename = patient['pre_filename'].replace('pre_', 'post_')
    post_path = os.path.join(post_scans_dir, post_filename)
    
    # Check files exist
    if not os.path.exists(pre_path):
        return {'error': f"Pre-scan not found: {pre_path}"}
    if not os.path.exists(post_path):
        return {'error': f"Post-scan not found: {post_path}"}
    
    # Load data
    pre_data = load_nifti(pre_path)
    post_data = load_nifti(post_path)
    
    # Get mask
    unhealthy_mask = get_combined_mask(patient['mask_side'], masks_dir)
    
    # Compute percentage changes
    unhealthy_pct_change, unhealthy_pixel_count = compute_percentage_change(
        pre_data, post_data, unhealthy_mask, is_unhealthy=True
    )
    healthy_pct_change, healthy_pixel_count = compute_percentage_change(
        pre_data, post_data, unhealthy_mask, is_unhealthy=False
    )
    
    return {
        'patient_id': f"{patient['year']}_{patient['id']}",
        'mask_side': patient['mask_side'],
        'unhealthy_pct_change': float(unhealthy_pct_change) if not np.isnan(unhealthy_pct_change) else None,
        'healthy_pct_change': float(healthy_pct_change) if not np.isnan(healthy_pct_change) else None,
        'unhealthy_pixel_count': int(unhealthy_pixel_count),
        'healthy_pixel_count': int(healthy_pixel_count),
        'pre_scan': patient['pre_filename'],
        'post_scan': post_filename
    }


def main():
    # Define paths
    base_dir = Path('/data/rydham')
    eval_dir = base_dir / 'Mask evaluations'
    pre_scans_dir = base_dir / 'pre_scans'
    post_scans_dir = base_dir / 'post_scans'
    masks_dir = base_dir / 'Masks'
    
    # Parse patient file
    patient_file = eval_dir / 'test_set_patients_moss.txt'
    patients = parse_patient_file(patient_file)
    
    print(f"Processing {len(patients)} test patients...")
    
    # Evaluate each patient
    results = []
    for i, patient in enumerate(patients):
        print(f"  [{i+1}/{len(patients)}] Evaluating patient {patient['year']}_{patient['id']}...")
        result = evaluate_patient(patient, str(pre_scans_dir), str(post_scans_dir), str(masks_dir))
        results.append(result)
    
    # Compute summary statistics by mask side
    summary = {}
    for side in ['left', 'right', 'both']:
        side_results = [r for r in results if r.get('mask_side') == side and 'error' not in r]
        if side_results:
            unhealthy_changes = [r['unhealthy_pct_change'] for r in side_results if r['unhealthy_pct_change'] is not None]
            healthy_changes = [r['healthy_pct_change'] for r in side_results if r['healthy_pct_change'] is not None]
            
            summary[side] = {
                'count': len(side_results),
                'unhealthy_mean': float(np.mean(unhealthy_changes)) if unhealthy_changes else None,
                'unhealthy_std': float(np.std(unhealthy_changes)) if unhealthy_changes else None,
                'healthy_mean': float(np.mean(healthy_changes)) if healthy_changes else None,
                'healthy_std': float(np.std(healthy_changes)) if healthy_changes else None
            }
    
    # Overall summary
    all_valid = [r for r in results if 'error' not in r]
    unhealthy_all = [r['unhealthy_pct_change'] for r in all_valid if r['unhealthy_pct_change'] is not None]
    healthy_all = [r['healthy_pct_change'] for r in all_valid if r['healthy_pct_change'] is not None]
    
    summary['overall'] = {
        'count': len(all_valid),
        'unhealthy_mean': float(np.mean(unhealthy_all)) if unhealthy_all else None,
        'unhealthy_std': float(np.std(unhealthy_all)) if unhealthy_all else None,
        'healthy_mean': float(np.mean(healthy_all)) if healthy_all else None,
        'healthy_std': float(np.std(healthy_all)) if healthy_all else None
    }
    
    # Save results
    output = {
        'description': 'Ground truth CBF percentage change: pre-scan vs post-scan',
        'patients': results,
        'summary': summary
    }
    
    output_path = eval_dir / 'ground_truth_evaluation.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("GROUND TRUTH EVALUATION SUMMARY (Pre vs Post Scan)")
    print("="*60)
    for side, stats in summary.items():
        print(f"\n{side.upper()} (n={stats['count']}):")
        if stats['unhealthy_mean'] is not None:
            print(f"  Unhealthy region: {stats['unhealthy_mean']:.2f}% ± {stats['unhealthy_std']:.2f}%")
        if stats['healthy_mean'] is not None:
            print(f"  Healthy region:   {stats['healthy_mean']:.2f}% ± {stats['healthy_std']:.2f}%")


if __name__ == '__main__':
    main()
