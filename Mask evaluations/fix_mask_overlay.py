#!/usr/bin/env python3
"""
Fix the MCA mask overlay visualization:
1. Rotate brain 180 degrees
2. Swap left/right labels (anatomical right appears on image left after rotation)
3. Reduce whitespace
4. Update title
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom

# Paths
base_dir = Path("/data/rydham")
masks_dir = base_dir / "Masks"
output_path = base_dir / "Mask evaluations" / "masks_on_cbf_overlay.png"

# Load example CBF scan (use a pre-scan from UNet predictions)
pred_dir = base_dir / "cAE" / "runs_axial_post_from_pre" / "pred_samples"
pre_files = sorted(pred_dir.glob("in_*.npy"))
if not pre_files:
    raise RuntimeError(f"No prediction files found in {pred_dir}")

# Load the first example (this is already a 2D slice, normalized to [0,1])
cbf_slice = np.load(str(pre_files[0]))
print(f"CBF slice shape: {cbf_slice.shape}")

# Load MCA masks
left_mask = nib.load(str(masks_dir / "MNI_left_MCA_2mm.nii.gz")).get_fdata()
right_mask = nib.load(str(masks_dir / "MNI_right_MCA_2mm.nii.gz")).get_fdata()
print(f"Mask shape: {left_mask.shape}")

# Get middle axial slice from 3D masks
middle_slice_idx = left_mask.shape[2] // 2
left_mask_slice = left_mask[:, :, middle_slice_idx]
right_mask_slice = right_mask[:, :, middle_slice_idx]

# Resize masks to match CBF dimensions (128x128)
target_shape = cbf_slice.shape
zoom_factors = (target_shape[0] / left_mask_slice.shape[0], 
                target_shape[1] / left_mask_slice.shape[1])
left_mask_resized = zoom(left_mask_slice, zoom_factors, order=0)  # order=0 for nearest neighbor
right_mask_resized = zoom(right_mask_slice, zoom_factors, order=0)
print(f"Resized mask shape: {left_mask_resized.shape}")

# Rotate all images 180 degrees (k=2 means 180 degrees)
cbf_slice = np.rot90(cbf_slice, k=2)
left_mask_resized = np.rot90(left_mask_resized, k=2)
right_mask_resized = np.rot90(right_mask_resized, k=2)

# Create figure with reduced whitespace
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Adjust spacing
plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.05, wspace=0.08)

# CBF base image (already normalized to [0, 1])
vmin, vmax = 0, 1

# Panel 1: Left MCA mask (anatomically left, appears on right side of image after 180° rotation)
axes[0].imshow(cbf_slice, cmap='gray', vmin=vmin, vmax=vmax)
axes[0].imshow(np.ma.masked_where(left_mask_resized == 0, left_mask_resized), 
               cmap='Reds', alpha=0.5, vmin=0, vmax=1)
axes[0].set_title('Left MCA Mask', fontsize=14, fontweight='bold', pad=8)
axes[0].axis('off')

# Panel 2: Right MCA mask (anatomically right, appears on left side of image after 180° rotation)
axes[1].imshow(cbf_slice, cmap='gray', vmin=vmin, vmax=vmax)
axes[1].imshow(np.ma.masked_where(right_mask_resized == 0, right_mask_resized), 
               cmap='Blues', alpha=0.5, vmin=0, vmax=1)
axes[1].set_title('Right MCA Mask', fontsize=14, fontweight='bold', pad=8)
axes[1].axis('off')

# Panel 3: Combined masks
axes[2].imshow(cbf_slice, cmap='gray', vmin=vmin, vmax=vmax)
axes[2].imshow(np.ma.masked_where(left_mask_resized == 0, left_mask_resized), 
               cmap='Reds', alpha=0.5, vmin=0, vmax=1)
axes[2].imshow(np.ma.masked_where(right_mask_resized == 0, right_mask_resized), 
               cmap='Blues', alpha=0.5, vmin=0, vmax=1)
axes[2].set_title('Both MCA Masks', fontsize=14, fontweight='bold', pad=8)
axes[2].axis('off')

# Overall title
fig.suptitle('MCA Mask Regions on CBF', fontsize=16, fontweight='bold', y=0.98)

plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
plt.close()

print(f"✅ Fixed overlay saved to: {output_path}")
print("   - Brain rotated 180°")
print("   - Left/Right labels corrected for radiological viewing")
print("   - Whitespace reduced")
print(f"   - Masks resized to match CBF dimensions {cbf_slice.shape}")
