#!/usr/bin/env python3
"""
Cold Diffusion Analytics: Generates 2x3 difference panels and metrics CSV.

Creates visualizations comparing:
- Pre (input), Prediction, Post (target) in top row
- |Pre-Post|, |Pre-Pred|, |Pred-Post| difference heatmaps in bottom row

Also computes SSIM, PSNR, MAE for each sample.
"""

import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(BASE_DIR, "runs_cold_diffusion", "pred_samples")
OUT_DIR = os.path.join(BASE_DIR, "runs_cold_diffusion", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Metrics
# ============================================================
def ssim_np(a, b):
    dr = b.max() - b.min()
    if dr == 0:
        dr = 1.0
    return float(sk_ssim(a, b, data_range=dr))


def psnr_np(a, b):
    dr = b.max() - b.min()
    if dr == 0:
        dr = 1.0
    return float(sk_psnr(b, a, data_range=dr))


def mae_np(a, b):
    return float(np.mean(np.abs(a - b)))


# ============================================================
# File Discovery
# ============================================================
def find_indices(pred_dir):
    outs = sorted(glob.glob(os.path.join(pred_dir, "out_*.npy")))
    idxs = [int(os.path.splitext(os.path.basename(p))[0].split("_")[1]) for p in outs]
    return sorted(idxs)


# ============================================================
# Panel Creation (2 rows, 3 columns)
# ============================================================
def panel_2x3(inp, pred, tgt, diff1, diff2, diff3, mae1, mae2, mae3, ssim, psnr, fname):
    """
    Create a 2x3 visualization panel.
    
    Top row: Pre / Prediction / Post (grayscale)
    Bottom row: |Pre-Post| / |Pre-Pred| / |Pred-Post| (heatmaps)
    """
    # Rotate all images 90° counter-clockwise for radiology standard
    inp = np.rot90(inp, k=1)
    pred = np.rot90(pred, k=1)
    tgt = np.rot90(tgt, k=1)
    diff1 = np.rot90(diff1, k=1)
    diff2 = np.rot90(diff2, k=1)
    diff3 = np.rot90(diff3, k=1)

    # Shared vmax for all three heatmaps
    vmax = max(diff1.max(), diff2.max(), diff3.max())
    if vmax == 0:
        vmax = 1.0

    # Larger figure, reduced spacing
    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05], 
                           wspace=0.08, hspace=0.15,
                           left=0.02, right=0.92, top=0.88, bottom=0.05)

    # --- Top row (grayscale) ---
    # Use consistent intensity scaling (vmin=0, vmax=1) so predictions don't look faded
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(inp, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("Input (Pre)", fontsize=14, fontweight='bold', pad=8)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(pred, cmap="gray", vmin=0, vmax=1)
    ax2.set_title(f"Prediction\nSSIM={ssim:.3f}, PSNR={psnr:.1f}dB", fontsize=12, fontweight='bold', pad=6)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(tgt, cmap="gray", vmin=0, vmax=1)
    ax3.set_title("Target (Post)", fontsize=14, fontweight='bold', pad=8)
    ax3.axis("off")

    # --- Bottom row (heatmaps) ---
    ax4 = fig.add_subplot(gs[1, 0])
    im = ax4.imshow(diff1, cmap="magma", vmin=0, vmax=vmax)
    ax4.set_title(f"|Pre–Post|\nMAE={mae1:.4f}", fontsize=12, pad=6)
    ax4.axis("off")

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(diff2, cmap="magma", vmin=0, vmax=vmax)
    ax5.set_title(f"|Pre–Pred|\nMAE={mae2:.4f}", fontsize=12, pad=6)
    ax5.axis("off")

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(diff3, cmap="magma", vmin=0, vmax=vmax)
    ax6.set_title(f"|Pred–Post|\nMAE={mae3:.4f}", fontsize=12, pad=6)
    ax6.axis("off")

    # Shared colorbar
    cax = fig.add_subplot(gs[:, 3])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Absolute Difference", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.suptitle("Cold Diffusion Analysis", fontsize=14, fontweight="bold")
    plt.savefig(fname, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    if not os.path.exists(PRED_DIR):
        raise RuntimeError(
            f"❌ Prediction directory not found: {PRED_DIR}\n"
            "Please run model.py first to generate predictions."
        )

    idxs = find_indices(PRED_DIR)
    if not idxs:
        raise RuntimeError(f"❌ No prediction files found in {PRED_DIR}")

    csv_rows = []
    ssims, psnrs, maes = [], [], []

    print(f"Found {len(idxs)} samples to analyze...")

    for i in idxs:
        paths = {
            "inp": os.path.join(PRED_DIR, f"in_{i:04d}.npy"),
            "pred": os.path.join(PRED_DIR, f"out_{i:04d}.npy"),
            "tgt": os.path.join(PRED_DIR, f"tgt_{i:04d}.npy"),
        }

        if not all(os.path.exists(p) for p in paths.values()):
            print(f"[WARN] Missing files for {i:04d}, skipping.")
            continue

        inp = np.load(paths["inp"])
        pred = np.load(paths["pred"])
        tgt = np.load(paths["tgt"])

        # Metrics for pred vs tgt
        ssim = ssim_np(pred, tgt)
        psnr = psnr_np(pred, tgt)
        mae = mae_np(pred, tgt)

        ssims.append(ssim)
        psnrs.append(psnr)
        maes.append(mae)

        csv_rows.append({
            "index": i,
            "SSIM": ssim,
            "PSNR_dB": psnr,
            "MAE": mae,
            "MAE_pre_post": mae_np(inp, tgt),
            "MAE_pre_pred": mae_np(inp, pred),
        })

        # Difference images
        diff_pre_post = np.abs(inp - tgt)
        diff_pre_pred = np.abs(inp - pred)
        diff_pred_post = np.abs(pred - tgt)

        # MAEs for titles
        mae1 = mae_np(inp, tgt)
        mae2 = mae_np(inp, pred)
        mae3 = mae_np(pred, tgt)

        out_path = os.path.join(OUT_DIR, f"sample_{i:04d}.png")
        panel_2x3(
            inp, pred, tgt,
            diff_pre_post, diff_pre_pred, diff_pred_post,
            mae1, mae2, mae3,
            ssim, psnr,
            out_path
        )

    # Save metrics CSV
    csv_path = os.path.join(OUT_DIR, "test_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "SSIM", "PSNR_dB", "MAE", "MAE_pre_post", "MAE_pre_pred"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Print summary
    print("\n" + "=" * 50)
    print("Cold Diffusion Test Set Summary")
    print("=" * 50)
    print(f"Samples analyzed: {len(ssims)}")
    print(f"Mean SSIM : {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
    print(f"Mean PSNR : {np.mean(psnrs):.2f} dB ± {np.std(psnrs):.2f}")
    print(f"Mean MAE  : {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print("=" * 50)
    print(f"\n✅ Saved panels to: {OUT_DIR}")
    print(f"✅ Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
