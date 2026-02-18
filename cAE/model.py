#!/usr/bin/env python3
import os, re, math, random, json, time
from glob import glob
from typing import List, Dict, Tuple

import numpy as np
import nibabel as nib
from sklearn.model_selection import KFold, train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from monai.networks.nets import UNet
from monai.losses import SSIMLoss
from monai.transforms import Compose, ScaleIntensity

from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Base paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)  # parent folder containing pre_scans / post_scans

PRE_DIR = os.path.join(ROOT_DIR, "pre_scans")     # e.g., ../pre_scans/pre_2021_001.nii.gz
POST_DIR = os.path.join(ROOT_DIR, "post_scans")   # e.g., ../post_scans/post_2021_001.nii.gz

OUT_DIR = os.path.join(BASE_DIR, "runs_axial_post_from_pre")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparam grid for CV
PARAM_GRID = [
    {"lr": 1e-3, "channels": (16, 32, 64, 128)},
    {"lr": 5e-4, "channels": (16, 32, 64, 128)},
    {"lr": 1e-3, "channels": (32, 64, 128, 256)},
]

BATCH_SIZE = 8
EPOCHS_CV = 10         # keep small for quick sweep
EPOCHS_FINAL = 50      # FINAL training budget (increased from 25)
NUM_FOLDS = 3
VAL_FRACTION = 0.25    # used only to make a hold-out test set

LOG_F = None  # will be opened in main()


def log(msg: str):
    """Print to stdout and also write to a log file."""
    print(msg)
    if LOG_F is not None:
        LOG_F.write(msg + "\n")
        LOG_F.flush()


# ----------------------------
# Utilities
# ----------------------------
ID_RE = re.compile(r"(pre)_(\d{4})_(\d{3})\.nii\.gz$", re.IGNORECASE)


def id_from_pre_path(p: str) -> Tuple[str, str]:
    """
    From pre_YYYY_NNN.nii.gz → ('YYYY', 'NNN')
    """
    m = ID_RE.search(os.path.basename(p))
    if not m:
        return None
    return m.group(2), m.group(3)


def pre_to_post(pre_path: str) -> str:
    """
    Map pre_YYYY_NNN.nii.gz → post_YYYY_NNN.nii.gz (same YYYY, NNN).
    """
    year_num = id_from_pre_path(pre_path)
    if year_num is None:
        return None
    year, num = year_num
    return os.path.join(POST_DIR, f"post_{year}_{num}.nii.gz")


def load_middle_axial_slice(nii_path: str) -> np.ndarray:
    """
    Load NIfTI, extract middle axial slice (3rd dim index = D//2), return float32 array.
    Assumes array shape (H, W, D) in axial orientation. If not, you may need reorientation.
    """
    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)

    if data.ndim != 3:
        # If 4D, take first channel/timepoint
        if data.ndim == 4:
            data = data[..., 0]
        else:
            raise ValueError(f"Unexpected shape {data.shape} in {nii_path}")

    H, W, D = data.shape
    mid = D // 2
    slc = data[:, :, mid]  # axial middle slice
    return slc


def minmax_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize an array to [0,1] via (x - min)/(max - min); if nearly constant, return all-zeros.
    """
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def ssim_np(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute SSIM between two 2D images (expects values ~[0,1]); 1.0 = identical.
    Uses target’s range for data_range.
    """
    dr = b.max() - b.min()
    if dr == 0:
        dr = 1.0
    return float(sk_ssim(a, b, data_range=dr))


def psnr_np(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute PSNR (dB) between prediction and target; higher is better.
    Uses target’s intensity range for data_range.
    """
    dr = b.max() - b.min()
    if dr == 0:
        dr = 1.0
    return float(sk_psnr(b, a, data_range=dr))


# ----------------------------
# Dataset
# ----------------------------
class AxialSlicePairs(Dataset):
    def __init__(self, pre_paths: List[str], transforms=None, target_size=(128, 128)):
        self.items: List[Tuple[str, str]] = []
        self.transforms = transforms
        self.target_size = target_size

        missing = 0
        for pre_p in pre_paths:
            post_p = pre_to_post(pre_p)
            if post_p is None or not os.path.exists(post_p):
                missing += 1
                continue
            self.items.append((pre_p, post_p))

        if len(self.items) == 0:
            raise RuntimeError(
                "No paired pre/post files found.\n"
                f"Checked {len(pre_paths)} pre files, missing matches for {missing}."
            )
        if missing > 0:
            print(f"[WARN] Skipped {missing} pre scans with no matching post.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pre_p, post_p = self.items[idx]

        # Load middle slices (H, W) numpy float32
        pre_slice = load_middle_axial_slice(pre_p)
        post_slice = load_middle_axial_slice(post_p)

        # Normalize to [0,1]
        pre_slice = minmax_norm(pre_slice)
        post_slice = minmax_norm(post_slice)

        # To torch and add channel & batch dims: (1,1,H,W)
        pre_t = torch.from_numpy(pre_slice).unsqueeze(0).unsqueeze(0).float()
        post_t = torch.from_numpy(post_slice).unsqueeze(0).unsqueeze(0).float()

        # Resize to fixed size with bilinear (channels stay = 1)
        pre_t = F.interpolate(pre_t, size=self.target_size, mode="bilinear", align_corners=False)
        post_t = F.interpolate(post_t, size=self.target_size, mode="bilinear", align_corners=False)

        # Drop the temporary batch dim: (1,H,W)
        pre_t = pre_t.squeeze(0)
        post_t = post_t.squeeze(0)

        # Optional extra transforms that expect (C,H,W) tensors
        if self.transforms is not None:
            pre_t = self.transforms(pre_t)
            post_t = self.transforms(post_t)

        return pre_t, post_t


# ----------------------------
# Model / Train / Eval
# ----------------------------
def make_unet_2d(ch=(16, 32, 64, 128)) -> nn.Module:
    # Simple 2D UNet for regression (1→1)
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


def train_one_epoch(model, loader, criterion_l1, criterion_ssim, optimizer):
    model.train()
    total_l1, total_ssim = 0.0, 0.0
    n = 0
    for pre_t, post_t in loader:
        pre_t = pre_t.to(DEVICE)
        post_t = post_t.to(DEVICE)

        optimizer.zero_grad()
        out = model(pre_t)

        # Combined loss: L1 + (1 - SSIM)
        l1 = criterion_l1(out, post_t)
        ssim_loss = criterion_ssim(out, post_t)  # smaller is better (1-SSIM internally)
        loss = l1 + ssim_loss

        loss.backward()
        optimizer.step()

        total_l1 += l1.item() * pre_t.size(0)
        total_ssim += (1.0 - ssim_loss.item()) * pre_t.size(0)  # convert to SSIM-like
        n += pre_t.size(0)

    return total_l1 / n, total_ssim / n


@torch.no_grad()
def evaluate(model, loader, criterion_l1, criterion_ssim):
    model.eval()
    total_l1, total_ssim = 0.0, 0.0
    total_mae = 0.0
    total_psnr, total_ssim_np = 0.0, 0.0
    n = 0

    for pre_t, post_t in loader:
        pre_t = pre_t.to(DEVICE)
        post_t = post_t.to(DEVICE)
        out = model(pre_t)

        l1 = criterion_l1(out, post_t)
        ssim_loss = criterion_ssim(out, post_t)
        mae = torch.mean(torch.abs(out - post_t))

        total_l1 += l1.item() * pre_t.size(0)
        total_ssim += (1.0 - ssim_loss.item()) * pre_t.size(0)
        total_mae += mae.item() * pre_t.size(0)

        # Compute PSNR/SSIM from numpy (on first channel)
        out_np = out.detach().cpu().numpy()[:, 0]
        tgt_np = post_t.detach().cpu().numpy()[:, 0]
        for i in range(out_np.shape[0]):
            total_psnr += psnr_np(out_np[i], tgt_np[i])
            total_ssim_np += ssim_np(out_np[i], tgt_np[i])

        n += pre_t.size(0)

    metrics = {
        "L1": total_l1 / n,
        "MAE": total_mae / n,
        "SSIM_torch": total_ssim / n,
        "SSIM_np": total_ssim_np / n,
        "PSNR": total_psnr / n,
    }
    return metrics


def run_one_training(
    train_idx, val_idx, pairs, params, epochs=EPOCHS_CV, run_name="cv_run"
):
    # Build datasets/loaders from indices
    train_pre = [pairs[i] for i in train_idx]
    val_pre = [pairs[i] for i in val_idx]

    train_ds = AxialSlicePairs(train_pre, transforms=None)
    val_ds = AxialSlicePairs(val_pre, transforms=None)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = make_unet_2d(ch=params["channels"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIMLoss(spatial_dims=2)

    history = {"train_L1": [], "train_SSIM": [], "val_L1": [], "val_MAE": [], "val_SSIM_np": [], "val_PSNR": []}

    best_ssim = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        tr_l1, tr_ssim = train_one_epoch(model, train_loader, criterion_l1, criterion_ssim, optimizer)
        val_metrics = evaluate(model, val_loader, criterion_l1, criterion_ssim)

        history["train_L1"].append(tr_l1)
        history["train_SSIM"].append(tr_ssim)
        history["val_L1"].append(val_metrics["L1"])
        history["val_MAE"].append(val_metrics["MAE"])
        history["val_SSIM_np"].append(val_metrics["SSIM_np"])
        history["val_PSNR"].append(val_metrics["PSNR"])

        log(f"[{run_name}] Epoch {ep:02d}/{epochs} | "
            f"Train L1 {tr_l1:.4f}, SSIM {tr_ssim:.4f} | "
            f"Val L1 {val_metrics['L1']:.4f}, MAE {val_metrics['MAE']:.4f}, "
            f"SSIM {val_metrics['SSIM_np']:.4f}, PSNR {val_metrics['PSNR']:.2f}")

        # Track best by SSIM
        if val_metrics["SSIM_np"] > best_ssim:
            best_ssim = val_metrics["SSIM_np"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_ssim, best_state, history


def plot_curves(history: Dict[str, List[float]], out_dir: str):
    """Save training/validation curves as PNGs."""
    epochs = np.arange(1, len(history["train_L1"]) + 1)

    # L1
    plt.figure()
    plt.plot(epochs, history["train_L1"], label="Train L1")
    plt.plot(epochs, history["val_L1"], label="Val L1")
    plt.xlabel("Epoch")
    plt.ylabel("L1 loss")
    plt.legend()
    plt.title("Train vs Val L1")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_L1.png"), dpi=200)
    plt.close()

    # SSIM
    plt.figure()
    plt.plot(epochs, history["train_SSIM"], label="Train SSIM")
    plt.plot(epochs, history["val_SSIM_np"], label="Val SSIM (np)")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.legend()
    plt.title("Train vs Val SSIM")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_SSIM.png"), dpi=200)
    plt.close()

    # MAE & PSNR (val)
    plt.figure()
    plt.plot(epochs, history["val_MAE"], label="Val MAE")
    plt.plot(epochs, history["val_PSNR"], label="Val PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.title("Val MAE & PSNR")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_MAE_PSNR.png"), dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    global LOG_F
    # open log file inside OUT_DIR
    LOG_F = open(os.path.join(OUT_DIR, "training_log.txt"), "w")

    all_pre = sorted(glob(os.path.join(PRE_DIR, "pre_*.nii.gz")))
    if len(all_pre) == 0:
        log(f"No files found in {PRE_DIR}")
        return

    # Filter to those that *can* be paired (so CV sees only learnable pairs)
    paired_pre = []
    for p in all_pre:
        q = pre_to_post(p)
        if q and os.path.exists(q):
            paired_pre.append(p)

    if len(paired_pre) == 0:
        log(
            "Found no matched pairs. Please add post scans to "
            f"'{POST_DIR}' with names like 'post_YYYY_NNN.nii.gz'."
        )
        LOG_F.close()
        return

    # Split once into Train+Val pool vs Test (held-out)
    trainval_pre, test_pre = train_test_split(
        paired_pre, test_size=VAL_FRACTION, random_state=SEED, shuffle=True
    )

    log(f"Total paired: {len(paired_pre)} | Train+Val: {len(trainval_pre)} | Test: {len(test_pre)}")

    # K-Fold CV over trainval to pick best hyperparams
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    best_overall = None  # (mean_ssim, params)
    for grid_i, params in enumerate(PARAM_GRID):
        fold_ssims = []
        log(f"\n=== Hyperparam set {grid_i+1}/{len(PARAM_GRID)}: {params} ===")
        for fold, (tr_idx, va_idx) in enumerate(kf.split(trainval_pre), 1):
            mean_ssim_fold, _, _ = run_one_training(
                tr_idx, va_idx, trainval_pre, params,
                epochs=EPOCHS_CV, run_name=f"cv_{grid_i+1}_fold{fold}"
            )
            fold_ssims.append(mean_ssim_fold)

        mean_ssim = float(np.mean(fold_ssims))
        log(f"[CV] Params {params} → mean SSIM {mean_ssim:.4f}")

        if best_overall is None or mean_ssim > best_overall[0]:
            best_overall = (mean_ssim, params)

    log(f"\n>>> Best CV params: {best_overall[1]} with mean SSIM {best_overall[0]:.4f}")

    # Final train on full Train+Val with best params
    best_params = best_overall[1]
    trainval_ds = AxialSlicePairs(trainval_pre, transforms=None)
    trainval_loader = DataLoader(trainval_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    model = make_unet_2d(ch=best_params["channels"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIMLoss(spatial_dims=2)

    best_val = -1.0
    # Small internal val split from trainval for monitoring (no test leakage)
    tv_indices = np.arange(len(trainval_ds))
    tv_tr_idx, tv_va_idx = train_test_split(tv_indices, test_size=0.12, random_state=SEED, shuffle=True)
    tv_tr_subset = torch.utils.data.Subset(trainval_ds, tv_tr_idx)
    tv_va_subset = torch.utils.data.Subset(trainval_ds, tv_va_idx)

    tv_tr_loader = DataLoader(tv_tr_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    tv_va_loader = DataLoader(tv_va_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    final_hist = {"train_L1": [], "train_SSIM": [], "val_L1": [], "val_MAE": [], "val_SSIM_np": [], "val_PSNR": []}
    best_state = None

    log("\n=== Final training with best params ===")
    for ep in range(1, EPOCHS_FINAL + 1):
        tr_l1, tr_ssim = train_one_epoch(model, tv_tr_loader, criterion_l1, criterion_ssim, optimizer)
        val_metrics = evaluate(model, tv_va_loader, criterion_l1, criterion_ssim)

        final_hist["train_L1"].append(tr_l1)
        final_hist["train_SSIM"].append(tr_ssim)
        final_hist["val_L1"].append(val_metrics["L1"])
        final_hist["val_MAE"].append(val_metrics["MAE"])
        final_hist["val_SSIM_np"].append(val_metrics["SSIM_np"])
        final_hist["val_PSNR"].append(val_metrics["PSNR"])

        log(f"[FINAL] Epoch {ep:02d}/{EPOCHS_FINAL} | "
            f"Train L1 {tr_l1:.4f}, SSIM {tr_ssim:.4f} | "
            f"Val L1 {val_metrics['L1']:.4f}, MAE {val_metrics['MAE']:.4f}, "
            f"SSIM {val_metrics['SSIM_np']:.4f}, PSNR {val_metrics['PSNR']:.2f}")

        if val_metrics["SSIM_np"] > best_val:
            best_val = val_metrics["SSIM_np"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save best model + training curves
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_unet_axial_postfrompre.pt"))
    with open(os.path.join(OUT_DIR, "final_history.json"), "w") as f:
        json.dump(final_hist, f, indent=2)

    # Plot curves for the final training
    plot_curves(final_hist, OUT_DIR)

    # Evaluate on hold-out TEST set
    log("\n=== Test evaluation (hold-out set) ===")
    test_ds = AxialSlicePairs(test_pre, transforms=None)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_metrics = evaluate(model, test_loader, criterion_l1, criterion_ssim)
    log(f"[TEST] L1 {test_metrics['L1']:.4f}, MAE {test_metrics['MAE']:.4f}, "
        f"SSIM {test_metrics['SSIM_np']:.4f}, PSNR {test_metrics['PSNR']:.2f}")

    # Save ALL test predictions for cohort analysis
    os.makedirs(os.path.join(OUT_DIR, "pred_samples"), exist_ok=True)
    model.eval()
    with torch.no_grad():
        count = 0
        for pre_t, post_t in test_loader:
            pre_t = pre_t.to(DEVICE)
            out = model(pre_t).cpu().numpy()
            tgt = post_t.numpy()
            inp = pre_t.cpu().numpy()
            B = out.shape[0]
            for i in range(B):
                np.save(os.path.join(OUT_DIR, "pred_samples", f"in_{count:04d}.npy"), inp[i, 0])
                np.save(os.path.join(OUT_DIR, "pred_samples", f"out_{count:04d}.npy"), out[i, 0])
                np.save(os.path.join(OUT_DIR, "pred_samples", f"tgt_{count:04d}.npy"), tgt[i, 0])
                count += 1
    log(f"Saved {count} test predictions to pred_samples/")

    log(f"\nDone. Best params: {best_params}. Test metrics: {test_metrics}")
    LOG_F.close()


if __name__ == "__main__":
    main()
