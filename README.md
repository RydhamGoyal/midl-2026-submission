# Generating Post-Acetazolamide Cerebral Blood Flow MRI for High-risk Stroke Patients

**Accepted to [MIDL 2026](https://2026.midl.io) (Medical Imaging with Deep Learning) — Taipei, Taiwan | July 8–10, 2026**

> **Rydham Goyal**\*, Camila Gonzalez, Sasha Alexander, Aja Zou, Michael E. Moseley, Moss Y. Zhao†, Gary K. Steinberg‡  
> Stanford University  
> \*First author, †Co-senior author, ‡Senior author

**[Paper Link](https://openreview.net/forum?id=WMBUxtRdxB#discussion)
---

## Abstract

Cerebrovascular reserve (CVR) quantifies the brain's ability to augment cerebral blood flow in response to a vasodilatory stimulus and is a key biomarker in Moyamoya disease and other steno-occlusive cerebrovascular disorders. Clinically, CVR is typically assessed by administering acetazolamide (ACZ) and acquiring post-ACZ perfusion maps, but this workflow is time-consuming, costly, and contraindicated in a subset of patients. In this work, we investigate whether deep learning can predict post-ACZ perfusion directly from baseline arterial spin labeling (ASL) MRI, enabling pharmacological-free CVR estimation.

We curate a single-center dataset of Moyamoya ASL perfusion imaging, comprising pre/post-ACZ scan pairs from 194 patients. We design a **post-ACZ conditional Autoencoder (cAE)** to regress the middle axial post-ACZ slice from the corresponding pre-ACZ slice using a combined L1 and SSIM loss. We evaluate our method against three diffusion-based formulations (**conditional DDPM**, **Cold Diffusion**, and **Residual Diffusion**). On a hold-out test set of 49 patients, the proposed post-ACZ cAE achieves the highest reconstruction fidelity (SSIM ≈ 0.79), outperforming diffusion-based baselines in MAE, SSIM, and PSNR. Region-wise analysis of CBF percentage change in affected versus healthy MCA territories showed that the generated post-ACZ model outputs followed ground truth patterns of cerebrovascular reserve.

**Keywords:** Generative models, Conditional Autoencoder, Pixel-space diffusion, Image-to-image translation, Arterial Spin Labeling, MRI, Cerebrovascular reserve

---

## Models

| Model | Architecture | Description |
|-------|-------------|-------------|
| **cAE** | post-ACZ conditional Autoencoder | Direct regression via MONAI 2D UNet. Maps pre-ACZ → post-ACZ with L1 + (1−SSIM) loss. Our primary proposed model. |
| **DDPM** | Conditional Denoising Diffusion Probabilistic Model | Gaussian noise-based diffusion using MONAI Generative's `DiffusionModelUNet`. Learns to denoise post-ACZ slices conditioned on pre-ACZ input (T=200 steps). |
| **ColdDiffusion** | Cold Diffusion | Deterministic degradation via linear interpolation between post-ACZ and pre-ACZ (T=10 steps). Based on [Bansal et al., NeurIPS 2023](https://arxiv.org/abs/2208.09392). |
| **ResidualDiffusion** | Residual Diffusion | Applies DDPM-style diffusion to the *residual* (post−pre) rather than the full image (T=100 steps). Isolates the treatment effect for targeted generation. |

## Results

| Model | MAE ↓ | SSIM ↑ | PSNR (dB) ↑ |
|-------|-------|--------|-------------|
| **post-ACZ cAE** | **0.0497 ± 0.0176** | **0.7886 ± 0.1135** | **21.49 ± 2.70** |
| Cold Diffusion | 0.0660 ± 0.0260 | 0.7195 ± 0.0920 | 18.66 ± 2.39 |
| DDPM | 0.0841 ± 0.0243 | 0.4486 ± 0.0800 | 17.33 ± 2.04 |
| Residual Diffusion | 0.1976 ± 0.0094 | 0.0863 ± 0.0215 | 11.25 ± 0.38 |

*Evaluated on a held-out test set of 49 patients. Values reported as mean ± standard deviation.*

**Key findings:**
- The deterministic **cAE** outperforms all stochastic diffusion-based formulations, suggesting that the pre-to-post ACZ mapping in Moyamoya is sufficiently constrained for direct conditional regression
- **Cold Diffusion** is the strongest diffusion-based alternative, achieving competitive SSIM
- MCA territory analyses confirm the **cAE** recovers the clinically expected ordering of responses — larger CBF augmentation in healthy vs. diseased territories
- The cAE also captures "steal"-like phenomena (paradoxical CBF reductions in high-risk regions), which is clinically meaningful for identifying at-risk patients

---

## Project Structure

```
├── cAE/                              # post-ACZ conditional Autoencoder (proposed model)
│   ├── model.py                      # Training & evaluation
│   ├── differences.py                # Pre vs Post vs Prediction visualization
│   ├── get_test_patients.py          # Reproduce train/test split
│   └── Mask evaluations/
│       └── evaluate_model.py         # MCA territory CBF evaluation
│
├── DDPM/                             # Conditional DDPM
│   ├── diffusion_model.py            # Training & evaluation
│   ├── differences.py                # Visualization panels
│   ├── get_test_patients.py          # Reproduce train/test split
│   └── Mask evaluations/
│       └── evaluate_model.py         # MCA territory CBF evaluation
│
├── ColdDiffusion/                    # Cold Diffusion
│   ├── model.py                      # Training & evaluation
│   ├── differences.py                # Visualization panels
│   ├── get_test_patients.py          # Reproduce train/test split
│   └── Mask evaluations/
│       └── evaluate_model.py         # MCA territory CBF evaluation
│
├── ResidualDiffusion/                # Residual Diffusion
│   ├── model.py                      # Training & evaluation
│   ├── differences.py                # Visualization panels
│   ├── get_test_patients.py          # Reproduce train/test split
│   └── Mask evaluations/
│       └── evaluate_model.py         # MCA territory CBF evaluation
│
├── Mask evaluations/                 # Cross-model evaluation scripts
│   ├── evaluate_ground_truth.py      # Ground truth CBF % change computation
│   ├── fix_mask_overlay.py           # MCA mask overlay visualization
│   ├── mask_evaluation_report.txt    # Comprehensive evaluation report
│   └── Cohort Summary/
│       └── analyze_all_models.py     # Aggregate cross-model comparison
│
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** Patient scan data, vascular territory masks, trained model weights, and prediction outputs are excluded from this repository for patient privacy and file size reasons.

---

## Setup

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)

### Installation

```bash
git clone https://github.com/RydhamGoyal/midl-2026-submission.git
cd midl-2026-submission
pip install -r requirements.txt
```

### Data Preparation

This study uses paired pre/post-ACZ ASL CBF maps from 194 patients with cerebrovascular disease (Moyamoya). Place your paired NIfTI brain scans in the following structure:

```
pre_scans/
  pre_YYYY_NNN.nii.gz     # Baseline (pre-ACZ) CBF maps
post_scans/
  post_YYYY_NNN.nii.gz    # Post-acetazolamide CBF maps (matched by YYYY_NNN)
Masks/
  MNI_left_MCA_2mm.nii.gz   # Left MCA vascular territory mask (2mm MNI atlas)
  MNI_right_MCA_2mm.nii.gz  # Right MCA vascular territory mask (2mm MNI atlas)
```

Each model extracts the **middle axial slice** from each 3D volume, resizes to **128×128**, and applies min–max normalization to [0, 1].

---

## Usage

### Training

Each model is self-contained. To train, run the main script:

```bash
# post-ACZ conditional Autoencoder (proposed model)
python cAE/model.py

# Conditional DDPM
python DDPM/diffusion_model.py

# Cold Diffusion
python ColdDiffusion/model.py

# Residual Diffusion
python ResidualDiffusion/model.py
```

Each script will:
1. Load and pair pre/post-ACZ scans
2. Split into train (63%), validation (12%), and test (25%) sets with seed=1337
3. Train the model with validation monitoring
4. Save the best checkpoint, training curves, and training history
5. Evaluate on the held-out test set and save predictions

### MCA Territory Evaluation

After training, run the mask-based CBF percentage change evaluation:

```bash
python cAE/Mask\ evaluations/evaluate_model.py
python DDPM/Mask\ evaluations/evaluate_model.py
python ColdDiffusion/Mask\ evaluations/evaluate_model.py
python ResidualDiffusion/Mask\ evaluations/evaluate_model.py
```

For cross-model cohort comparison:

```bash
python Mask\ evaluations/Cohort\ Summary/analyze_all_models.py
```

### Visualization

Generate difference panels (pre-ACZ vs post-ACZ vs prediction):

```bash
python cAE/differences.py
python DDPM/differences.py
python ColdDiffusion/differences.py
python ResidualDiffusion/differences.py
```

---

## Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{goyal2026generating,
  title={Generating Post-Acetazolamide Cerebral Blood Flow MRI for High-risk Stroke Patients},
  author={Goyal, Rydham and Gonzalez, Camila and Alexander, Sasha and Zou, Aja and Moseley, Michael E and Zhao, Moss Y and Steinberg, Gary K},
  booktitle={Medical Imaging with Deep Learning (MIDL)},
  year={2026}
}
```

## License

© 2025 CC-BY 4.0, R. Goyal, C. Gonzalez, S. Alexander, A. Zou, M.E. Moseley, M.Y. Zhao & G.K. Steinberg.
