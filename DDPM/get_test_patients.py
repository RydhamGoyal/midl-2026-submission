#!/usr/bin/env python3
"""
Script to identify which patients are in the test set for DDPM.
Reproduces the exact same train/test split used in diffusion_model.py.
"""

import os
import re
from glob import glob
from sklearn.model_selection import train_test_split

SEED = 1337
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

PRE_DIR = os.path.join(ROOT_DIR, "pre_scans")
POST_DIR = os.path.join(ROOT_DIR, "post_scans")

# Regex to extract patient info from filename
ID_RE = re.compile(r"pre_(\d{4})_(\d{3})\.nii\.gz$", re.IGNORECASE)


def id_from_pre_path(p: str):
    m = ID_RE.search(os.path.basename(p))
    if not m:
        return None
    return m.group(1), m.group(2)  # (year, patient_id)


def pre_to_post(pre_path: str):
    ids = id_from_pre_path(pre_path)
    if not ids:
        return None
    y, n = ids
    return os.path.join(POST_DIR, f"post_{y}_{n}.nii.gz")


def main():
    # Gather paired files (same logic as diffusion_model.py)
    all_pre = sorted(glob(os.path.join(PRE_DIR, "pre_*.nii.gz")))
    paired = []
    for p in all_pre:
        q = pre_to_post(p)
        if q and os.path.exists(q):
            paired.append((p, q))

    print(f"Total paired scans: {len(paired)}")

    # Same split as diffusion_model.py (25% test)
    train_pairs, test_pairs = train_test_split(
        paired, test_size=0.25, random_state=SEED, shuffle=True
    )

    print(f"\nTrain+Val: {len(train_pairs)} | Test: {len(test_pairs)}")
    print("\n" + "=" * 60)
    print("TEST SET PATIENTS (DDPM)")
    print("=" * 60)

    test_patients = []
    for pre_path, post_path in sorted(test_pairs, key=lambda x: x[0]):
        ids = id_from_pre_path(pre_path)
        if ids:
            year, patient_id = ids
            test_patients.append((year, patient_id, os.path.basename(pre_path)))
            print(f"  Year: {year}, Patient ID: {patient_id} -> {os.path.basename(pre_path)}")

    print("\n" + "=" * 60)
    print(f"Total test patients: {len(test_patients)}")
    print("=" * 60)

    # Save to file
    output_file = os.path.join(BASE_DIR, "test_set_patients.txt")
    with open(output_file, "w") as f:
        f.write("TEST SET PATIENTS (DDPM)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total: {len(test_patients)} patients\n\n")
        f.write("Year\tPatient_ID\tFilename\n")
        f.write("-" * 60 + "\n")
        for year, patient_id, filename in sorted(test_patients):
            f.write(f"{year}\t{patient_id}\t\t{filename}\n")

    print(f"\n✅ Saved to: {output_file}")


if __name__ == "__main__":
    main()
