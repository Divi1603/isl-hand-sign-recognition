"""
STEP 1 - LANDMARK EXTRACTION
Run this first to convert your image dataset into MediaPipe hand landmarks.
Usage: python 1_extract_landmarks.py
"""

import os
import csv
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3   # lower = catches harder images
)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR = "dataset"            # folder with sub-folders per gesture label
OUTPUT_CSV  = "landmarks.csv"

# 21 landmarks × (x, y, z) = 63 raw features
# + 20 pairwise distances between consecutive joints = 83 total features
HEADER = (
    [f"x{i}" for i in range(21)] +
    [f"y{i}" for i in range(21)] +
    [f"z{i}" for i in range(21)] +
    [f"dist_{i}_{i+1}" for i in range(20)] +
    ["label"]
)

# ── Helpers ───────────────────────────────────────────────────────────────────
CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)   # 20 pairs

def normalize_landmarks(lm_list):
    """Translate so wrist=origin, scale by max distance from wrist."""
    arr = np.array([[l.x, l.y, l.z] for l in lm_list])
    wrist = arr[0]
    arr -= wrist
    scale = np.max(np.linalg.norm(arr, axis=1)) + 1e-6
    arr /= scale
    return arr

def pairwise_distances(arr):
    """20 distances along MediaPipe hand connections."""
    dists = []
    for a, b in CONNECTIONS:
        dists.append(float(np.linalg.norm(arr[a] - arr[b])))
    return dists

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Try original first, then augmented versions if no hand found
    for attempt in range(4):
        if attempt == 1:
            img_rgb = cv2.flip(img_rgb, 1)
        elif attempt == 2:
            img_rgb = cv2.convertScaleAbs(img_rgb, alpha=1.3, beta=20)
        elif attempt == 3:
            img_rgb = cv2.GaussianBlur(img_rgb, (3, 3), 0)

        result = hands.process(img_rgb)
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark
            arr = normalize_landmarks(lm)
            xs    = arr[:, 0].tolist()
            ys    = arr[:, 1].tolist()
            zs    = arr[:, 2].tolist()
            dists = pairwise_distances(arr)
            return xs + ys + zs + dists

    return None   # no hand detected in any attempt

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    labels = sorted(os.listdir(DATASET_DIR))
    print(f"Found {len(labels)} gesture classes: {labels}\n")

    rows = []
    skipped = 0

    for label in labels:
        folder = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(folder):
            continue
        images = [f for f in os.listdir(folder)
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        print(f"Processing '{label}' — {len(images)} images")

        for fname in tqdm(images, desc=label, ncols=70):
            path = os.path.join(folder, fname)
            feats = extract_features(path)
            if feats:
                rows.append(feats + [label])
            else:
                skipped += 1

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)

    print(f"\n✅ Saved {len(rows)} samples → {OUTPUT_CSV}")
    print(f"⚠️  Skipped {skipped} images (no hand detected)")

if __name__ == "__main__":
    main()