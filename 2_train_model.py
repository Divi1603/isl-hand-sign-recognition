"""
ISL TRAINING — TWO-HAND SUPPORT (166 features)
FIXED VERSION:
  ✅ Split BEFORE scale (no data leakage)
  ✅ Reduced RF depth 8 (was 12) — main overfit fix
  ✅ Smaller MLP (128,64) with stronger L2 alpha=0.05
  ✅ Stronger augmentation noise=0.015 (was 0.008)
  ✅ 4x augmentation copies (was 2x) for better generalization
  ✅ Pipeline CV for honest cross-validation scores
  ✅ Scaler saved from train-only fit
Target accuracy: 85–90%  |  Run: python train_model.py
"""

import ast
import pandas as pd
import numpy as np
import pickle
import json
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

CSV_PATH          = "live_landmarks.csv"
EXPECTED_FEATURES = 166

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading CSV …")
try:
    df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
except TypeError:
    df = pd.read_csv(CSV_PATH, error_bad_lines=False)

print(f"Raw rows: {len(df)}")

# ── Remove junk labels ─────────────────────────────────────────────────────────
JUNK = {"nan", "NaN", "1.0", "2.0", "3.0", "4.0", "5.0",
        "6.0", "7.0", "8.0", "9.0"}
before = len(df)
df = df[~df["label"].astype(str).isin(JUNK)]
df.dropna(subset=["label"], inplace=True)
print(f"Dropped {before - len(df)} junk rows. Remaining: {len(df)}")

# ── Fix stringified-list rows ──────────────────────────────────────────────────
drop_cols    = [c for c in ["label", "person"] if c in df.columns]
feature_cols = [c for c in df.columns if c not in drop_cols]
first_feat   = feature_cols[0]

bad_mask = df[first_feat].apply(
    lambda v: isinstance(v, str) and v.strip().startswith("[")
)
if bad_mask.sum():
    print(f"Expanding {bad_mask.sum()} stringified-list rows …")
    good_df = df[~bad_mask].copy()
    bad_df  = df[bad_mask].copy()

    def expand_row(row):
        try:
            vals = [float(v) for v in
                    ast.literal_eval(row[first_feat])[:EXPECTED_FEATURES]]
            vals += [0.0] * (EXPECTED_FEATURES - len(vals))
            return pd.Series(vals, index=feature_cols)
        except Exception:
            return pd.Series([np.nan] * len(feature_cols), index=feature_cols)

    exp_feats = bad_df.apply(expand_row, axis=1)
    exp_feats["label"] = bad_df["label"].values
    if "person" in bad_df.columns:
        exp_feats["person"] = bad_df["person"].values
    df = pd.concat([good_df, exp_feats], ignore_index=True)
    print(f"After expansion: {len(df)} rows")

print(f"\nDataset: {len(df)} samples, {df['label'].nunique()} classes")
print(df["label"].value_counts().to_string())

# ── Build X, y ─────────────────────────────────────────────────────────────────
X = df.drop(columns=drop_cols).values
y = df["label"].astype(str).values

try:
    X = X.astype(float)
except ValueError:
    X_df = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
    X    = X_df.values

N_FEATURES = X.shape[1]
print(f"\nFeature count: {N_FEATURES}")
if N_FEATURES != EXPECTED_FEATURES:
    print(f"⚠️  Expected {EXPECTED_FEATURES} but got {N_FEATURES} — check CSV!")
else:
    print(f"✅ Feature count correct: {N_FEATURES} (two-hand, 166 features)\n")

# ── Clean data ─────────────────────────────────────────────────────────────────
zero_mask = (X == 0).all(axis=1)
if zero_mask.sum():
    print(f"Dropping {zero_mask.sum()} fully-zero rows …")
    X, y = X[~zero_mask], y[~zero_mask]

nan_mask = np.isnan(X).any(axis=1)
if nan_mask.sum():
    print(f"Dropping {nan_mask.sum()} NaN rows …")
    X, y = X[~nan_mask], y[~nan_mask]

X = np.nan_to_num(X, nan=0.0)

# ── Encode labels ──────────────────────────────────────────────────────────────
le    = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

# ── Split FIRST, then scale (no data leakage) ─────────────────────────────────
print("\n⚙️  Splitting data BEFORE scaling (prevents data leakage) …")
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_enc, test_size=0.15, stratify=y_enc, random_state=42
)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)   # ← fit ONLY on train
X_te_sc = scaler.transform(X_te)       # ← transform test with train stats
print(f"Train: {len(X_tr_sc)}  |  Test: {len(X_te_sc)}\n")

# ── Data augmentation on training set only ─────────────────────────────────────
def augment_landmarks(X_data, y_data, noise_std=0.015, n_copies=3):
    """
    Stronger noise augmentation (0.015 vs old 0.008).
    3 extra copies = 4x total data for better generalization.
    """
    parts_X = [X_data]
    parts_y = [y_data]
    for _ in range(n_copies):
        noisy = X_data + np.random.randn(*X_data.shape) * noise_std
        parts_X.append(noisy)
        parts_y.append(y_data)
    return np.vstack(parts_X), np.concatenate(parts_y)

print("Augmenting training data with stronger noise …")
X_tr_aug, y_tr_aug = augment_landmarks(X_tr_sc, y_tr, noise_std=0.015, n_copies=3)
print(f"Augmented train size: {len(X_tr_aug)} (4× original)\n")

# ── Models — reduced complexity to prevent overfitting ────────────────────────

# Random Forest: shallower depth is the key overfit fix
rf = RandomForestClassifier(
    n_estimators=200,       # more trees = more stable (not more overfit)
    max_depth=8,            # ↓ was 12 — shallower = less memorization
    min_samples_split=15,   # ↑ was 10 — needs more evidence to split
    min_samples_leaf=6,     # ↑ was 4  — more samples required at each leaf
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

# MLP: smaller network + stronger regularization
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),   # ↓ was (256,128) — smaller = less memorization
    activation="relu",
    alpha=0.05,                      # ↑ was 0.01 — stronger L2 regularization
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=25,             # ↑ was 20 — more patience for convergence
    random_state=42
)

ensemble = VotingClassifier([("rf", rf), ("mlp", mlp)], voting="soft")

# ── Honest CV using Pipeline (scaler inside each CV fold) ─────────────────────
print("5-fold cross-validation (Pipeline — no leakage) …")
pipe_rf = Pipeline([("scaler", StandardScaler()), ("rf",
    RandomForestClassifier(
        n_estimators=200,
        max_depth=8,            # ← matches rf above
        min_samples_split=15,
        min_samples_leaf=6,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
)])
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_sc  = cross_val_score(pipe_rf, X, y_enc, cv=cv,
                         scoring="accuracy", n_jobs=-1)
print(f"CV: {cv_sc.mean()*100:.1f}%  ±  {cv_sc.std()*100:.1f}%\n")

# ── Train on augmented data ────────────────────────────────────────────────────
print("Training ensemble on augmented data …")
ensemble.fit(X_tr_aug, y_tr_aug)

# Evaluate on original (non-augmented) splits
tr_acc = ensemble.score(X_tr_sc, y_tr)
te_acc = ensemble.score(X_te_sc, y_te)
gap    = tr_acc - te_acc

print(f"\n{'='*52}")
print(f"  Split          : 85% Train / 15% Test")
print(f"  Train samples  : {len(X_tr)} (augmented to {len(X_tr_aug)} for training)")
print(f"  Test  samples  : {len(X_te)}")
print(f"  Train accuracy : {tr_acc*100:.1f}%")
print(f"  Test  accuracy : {te_acc*100:.1f}%")
print(f"  Gap            : {gap*100:.1f}%")
print(f"  CV accuracy    : {cv_sc.mean()*100:.1f}% ± {cv_sc.std()*100:.1f}%")

# Tighter target range — 85–92% is healthy for this dataset
if te_acc > 0.92:
    print("  STATUS: ⚠️  Still high — collect more varied data / people")
elif 0.85 <= te_acc <= 0.92:
    print("  STATUS: ✅ Target accuracy range (85–92%) — good to deploy!")
elif te_acc >= 0.75:
    print("  STATUS: Acceptable — more data or reduce gesture similarity")
else:
    print("  STATUS: Below target — collect more samples")

if gap > 0.10:
    print("  ⚠️  Overfit — add more people / recording sessions")
elif gap > 0.05:
    print("  ℹ️  Mild overfit — acceptable for this dataset size")
else:
    print("  ✅ Healthy train/test gap (≤5%)")
print(f"{'='*52}\n")

y_pred = ensemble.predict(X_te_sc)
print(classification_report(y_te, y_pred, target_names=le.classes_))

# ── Save artefacts ─────────────────────────────────────────────────────────────
with open("isl_model.pkl",     "wb") as f: pickle.dump(ensemble, f)
with open("label_encoder.pkl", "wb") as f: pickle.dump(le, f)
with open("scaler.pkl",        "wb") as f: pickle.dump(scaler, f)

report_data = {
    "test_accuracy":      float(te_acc),
    "train_accuracy":     float(tr_acc),
    "cv_mean":            float(cv_sc.mean()),
    "cv_std":             float(cv_sc.std()),
    "gap":                float(gap),
    "classes":            list(le.classes_),
    "n_features":         int(N_FEATURES),
    "expected_features":  EXPECTED_FEATURES,
    "two_hand_model":     N_FEATURES == EXPECTED_FEATURES,
    "train_split":        0.85,
    "test_split":         0.15,
    "augmentation":       "noise std=0.015, 3 copies (4x total)",
    "fixes_applied": [
        "split_before_scale",
        "reduced_rf_depth_8",          # ↓ from 12
        "reduced_rf_min_leaf_6",       # ↑ from 4
        "reduced_mlp_size_128x64",     # ↓ from 256x128
        "stronger_l2_alpha_0.05",      # ↑ from 0.01
        "stronger_augmentation_0.015", # ↑ from 0.008
        "more_aug_copies_3",           # ↑ from 2
        "pipeline_cv_no_leakage"
    ]
}
with open("training_report.json", "w") as f:
    json.dump(report_data, f, indent=2)

print("✅ Models saved: isl_model.pkl | label_encoder.pkl | scaler.pkl")
print("✅ Report saved: training_report.json")

# ── Confusion matrix ───────────────────────────────────────────────────────────
cm  = confusion_matrix(y_te, y_pred)
sz  = max(8, len(le.classes_))
fig, ax = plt.subplots(figsize=(sz, sz - 2))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_, yticklabels=le.classes_,
            cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix  (test acc: {te_acc*100:.1f}%)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
print("✅ confusion_matrix.png saved\n")

# ── Sanity / noise test ────────────────────────────────────────────────────────
print("Noise test (overfit sanity check):")
fake   = np.random.randn(1, N_FEATURES)
fake_s = scaler.transform(fake)
probs  = ensemble.predict_proba(fake_s)[0]
top_p  = max(probs)
print(f"  Random noise → {le.classes_[np.argmax(probs)]} ({top_p*100:.1f}%)")
if top_p > 0.70:
    print("  ⚠️  >70% on noise = still overfit. Collect more varied data.")
elif top_p < 0.40:
    print("  ✅ <40% on noise = healthy model.")
else:
    print("  ℹ️  Borderline — acceptable.")

# ── Per-class sample count warning ────────────────────────────────────────────
print("\nPer-class sample counts:")
counts = pd.Series(y).value_counts().sort_index()
for cls, cnt in counts.items():
    flag = "⚠️  (< 200, collect more)" if cnt < 200 else "✅"
    print(f"  {cls:>10} : {cnt:>4} samples  {flag}")

# ── Per-class accuracy check ───────────────────────────────────────────────────
print("\nPer-class test accuracy (gestures below 80% need more data):")
from sklearn.metrics import confusion_matrix as cm_fn
cm_raw = cm_fn(y_te, y_pred)
for i, cls in enumerate(le.classes_):
    row_sum = cm_raw[i].sum()
    if row_sum > 0:
        cls_acc = cm_raw[i, i] / row_sum * 100
        flag = "⚠️  needs more data" if cls_acc < 80 else "✅"
        print(f"  {cls:>10} : {cls_acc:>6.1f}%  {flag}")

print("\nDone! Restart your Streamlit app to load the new model.")
print("Run: streamlit run app.py")