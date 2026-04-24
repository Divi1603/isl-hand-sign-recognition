"""
ISL DATA COLLECTOR — TWO-HAND SUPPORT (FIXED)
Collects 600 samples per gesture from 3 people (100 each raw + augmented)
Run: python collect_data.py
"""

import cv2
import mediapipe as mp
import numpy as np
import csv, os, time

mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_draw_sty = mp.solutions.drawing_styles

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
    (15,16),(0,17),(17,18),(18,19),(19,20)
]

# ── Feature extraction (shared logic — keep identical to utils.py) ────────────
def normalize_landmarks(lm_list):
    arr   = np.array([[l.x, l.y, l.z] for l in lm_list])
    arr  -= arr[0].copy()
    scale = np.max(np.linalg.norm(arr, axis=1)) + 1e-6
    return arr / scale

def extract_single_hand(lm_list):
    arr   = normalize_landmarks(lm_list)
    dists = [float(np.linalg.norm(arr[a] - arr[b])) for a, b in CONNECTIONS]
    return arr[:,0].tolist() + arr[:,1].tolist() + arr[:,2].tolist() + dists   # 83 values

def get_empty_hand():
    return [0.0] * 83

def extract_features_two_hands(multi_hand_landmarks, multi_handedness):
    """
    Always returns 166 features:
      Left  hand: 83 (x21 + y21 + z21 + dist20)
      Right hand: 83 (x21 + y21 + z21 + dist20)
    Missing hand → zero-padded.
    NOTE: Frame must be flipped (cv2.flip) before MediaPipe processing
          so that hand labels match what was collected during training.
    """
    left_feats  = None
    right_feats = None

    if multi_hand_landmarks and multi_handedness:
        for lm, handedness in zip(multi_hand_landmarks, multi_handedness):
            label = handedness.classification[0].label   # 'Left' or 'Right'
            feats = extract_single_hand(lm.landmark)
            if label == "Left":
                left_feats  = feats
            else:
                right_feats = feats

    left_feats  = left_feats  if left_feats  is not None else get_empty_hand()
    right_feats = right_feats if right_feats is not None else get_empty_hand()
    return left_feats + right_feats   # 166 features total

def augment(feats):
    """Stronger noise augmentation for better generalization."""
    arr = np.array(feats)
    return (arr + np.random.normal(0, 0.015, arr.shape)).tolist()

# ── CONFIG ────────────────────────────────────────────────────────────────────
PERSONS = [
    {"name": "Person1", "color": (0, 255, 255)},
    {"name": "Person2", "color": (255, 165,   0)},
    
]

# ↑ Increased from 50 → 100 per person for better generalization
# Total per gesture: 300 raw + 300 augmented = 600 samples
SAMPLES_PER_PERSON = 100

GESTURES_TO_COLLECT = [
    '1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','O','P'
]

OUTPUT_CSV = "live_landmarks.csv"

HEADER = (
    [f"Lx{i}"    for i in range(21)] +
    [f"Ly{i}"    for i in range(21)] +
    [f"Lz{i}"    for i in range(21)] +
    [f"Ldist{i}" for i in range(20)] +
    [f"Rx{i}"    for i in range(21)] +
    [f"Ry{i}"    for i in range(21)] +
    [f"Rz{i}"    for i in range(21)] +
    [f"Rdist{i}" for i in range(20)] +
    ["label", "person"]
)

# ── Resume logic ──────────────────────────────────────────────────────────────
all_rows = []
done_pairs = set()

if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(list(row.values()))
            if "person" in row and "label" in row:
                done_pairs.add((row["person"], row["label"]))
    print(f"Resuming — {len(all_rows)} samples already collected.")
    print(f"Done pairs: {sorted(done_pairs)}\n")

def save_csv():
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(all_rows)
    print(f"  ✅ Saved {len(all_rows)} rows → {OUTPUT_CSV}")

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    for person in PERSONS:
        pname  = person["name"]
        pcolor = person["color"]

        print(f"\n{'='*50}\n  NEXT: {pname}\n{'='*50}")

        # ── Person ready screen ───────────────────────────────────────────────
        waiting = True
        while waiting:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            cv2.putText(frame, f"NEXT PERSON: {pname}",
                        (30, 100), cv2.FONT_HERSHEY_DUPLEX, 1.8, pcolor, 3)
            cv2.putText(frame, "Sit in front of camera",
                        (30, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.putText(frame, "Press SPACE when ready",
                        (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,100), 2)
            cv2.putText(frame, "ESC = save and quit",
                        (30, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,180,180), 1)
            cv2.putText(frame, "TIP: Vary hand angle & distance slightly each session!",
                        (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 1)
            cv2.imshow("ISL Collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '): waiting = False
            elif key == 27:
                save_csv(); cap.release(); cv2.destroyAllWindows(); exit()

        # ── Per-gesture loop ──────────────────────────────────────────────────
        for gesture in GESTURES_TO_COLLECT:

            if (pname, gesture) in done_pairs:
                print(f"  Skip {pname} / '{gesture}' (already done)")
                continue

            samples = []
            print(f"\n  {pname} — Gesture: '{gesture}'")

            # Gesture ready screen
            waiting = True
            while waiting:
                ret, frame = cap.read()
                if not ret: break
                # IMPORTANT: flip before MediaPipe so hand labels are consistent
                frame  = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if result.multi_hand_landmarks:
                    for hlm in result.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hlm, mp_hands.HAND_CONNECTIONS,
                            mp_draw_sty.get_default_hand_landmarks_style(),
                            mp_draw_sty.get_default_hand_connections_style())

                cv2.rectangle(frame, (0,0), (frame.shape[1], 185), (0,0,0), -1)
                cv2.putText(frame, f"{pname}  —  Gesture: '{gesture}'",
                            (20, 55), cv2.FONT_HERSHEY_DUPLEX, 1.4, pcolor, 2)
                cv2.putText(frame, "Show gesture, then SPACE to start capturing",
                            (20,105), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,200,255), 2)
                cv2.putText(frame, f"Target: {SAMPLES_PER_PERSON} samples  |  Q = skip  |  ESC = save & quit",
                            (20,155), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180,180,180), 1)

                cv2.imshow("ISL Collector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '): waiting = False
                elif key in (ord('q'), ord('Q')): gesture = None; break
                elif key == 27:
                    save_csv(); cap.release(); cv2.destroyAllWindows(); exit()

            if not gesture: continue

            # Capture loop
            auto     = False
            last_cap = 0

            while len(samples) < SAMPLES_PER_PERSON:
                ret, frame = cap.read()
                if not ret: break
                # IMPORTANT: flip before MediaPipe — must match app behavior
                frame  = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                hands_detected = 0
                if result.multi_hand_landmarks:
                    hands_detected = len(result.multi_hand_landmarks)
                    for hlm in result.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hlm, mp_hands.HAND_CONNECTIONS,
                            mp_draw_sty.get_default_hand_landmarks_style(),
                            mp_draw_sty.get_default_hand_connections_style())

                # Capture a sample every 0.06 s when auto-capturing
                if auto and time.time() - last_cap > 0.06:
                    feats = extract_features_two_hands(
                        result.multi_hand_landmarks,
                        result.multi_handedness if result.multi_handedness else [])
                    samples.append(feats)
                    last_cap = time.time()

                # Progress bar at bottom
                prog = int(len(samples) / SAMPLES_PER_PERSON * frame.shape[1])
                cv2.rectangle(frame,
                              (0, frame.shape[0]-25), (prog, frame.shape[0]),
                              pcolor, -1)

                # HUD
                cv2.rectangle(frame, (0,0), (frame.shape[1], 125), (0,0,0), -1)
                sc  = pcolor if auto else (0,165,255)
                msg = "CAPTURING — KEEP MOVING HAND SLIGHTLY!" if auto else "PAUSED — SPACE to start"
                cv2.putText(frame,
                    f"{pname}  '{gesture}'  [{len(samples)}/{SAMPLES_PER_PERSON}]  {msg}",
                    (15,45), cv2.FONT_HERSHEY_DUPLEX, 0.85, sc, 2)
                hc  = (0,255,100) if hands_detected > 0 else (0,60,255)
                ht  = (f"Hands detected: {hands_detected}/2"
                       if hands_detected > 0 else "NO HANDS — move closer")
                cv2.putText(frame, ht, (15,90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, hc, 2)
                cv2.putText(frame,
                    "SPACE=pause/resume  Q=next gesture  ESC=save&quit",
                    (15, frame.shape[0]-35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

                cv2.imshow("ISL Collector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '): auto = not auto
                elif key in (ord('q'), ord('Q')): break
                elif key == 27:
                    for feats in samples:
                        all_rows.append(feats + [gesture, pname])
                        all_rows.append(augment(feats) + [gesture, pname])
                    save_csv(); cap.release(); cv2.destroyAllWindows(); exit()

            # Commit samples (raw + augmented)
            for feats in samples:
                all_rows.append(feats + [gesture, pname])
                all_rows.append(augment(feats) + [gesture, pname])

            done_pairs.add((pname, gesture))
            total_for_gesture = sum(1 for r in all_rows if r[-2] == gesture)
            print(f"  ✅ {len(samples)} raw → {len(samples)*2} with augment "
                  f"| Total for '{gesture}': {total_for_gesture}")
            save_csv()

cap.release()
cv2.destroyAllWindows()

print(f"\n{'='*50}")
print(f"COLLECTION COMPLETE! Total rows: {len(all_rows)}")
print(f"{'='*50}")
print("\nNow run: python train_model.py")