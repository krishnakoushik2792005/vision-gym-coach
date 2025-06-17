import cv2
import mediapipe as mp
import numpy as np
import csv
from pathlib import Path

# ─── 1. Folder Setup ────────────────────────────────────────────────
input_folder = Path("input_folder")    # Put your .jpg images here
output_folder = Path("output_folder")          # Annotated outputs go here
output_folder.mkdir(exist_ok=True)

# ─── 2. Mediapipe & Drawing Setup ──────────────────────────────────
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ─── 3. Utility Functions ───────────────────────────────────────────

def calculate_angle(a, b, c):
    """Return angle (in degrees) at point b formed by a–b–c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def _p(lm, idx):
    """Get (x, y) of landmark idx."""
    return lm[idx].x, lm[idx].y

def _dist(a, b):
    """Euclidean distance between two points."""
    return np.linalg.norm(np.array(a) - np.array(b))

# ─── 4. Pull-Up Metric Functions ────────────────────────────────────

def elbow_angle(lm): 
    return calculate_angle(_p(lm, 12), _p(lm, 14), _p(lm, 16))

def torso_angle(lm): 
    hip, shoulder = _p(lm, 24), _p(lm, 12)
    dy, dx = hip[1] - shoulder[1], hip[0] - shoulder[0]
    return abs(np.degrees(np.arctan2(dx, dy)))

def hip_knee_angle(lm): 
    return calculate_angle(_p(lm, 24), _p(lm, 26), _p(lm, 28))

def elbow_flare_pct(lm): 
    shoulder_width = _dist(_p(lm, 11), _p(lm, 12))
    offset = abs(_p(lm, 14)[0] - _p(lm, 12)[0])
    return offset / (shoulder_width + 1e-6)

def scap_elevation_deg(lm): 
    ear_y = lm[8].y
    shoulder_y = lm[12].y
    return (ear_y - shoulder_y) * 360

def chin_over_bar(lm, bar_y=0.4): 
    return lm[0].y < bar_y

# ─── 5. Classification Logic ────────────────────────────────────────

fault_map = {
    "TORSO_LEAN":   lambda D: D["torso_angle"] > 15,
    "ELBOW_FLARE":  lambda D: D["elbow_offset_pct"] > 0.2,
    "KIPPING_HIPS": lambda D: D["hip_knee_angle"] < 160,
    "SCAP_SHRUG":   lambda D: D["scap_elevation_deg"] > 10,
}

def classify_faults(metrics):
    """Return list of fault labels or ['PERFECT_FORM'] if none."""
    faults = [label for label, rule in fault_map.items() if rule(metrics)]
    return faults if faults else ["PERFECT_FORM"]

# ─── 6. Batch Processing ─────────────────────────────────────────────

def main():
    # CSV header
    rows = [["filename", "elbow_angle", "torso_angle", 
             "hip_knee_angle", "elbow_offset_pct", 
             "scap_elevation_deg", "chin_over_bar", "faults"]]

    with mp_pose.Pose(static_image_mode=True) as pose:
        for img_path in input_folder.glob("*.jpg"):
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"⚠️ Could not load {img_path.name}")
                continue

            # Pose detection
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                print(f"❌ No pose detected in {img_path.name}")
                continue

            lm = res.pose_landmarks.landmark
            # Calculate metrics
            metrics = {
                "elbow_angle": elbow_angle(lm),
                "torso_angle": torso_angle(lm),
                "hip_knee_angle": hip_knee_angle(lm),
                "elbow_offset_pct": elbow_flare_pct(lm),
                "scap_elevation_deg": scap_elevation_deg(lm),
                "chin_over_bar": chin_over_bar(lm)
            }
            faults = classify_faults(metrics)
            faults_str = ", ".join(faults)

            # Save row
            rows.append([
                img_path.name,
                metrics["elbow_angle"],
                metrics["torso_angle"],
                metrics["hip_knee_angle"],
                metrics["elbow_offset_pct"],
                metrics["scap_elevation_deg"],
                metrics["chin_over_bar"],
                faults_str
            ])

            # Annotate & save image
            mp_drawing.draw_landmarks(image, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(image, faults_str, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            image_output = output_folder / img_path.name
            cv2.imwrite(str(image_output), image)

    # Write CSV
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("✅ Batch analysis complete. Check 'output/' and 'results.csv'.")

if __name__ == "__main__":
    main()
