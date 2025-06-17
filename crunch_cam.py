import cv2, mediapipe as mp, numpy as np, time
from collections import deque
WINDOW = 5                           # running-average length
smooth = {k: deque(maxlen=WINDOW) for k in
          ("trunk","neck","hipK","scap","back")}
state  = {"in_rep": False}           # simple rep gate
def vec(a, b):                      # 2-D vector from a→b in image space
    return np.array([b[0]-a[0], b[1]-a[1]])

def unit(v):                        # avoid /0
    n = np.linalg.norm(v) + 1e-8
    return v / n
# ── tiny helpers we still need ───────────────────────────────────────
def p(lm, idx):
    """Return (x, y) of one landmark in *image* coordinates (0-1)."""
    return lm[idx].x, lm[idx].y

def angle(a, b, c):
    """3-point joint angle  (deg)  at b  using 2-D coords a-b-c."""
    a, b, c = map(np.array, (a, b, c))
    cos = np.dot(a-b, c-b) / (np.linalg.norm(a-b)*np.linalg.norm(c-b)+1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def angle_with_vertical(v):
    """Returns 0° when v points straight up (screen −y), 90° when horizontal."""
    vert = np.array([0.0, -1.0])
    cos  = np.dot(unit(v), vert)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

# ── Metric functions (crunch) ────────────────────────────────────────
def metrics_from_landmarks(lm, H):
    hip      = p(lm, 24)            # right-side only for speed; you can average L/R
    shoulder = p(lm, 12)
    knee     = p(lm, 26)
    ankle    = p(lm, 28)

    trunk_flex = angle_with_vertical(vec(hip, shoulder))   # 0–90°
    neck_flex  = angle_with_vertical(vec(shoulder, p(lm, 0)))
    hip_kink   = angle(p(lm, 24), p(lm, 26), p(lm, 28))    # unchanged

    scap_elev  = (lm[8].y - lm[12].y) * H                 # ear vs shoulder
    shldr_low  = H - lm[12].y * H                         # shoulder above mat

    return {
        "trunk": trunk_flex,
        "neck" : neck_flex,
        "hipK" : hip_kink,
        "scap" : scap_elev,
        "back" : shldr_low
    }

def classify(m, H):
    for k, v in m.items():
        smooth[k].append(v)
    avg = {k: sum(v)/len(v) for k, v in smooth.items()}

    # —— gate ——
    START, END = 20, 10             # ◄ trunk flex° to start / reset a rep
    if not state["in_rep"]:
        if avg["trunk"] > START:
            state["in_rep"] = True
        else:
            return []               # stay silent between crunches
    elif avg["trunk"] < END:
        state["in_rep"] = False

    # —— faults ——
    px = H / 1080
    faults = []
    if avg["trunk"] < 25:                faults.append("SHALLOW")
    if avg["trunk"] > 95:                faults.append("OVERFLEX")
    if avg["neck"]  > 20:                faults.append("NECK")
    if avg["back"]  < 40*px:             faults.append("HIP_UP")
    if avg["hipK"]  > 140:               faults.append("HIP_TUCK")
    if avg["scap"]  > -10*px:            faults.append("SHRUG")

    return faults or ["PERFECT"]

# ── MediaPipe & OpenCV setup ─────────────────────────────────────────
mp_pose, mp_draw = mp.solutions.pose, mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1)         # default webcam ~30 fps
cap  = cv2.VideoCapture(0)                       # change index if needed

print("🔴 Press q to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    H, W = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res  = pose.process(rgb)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        m  = metrics_from_landmarks(lm, H)
        faults = classify(m, H)
        label  = " ".join(faults)
        color  = (0,255,0) if faults==["PERFECT"] else (0,0,255)

        mp_draw.draw_landmarks(frame, res.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS,
                               landmark_drawing_spec=mp_draw.DrawingSpec(
                                   thickness=2, circle_radius=2, color=(255,255,255)),
                               connection_drawing_spec=mp_draw.DrawingSpec(
                                   thickness=2, color=(255,255,255)))

        y0 = H-60
        for i, word in enumerate(label.split()):
            cv2.putText(frame, word, (10, y0+i*25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2, cv2.LINE_AA)

        # Optional: print raw numbers once per second
        if int(time.time()) % 1 == 0:
            print({k: round(v,1) for k,v in m.items()}, "|", label)

    cv2.imshow("Crunch Coach", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
