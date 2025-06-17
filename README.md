# Vision-Gym-Coach 🏋️‍♂️ 📐

Real-time posture analysis for **crunches** and **pull-ups** using MediaPipe pose landmarks and classic computer vision – **no wearable sensors required**.


---

## ✨ Key features

**Crunch Exercise:**
- Metrics computed:
  - Trunk flexion
  - Neck flexion
  - Hip angle (kink)
  - Scapular elevation
  - Shoulder lift off-mat
- Faults flagged: `SHALLOW` · `OVERFLEX` · `NECK` · `HIP_UP` · `HIP_TUCK` · `SHRUG` · `PERFECT`

**Pull-up Exercise:**
- Metrics computed:
  - Elbow flexion (ROM)
  - Bar-path symmetry
  - Swing amplitude
  - Scapular retraction
- Faults flagged: `PARTIAL_ROM` · `KIPPING` · `ASYMMETRY` · `SHRUG` · `PERFECT`

*Sliding-window smoothing (5 frames) + a rule engine turn raw numbers into friendly, on-screen coaching.*

---

## 🔧 Quick start

1. Clone the repository:
   ```bash
   git clone https://github.com/krishnakoushik2792005/vision-gym-coach.git
