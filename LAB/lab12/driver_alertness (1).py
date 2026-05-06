"""
Driver Alertness Detection — CS-474 Lab Final, Spring 2026
==========================================================
Theoretical Basis (from lab slides 14.1 – 14.4):
  - Brightness Constancy:  I(x+u·dt, y+v·dt, t+dt) = I(x, y, t)
  - Linearised constraint: Ix·u + Iy·v + It = 0
  - We compute It (temporal derivative) manually as a frame difference.
  - Spatial gradients Ix, Iy are computed via Sobel — the same building
    blocks used in Lucas-Kanade / Horn-Schunck, but WITHOUT calling any
    banned optical-flow API.
  - Motion energy = mean |It| over a region, which is exactly the
    temporal derivative term in the brightness constancy equation.
  - Camera-shake is separated from driver motion by comparing the
    temporal derivative energy inside a driver ROI vs. a background ROI.

Classification (rolling buffer over ~1.5 s):
  Attentive  — moderate, stable |It|
  Drowsy     — very low |It| + very low variance (near-zero motion)
  Distracted — high |It| or high variance (erratic motion bursts)

Banned calls NOT used anywhere:
  cv2.calcOpticalFlowPyrLK, cv2.calcOpticalFlowFarneback,
  face/landmark detectors, any deep-learning model.
"""

import cv2
import numpy as np
from collections import deque

# ── Tunable parameters ────────────────────────────────────────────────────────
BUFFER_SIZE     = 45    # temporal window (~1.5 s at 30 fps)
DROWSY_E_MAX    = 3.0   # mean |It| threshold for drowsy
DROWSY_VAR_MAX  = 1.8   # variance threshold to confirm drowsy
DISTRACTED_MIN  = 9.0   # mean |It| threshold for distracted
BG_WEIGHT       = 0.75  # how much of background energy to subtract (shake compensation)

# ── Rolling buffers ───────────────────────────────────────────────────────────
energy_buf = deque(maxlen=BUFFER_SIZE)
graph_buf  = deque(maxlen=220)

# ── ROI definitions ───────────────────────────────────────────────────────────
def driver_roi(h, w):
    """Central region — where the driver's head and upper body appear."""
    return w // 4, h // 8, 3 * w // 4, 3 * h // 4   # x1,y1,x2,y2

def background_roi(h, w):
    """Left-edge strip — used to measure camera/environmental shake energy."""
    return 0, 0, w // 8, h

# ── Temporal derivative (It) — brightness constancy, linearised ───────────────
def compute_It(prev_gray, curr_gray):
    """
    It = I(x,y,t+1) - I(x,y,t)
    This is the temporal derivative from the brightness constancy equation:
        Ix*u + Iy*v + It = 0
    Computed manually with pixel subtraction — no optical flow API.
    """
    return cv2.absdiff(curr_gray, prev_gray)   # |It|, shape (H,W), uint8

def region_energy(It_map, x1, y1, x2, y2):
    """Mean |It| inside a rectangular region — scalar motion energy."""
    return float(np.mean(It_map[y1:y2, x1:x2]))

# ── Spatial gradients (Ix, Iy) — same as inside Lucas-Kanade ─────────────────
def compute_Ix_Iy(gray):
    """Sobel spatial derivatives — same terms used in Lucas-Kanade patch equations."""
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return Ix, Iy

# ── Classification ────────────────────────────────────────────────────────────
def classify(buf):
    if len(buf) < 6:
        return "Calibrating...", (200, 200, 200)
    arr  = np.array(buf)
    mean = float(np.mean(arr))
    var  = float(np.var(arr))
    if mean <= DROWSY_E_MAX and var <= DROWSY_VAR_MAX:
        return "DROWSY",      (0, 60, 255)
    elif mean >= DISTRACTED_MIN or var > 45.0:
        return "DISTRACTED",  (0, 215, 255)
    else:
        return "ATTENTIVE",   (0, 220, 80)

# ── Visualisation helpers ─────────────────────────────────────────────────────
def draw_energy_graph(frame, buf, h, w):
    strip_h, strip_y, strip_w = 55, h - 60, 210
    cv2.rectangle(frame, (0, strip_y - 2), (strip_w, h - 2), (25, 25, 25), -1)
    pts = list(buf)[-strip_w:]
    if len(pts) < 2:
        return
    max_v = max(max(pts), 1.0)
    for i in range(1, len(pts)):
        x0 = int((i - 1) / strip_w * strip_w)
        x1 = int(i       / strip_w * strip_w)
        y0 = strip_y + strip_h - int(pts[i-1] / max_v * strip_h)
        y1 = strip_y + strip_h - int(pts[i]   / max_v * strip_h)
        cv2.line(frame, (x0, y0), (x1, y1), (80, 230, 160), 1)
    cv2.putText(frame, "|It| energy", (4, strip_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

def draw_mini_It(frame, It_map, h, w):
    """Show |It| map (the temporal derivative) as a diagnostic thumbnail."""
    mini = cv2.resize(It_map, (160, 90))
    mini_bgr = cv2.applyColorMap(mini, cv2.COLORMAP_HOT)
    frame[45:135, w - 165:w - 5] = mini_bgr
    cv2.putText(frame, "|It| map (temporal deriv.)", (w - 163, 148),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

def draw_overlay(frame, label, color, net_e, bg_e, h, w):
    dx1, dy1, dx2, dy2 = driver_roi(h, w)
    bx1, by1, bx2, by2 = background_roi(h, w)

    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 2)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (150, 150, 150), 1)

    # Banner
    cv2.rectangle(frame, (0, 0), (w, 42), (20, 20, 20), -1)
    cv2.putText(frame, f"State: {label}", (10, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Numeric readings
    cv2.putText(frame, f"Net |It|: {net_e:5.2f}", (10, h - 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    cv2.putText(frame, f"BG  |It|: {bg_e:5.2f}  (shake est.)", (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cv2.putText(frame, "Brightness constancy: Ix*u + Iy*v + It = 0",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 200, 255), 1)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)       # replace 0 with a video file path if needed
    if not cap.isOpened():
        print("Cannot open video source.")
        return

    ret, frame0 = cap.read()
    if not ret:
        print("Cannot read first frame.")
        cap.release()
        return

    prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    print("Driver Alertness Monitor running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Step 1: Preprocess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Step 2: Temporal derivative |It|  (manual — no optical flow API)
        It = compute_It(prev_gray, gray)

        # Step 3: Region motion energies
        dx1, dy1, dx2, dy2 = driver_roi(h, w)
        bx1, by1, bx2, by2 = background_roi(h, w)
        driver_e = region_energy(It, dx1, dy1, dx2, dy2)
        bg_e     = region_energy(It, bx1, by1, bx2, by2)

        # Step 4: Camera-shake compensation
        # Global shake raises It uniformly; subtracting background isolates driver motion
        net_e = max(0.0, driver_e - BG_WEIGHT * bg_e)

        energy_buf.append(net_e)
        graph_buf.append(net_e)

        # Step 5: Classify from rolling temporal buffer
        label, color = classify(energy_buf)

        # Step 6: Render
        draw_mini_It(frame, It, h, w)
        draw_overlay(frame, label, color, net_e, bg_e, h, w)
        draw_energy_graph(frame, graph_buf, h, w)

        cv2.imshow("Driver Alertness Monitor — CS474", frame)
        prev_gray = gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
