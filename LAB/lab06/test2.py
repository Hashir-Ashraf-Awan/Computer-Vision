import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── 1. Load Image ──────────────────────────────────────────
img = cv2.imread("/home/claude/open_lab2/open lab/road.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]

# ── 2. Grayscale + Blur ────────────────────────────────────
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# ── 3. Canny Edge Detection ────────────────────────────────
edges = cv2.Canny(blur, 50, 150)

# ── 4. Region of Interest (ROI) ────────────────────────────
roi_points = np.array([[
    (50,       H),
    (W//2-60,  H//2 + 60),
    (W//2+60,  H//2 + 60),
    (W-50,     H)
]], dtype=np.int32)

mask = np.zeros_like(edges)
cv2.fillPoly(mask, roi_points, 255)
masked = cv2.bitwise_and(edges, mask)

# ── 5. Hough Line Detection ────────────────────────────────
lines = cv2.HoughLinesP(masked, 1, np.pi/180,
                        threshold=40,
                        minLineLength=80,
                        maxLineGap=60)

# ── 6. Draw Lines on Image ─────────────────────────────────
output = img_rgb.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 4)

# ── 7. Show Results ────────────────────────────────────────
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(masked, cmap='gray')
plt.title("Edges + ROI")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(output)
plt.title("Detected Lanes")
plt.axis('off')

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/lane_easy_result.png", dpi=150)
plt.show()
print("Done!")