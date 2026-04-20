"""
CS-474: Computer Vision - Open Lab Spring 2026
Lane Detection using Hough Transform
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# STEP 1: Load and Preprocess the Image
# ─────────────────────────────────────────────
img_path = "/home/claude/open_lab2/open lab/road.jpg"
original = cv2.imread(img_path)
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
H, W = original.shape[:2]

# Convert to grayscale
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise before edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# ─────────────────────────────────────────────
# STEP 2: Edge Detection using Canny
# ─────────────────────────────────────────────
# Thresholds tuned for road/lane markings
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# ─────────────────────────────────────────────
# STEP 3: Region of Interest (ROI) Masking
# ─────────────────────────────────────────────
# Define a trapezoidal ROI focused on the lower half of the image
# where lane markings are most likely to appear
roi_vertices = np.array([[
    (int(W * 0.05), H),           # bottom-left
    (int(W * 0.45), int(H * 0.55)),  # top-left
    (int(W * 0.55), int(H * 0.55)),  # top-right
    (int(W * 0.95), H)            # bottom-right
]], dtype=np.int32)

mask = np.zeros_like(edges)
cv2.fillPoly(mask, roi_vertices, 255)
roi_edges = cv2.bitwise_and(edges, mask)

# ─────────────────────────────────────────────
# STEP 4: Hough Transform Line Detection
# ─────────────────────────────────────────────
# Parameters:
#   rho        = 1 pixel resolution in distance
#   theta      = 1 degree (pi/180) angular resolution
#   threshold  = minimum votes to consider a line
#   minLineLen = minimum length of a line segment
#   maxLineGap = maximum allowed gap in a line segment

lines = cv2.HoughLinesP(
    roi_edges,
    rho=1,
    theta=np.pi / 180,
    threshold=40,
    minLineLength=80,
    maxLineGap=60
)

# ─────────────────────────────────────────────
# STEP 5: Separate & Average Left / Right Lanes
# ─────────────────────────────────────────────
def average_slope_intercept(lines, img_shape):
    """Cluster detected line segments into left and right lanes
       by slope sign, then fit a single representative line for each."""
    left_lines, right_lines = [], []

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:          # skip perfectly vertical (undefined slope)
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Filter by slope magnitude to remove near-horizontal noise
        if abs(slope) < 0.4:
            continue

        if slope < 0:          # negative slope → left lane (image y is flipped)
            left_lines.append((slope, intercept))
        else:                  # positive slope → right lane
            right_lines.append((slope, intercept))

    def make_coords(img_shape, line_params):
        if not line_params:
            return None
        slope, intercept = np.mean(line_params, axis=0)
        y1 = img_shape[0]                  # bottom of image
        y2 = int(y1 * 0.60)               # ~60% up the image
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return (x1, y1, x2, y2)

    left_avg  = make_coords(img_shape, left_lines)
    right_avg = make_coords(img_shape, right_lines)
    return left_avg, right_avg


left_lane, right_lane = average_slope_intercept(lines, original.shape)

# ─────────────────────────────────────────────
# STEP 6: Draw Detected Lanes on the Image
# ─────────────────────────────────────────────
lane_overlay = np.zeros_like(original)

def draw_lane(img, lane, color=(0, 255, 0), thickness=8):
    if lane is not None:
        x1, y1, x2, y2 = lane
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

draw_lane(lane_overlay, left_lane,  color=(255, 0, 0))   # blue  → left
draw_lane(lane_overlay, right_lane, color=(0, 255, 0))   # green → right

# Also draw ROI polygon outline on a separate copy for visualization
roi_vis = original_rgb.copy()
cv2.polylines(roi_vis, roi_vertices, isClosed=True, color=(255, 255, 0), thickness=2)

# Final composite: blend lane lines onto original
result = cv2.addWeighted(original, 0.8, lane_overlay, 1.0, 0)
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# ─────────────────────────────────────────────
# STEP 7: Visualise All Pipeline Stages
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#1a1a2e')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)

titles = [
    "1. Original Image",
    "2. Grayscale + Blur",
    "3. Canny Edge Detection",
    "4. ROI Masked Edges",
    "5. Raw Hough Lines",
    "6. Final Lane Detection"
]

# Build raw hough lines image
raw_hough = original_rgb.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(raw_hough, (x1, y1), (x2, y2), (255, 0, 0), 2)

images = [
    original_rgb,
    blurred,
    edges,
    roi_edges,
    raw_hough,
    result_rgb
]
cmaps = [None, 'gray', 'gray', 'gray', None, None]

for i, (title, image, cmap) in enumerate(zip(titles, images, cmaps)):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=6)
    ax.axis('off')

    # Annotate ROI on panel 4
    if i == 3:
        pts = roi_vertices[0].reshape((-1, 1, 2))
        poly = plt.Polygon(roi_vertices[0], fill=False,
                           edgecolor='yellow', linewidth=1.5)
        ax.add_patch(poly)

plt.suptitle("Lane Detection Pipeline — Hough Transform\nCS-474 Computer Vision | Open Lab Spring 2026",
             color='white', fontsize=14, fontweight='bold', y=1.01)

plt.savefig("/mnt/user-data/outputs/lane_detection_result.png",
            dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("Saved successfully.")

# ─────────────────────────────────────────────
# Also save just the clean final result
# ─────────────────────────────────────────────
cv2.imwrite("/mnt/user-data/outputs/lane_final.jpg", result)
print("Final image saved.")