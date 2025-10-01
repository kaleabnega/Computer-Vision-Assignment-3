import cv2
import numpy as np
import matplotlib.pyplot as plt

def order_points(pts):
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    pts: np.array shape (4,2)
    """
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left has smallest sum (x+y)
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest diff (x-y)
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest diff
    return rect

# ----------------------------
# CONFIG: change file name to your image
img_path = "business-card.jpg"   # <- replace with your filename
out_path = "card_topdown_250x250.jpg"
# ----------------------------

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Cannot read '{img_path}' — put the photo in the same folder or fix the path.")

# Show the image (matplotlib shows RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title("Click the 4 corners in this order: top-left, top-right, bottom-right, bottom-left\nClose the figure when done")
pts = plt.ginput(4)   # click 4 times, then close the window
plt.close()

# pts is list of (x,y) floats; convert to numpy array
src = np.array(pts, dtype=np.float32)
print(src)

# ensure consistent ordering (optional if you clicked in correct order)
src = order_points(src)

# Destination square corners (250x250)
W_out, H_out = 250, 250
dst = np.array([
    [0, 0],
    [W_out - 1, 0],
    [W_out - 1, H_out - 1],
    [0, H_out - 1]
], dtype=np.float32)

# Compute homography and warp
H = cv2.getPerspectiveTransform(src, dst)   # src -> dst
warped = cv2.warpPerspective(img, H, (W_out, H_out), flags=cv2.INTER_LINEAR)

# Save and show
cv2.imwrite(out_path, warped)
print("Saved:", out_path)

plt.figure(figsize=(5,5))
plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
plt.title("Warped 250x250 (top-down)")
plt.axis('off')
plt.show()
