"""
Interactive script to replace paintings in a gallery image by clicking frame corners.

Usage:
    python replace_frames_click.py

Controls (while gallery window open):
    - Left-click: add a corner point (any order).
    - 'r' key: reset the current selection (clear clicked points).
    - 'c' key: confirm the 4-point selection and proceed to choose replacement image.
    - 'q' key: quit the program.

After pressing 'c', you'll be prompted in the console to enter the replacement image filename/path.
"""

import cv2
import numpy as np
import os
import sys

# --------- Helper functions ---------
def order_points(pts):
    """
    Order 4 points as top-left, top-right, bottom-right, bottom-left.
    pts: array-like shape (4,2)
    Returns float32 array shape (4,2)
    """
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def blend_warped_into_gallery(gallery, warped, polygon_int, feather_radius=15):
    """
    Blend the warped image into gallery using a soft (feathered) mask.
    - gallery, warped: full-image arrays (same shape)
    - polygon_int: Nx2 int32 polygon (in original gallery coords)
    - feather_radius: approximate gaussian blur radius for feathering edges
    Returns blended result (uint8).
    """
    h, w = gallery.shape[:2]

    # single-channel mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon_int, 255)

    # Feather (blur) the mask to smooth edges
    # Kernel size should be odd and scale with feather_radius, clip to image size
    k = max(1, int(feather_radius))
    if k % 2 == 0:
        k += 1
    mask_blur = cv2.GaussianBlur(mask, (k, k), 0)

    # Convert to float alpha in [0,1]
    alpha = (mask_blur.astype(np.float32) / 255.0)[:, :, None]

    # Ensure warped and gallery are float32 for blending
    gallery_f = gallery.astype(np.float32)
    warped_f = warped.astype(np.float32)

    blended = gallery_f * (1.0 - alpha) + warped_f * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


# --------- Mouse callback and interactive selection ---------
class ClickSelector:
    def __init__(self, orig_img, max_display_w=1200, max_display_h=800):
        self.orig = orig_img  # original full-resolution gallery (BGR)
        self.h, self.w = orig_img.shape[:2]
        # compute display scale (shrink for comfortable screen)
        self.scale = min(1.0, max_display_w / float(self.w), max_display_h / float(self.h))
        # display image for clicking (resized copy)
        self.disp = cv2.resize(self.orig, (int(self.w * self.scale), int(self.h * self.scale)),
                               interpolation=cv2.INTER_AREA) if self.scale != 1.0 else self.orig.copy()
        self.window_name = "Gallery - click 4 corners (any order). 'r' reset, 'c' confirm, 'q' quit"
        self.clicked_display = []  # points in display coordinates
        self.clicked_original = []  # mapped to original image coords
        self.done = False
        self.current_copy = self.disp.copy()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_cb)

    def mouse_cb(self, event, x, y, flags, param):
        # Only respond to left button down
        if event == cv2.EVENT_LBUTTONDOWN:
            # record display coords
            self.clicked_display.append((int(x), int(y)))
            # map to original coordinates
            ox = int(round(x / self.scale))
            oy = int(round(y / self.scale))
            self.clicked_original.append((ox, oy))

    def reset(self):
        self.clicked_display = []
        self.clicked_original = []
        self.current_copy = self.disp.copy()

    def draw_overlay(self):
        # draw points and indices on a fresh copy
        img = self.disp.copy()
        # Instructions text
        cv2.putText(img, "Click 4 corners (any order). Press 'c' to confirm, 'r' to reset, 'q' to quit.",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        for i, (dx, dy) in enumerate(self.clicked_display):
            cv2.circle(img, (dx, dy), 6, (0, 255, 255), -1)
            cv2.putText(img, str(i+1), (dx+8, dy+8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)
        # If 4+ clicked, draw polygon connecting them (display order)
        if len(self.clicked_display) >= 4:
            pts = np.array(self.clicked_display[:4], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(0,128,255), thickness=2)
        self.current_copy = img
        cv2.imshow(self.window_name, img)

    def wait_for_selection(self):
        # show loop until confirmed or quit
        while True:
            self.draw_overlay()
            key = cv2.waitKey(20) & 0xFF
            if key == ord('r'):  # reset
                self.reset()
            elif key == ord('c'):  # confirm selection, only if 4 points present
                if len(self.clicked_original) >= 4:
                    pts4 = self.clicked_original[:4]
                    return pts4
                else:
                    print("Need 4 clicked points to confirm. Click more points or press 'r' to reset.")
            elif key == ord('q'):
                return None
            # continue looping; clicking will update clicked points via mouse callback


# --------- Main interactive loop ---------
def interactive_replace_loop(gallery_path):
    if not os.path.exists(gallery_path):
        raise FileNotFoundError(f"Gallery image not found: {gallery_path}")

    gallery = cv2.imread(gallery_path)  # BGR
    if gallery is None:
        raise RuntimeError("Failed to read gallery image.")

    print("Loaded gallery:", gallery_path)
    selector = ClickSelector(gallery)

    while True:
        # Let user click 4 points
        pts = selector.wait_for_selection()
        print(pts)
        if pts is None:
            print("Quitting.")
            break

        # Order points safely (TL,TR,BR,BL)
        dst_pts = order_points(pts)  # float32 in original image coords

        # Ask user for replacement image path
        repl_path = input("Enter replacement image filename/path (or blank to cancel): ").strip()
        if repl_path == "":
            print("Replacement canceled. You can select another frame or press 'q' to quit.")
            selector.reset()
            continue
        if not os.path.exists(repl_path):
            print("File not found:", repl_path)
            selector.reset()
            continue

        replacement = cv2.imread(repl_path)
        if replacement is None:
            print("Failed to read replacement image:", repl_path)
            selector.reset()
            continue

        # Compute homography from replacement image to destination polygon
        h_rep, w_rep = replacement.shape[:2]
        src_pts = np.array([[0,0], [w_rep-1,0], [w_rep-1,h_rep-1], [0,h_rep-1]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Warp replacement to gallery size
        warped = cv2.warpPerspective(replacement, M, (gallery.shape[1], gallery.shape[0]),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

        # Blend warped into gallery using soft mask (feathering)
        polygon_int = np.array(dst_pts, dtype=np.int32)
        gallery = blend_warped_into_gallery(gallery, warped, polygon_int, feather_radius=25)

        # Update the selector's original image and display to allow further edits
        selector.orig = gallery
        selector.h, selector.w = gallery.shape[:2]
        selector.scale = min(1.0, selector.disp.shape[1] / selector.w) if selector.w != 0 else 1.0
        # recompute display image (fit to same max display constraints)
        max_w, max_h = 1200, 800
        selector.scale = min(1.0, max_w / float(selector.w), max_h / float(selector.h))
        selector.disp = cv2.resize(selector.orig, (int(selector.w * selector.scale), int(selector.h * selector.scale)),
                                   interpolation=cv2.INTER_AREA) if selector.scale != 1.0 else selector.orig.copy()
        selector.reset()

        # Show updated gallery in a separate window (or simply continue; window already shows)
        cv2.imshow("Updated result", gallery)
        cv2.imwrite("my_gallery_replaced.jpg", gallery)
        print("Replacement done. Saved current result as 'my_gallery_replaced.jpg'.")
        print("You can replace another frame or press 'q' in the gallery window to exit.")

    # cleanup
    cv2.destroyAllWindows()
    print("Final image saved as 'my_gallery_replaced.jpg' (if any replacements were performed).")


# --------- Run script ---------
if __name__ == "__main__":
    gallery_file = "art-gallery.jpg"  # change if your filename differs
    try:
        interactive_replace_loop(gallery_file)
    except Exception as e:
        print("Error:", e)
        cv2.destroyAllWindows()
        sys.exit(1)
