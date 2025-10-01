import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import queue
import threading

# ---------------- CONFIG ----------------
root_path: str = "./images"
base_filenames: list[str] = ["ashtel", "kbz", "note"]
# ----------------------------------------

suffixes = ["-30", "-60", "-90"]
extensions = [".jpg", ".png"]

# thread-safe queue for commands
cmd_queue = queue.Queue()

def cli_listener():
    """Background thread to listen for CLI commands."""
    while True:
        cmd = input().strip().lower()
        cmd_queue.put(cmd)

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def calculate_selection_dimensions(points: np.ndarray) -> tuple:
    tl, tr, br, bl = points
    top_width = np.linalg.norm(tr - tl)
    bottom_width = np.linalg.norm(br - bl)
    width = (top_width + bottom_width) / 2
    left_height = np.linalg.norm(bl - tl)
    right_height = np.linalg.norm(br - tr)
    height = (left_height + right_height) / 2
    return width, height

def warp_images(img, src_points, fixed_dim=(250, 250)):
    src = order_points(np.array(src_points, dtype=np.float32))

    selection_width, selection_height = calculate_selection_dimensions(src)
    max_dimension = max(fixed_dim) * 2
    if selection_width >= selection_height:
        prop_width = max_dimension
        prop_height = int((selection_height / selection_width) * max_dimension)
    else:
        prop_height = max_dimension
        prop_width = int((selection_width / selection_height) * max_dimension)

    W_fixed, H_fixed = fixed_dim
    dst_fixed = np.array([[0,0],[W_fixed-1,0],[W_fixed-1,H_fixed-1],[0,H_fixed-1]], dtype=np.float32)
    dst_prop = np.array([[0,0],[prop_width-1,0],[prop_width-1,prop_height-1],[0,prop_height-1]], dtype=np.float32)

    H1 = cv2.getPerspectiveTransform(src, dst_fixed)
    warped_fixed = cv2.warpPerspective(img, H1, (W_fixed, H_fixed))

    H2 = cv2.getPerspectiveTransform(src, dst_prop)
    warped_prop = cv2.warpPerspective(img, H2, (prop_width, prop_height))

    return warped_fixed, warped_prop, (prop_width, prop_height)

def select_points(img, path):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img_rgb)
    plt.title(f"Select 4 corners: {os.path.basename(path)}")
    pts = plt.ginput(4)
    plt.close()
    return pts

def show_results(img, warped_fixed, warped_prop, base_name, prop_size):
    plt.close("all")  # close any previous figure
    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Original: {base_name}")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(cv2.cvtColor(warped_fixed, cv2.COLOR_BGR2RGB))
    plt.title(f"{base_name} Fixed 250x250")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(warped_prop, cv2.COLOR_BGR2RGB))
    plt.title(f"{base_name} Proportional {prop_size[0]}x{prop_size[1]}")
    plt.axis("off")

    plt.tight_layout()
    plt.show(block=True)  # block until closed

def main():
    results = []
    all_files = []

    for base in base_filenames:
        for s in suffixes:
            for ext in extensions:
                candidate = os.path.join(root_path, f"{base}{s}{ext}")
                if os.path.exists(candidate):
                    all_files.append(candidate)

    if not all_files:
        print("No images found.")
        return

    # Start CLI listener thread
    threading.Thread(target=cli_listener, daemon=True).start()

    idx = 0
    while idx < len(all_files):
        path = all_files[idx]
        img = cv2.imread(path)
        if img is None:
            print(f"Skipping unreadable {path}")
            idx += 1
            continue

        # === Step 1: select points for this image ===
        pts = select_points(img, path)
        if len(pts) != 4:
            print("Window closed without selecting 4 points. Exiting.")
            break

        # === Step 2: show results for this image ===
        warped_fixed, warped_prop, prop_size = warp_images(img, pts)
        show_results(img, warped_fixed, warped_prop, os.path.basename(path), prop_size)

        # Auto-accept points
        results.append((path, pts))
        idx += 1

        # === Step 3: handle commands (goback/skip) ===
        while not cmd_queue.empty():
            cmd = cmd_queue.get()
            if cmd == "skip":
                print("Skip command received.")
                if results:
                    results.pop()
                # skip means: keep idx where it is (already incremented)
            elif cmd == "goback":
                print("GoBack command received.")
                if results:
                    results.pop()
                idx = max(0, idx-1)
            else:
                print("Unknown command:", cmd)

    print("\nFinal results:")
    for path, pts in results:
        print(path, [(round(x,1), round(y,1)) for x,y in pts])

if __name__ == "__main__":
    main()
