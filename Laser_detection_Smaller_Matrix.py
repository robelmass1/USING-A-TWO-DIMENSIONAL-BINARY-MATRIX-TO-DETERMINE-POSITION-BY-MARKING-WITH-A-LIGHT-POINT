import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


def detect_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4:
            corners = approx.reshape(4, 2)
            return corners
    return None

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_perspective_transform(src_img, src_points, dst_size):
    src_points = order_points(np.array(src_points, dtype='float32'))
    dst_points = np.array([  # Destination points for perspective transform
        [0, 0],
        [dst_size[0] - 1, 0],
        [dst_size[0] - 1, dst_size[1] - 1],
        [0, dst_size[1] - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    dst_img = cv2.warpPerspective(src_img, M, dst_size)
    return dst_img, M

def detect_grid_and_box_size(warped_image, grid_shape):
    h, w = warped_image.shape[:2]
    box_size = (w // grid_shape[1], h // grid_shape[0])
    grid = np.zeros(grid_shape, dtype=int)

    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    box_h, box_w = box_size

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            cell = binary[i * box_h:(i + 1) * box_h, j * box_w:(j + 1) * box_w]
            grid[i, j] = 1 if np.mean(cell) > 127 else 0

    return grid, box_size

def draw_grid(warped_image, grid_shape, box_size):
    for i in range(1, grid_shape[0]):
        cv2.line(warped_image, (0, i * box_size[1]), (warped_image.shape[1], i * box_size[1]), (0, 255, 0), 1)
    for j in range(1, grid_shape[1]):
        cv2.line(warped_image, (j * box_size[0], 0), (j * box_size[0], warped_image.shape[0]), (0, 255, 0), 1)

        
def capture_baseline_grid(warped_image, grid_shape):
    grid, box_size = detect_grid_and_box_size(warped_image, grid_shape)
    return grid, box_size


def detect_laser_pointer(frame):
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea) if contours else None

    # Contour filtering based on circularity
    for contour in contours:
        if cv2.isContourConvex(contour):
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity > 0.7:  # Threshold for circularity
                # Proceed with laser detection
                pass

    if largest_contour is not None:
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
    return None


def debug_laser_pointer(warped_image, laser_pos, box_size, grid_shape):
    cv2.circle(warped_image, laser_pos, 10, (0, 0, 255), -1)

    grid_x = laser_pos[0] // box_size[0]
    grid_y = laser_pos[1] // box_size[1]

    if 0 <= grid_x < grid_shape[1] and 0 <= grid_y < grid_shape[0]:
        top_left = (grid_x * box_size[0], grid_y * box_size[1])
        bottom_right = ((grid_x + 1) * box_size[0], (grid_y + 1) * box_size[1])
        cv2.rectangle(warped_image, top_left, bottom_right, (255, 0, 0), 2)

        cv2.putText(warped_image, f"Laser at: ({grid_x}, {grid_y})", 
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


def handle_laser_stability(transformed_laser, grid_shape, box_size, baseline_grid, M,
                           laser_stable_start_time, previous_laser_cell):
    grid_x = transformed_laser[0] // box_size[0]
    grid_y = transformed_laser[1] // box_size[1]

    if 0 <= grid_x < grid_shape[1] and 0 <= grid_y < grid_shape[0]:
        current_laser_cell = (grid_x, grid_y)
        if current_laser_cell == previous_laser_cell:
            if laser_stable_start_time is None:
                laser_stable_start_time = time.time()
            elif time.time() - laser_stable_start_time >= 0.5:
                print_laser_info(grid_x, grid_y, box_size, baseline_grid, M)
                laser_stable_start_time = None
        else:
            laser_stable_start_time = None
        previous_laser_cell = current_laser_cell

    return laser_stable_start_time, previous_laser_cell

def print_laser_info(grid_x, grid_y, box_size, baseline_grid, M):

        # Extract 3x3 binary matrix around the clicked point
    roi_binary = baseline_grid[max(0, grid_y - 1):min(baseline_grid.shape[0], grid_y + 2),
                               max(0, grid_x - 1):min(baseline_grid.shape[1], grid_x + 2)]
    
        # Flatten the matrix
    flattened = roi_binary.flatten()
    flattened_str = ' '.join(map(str, flattened))
    print("Flattened Binary Matrix:")
    print(flattened_str)

    color = 'White' if baseline_grid[grid_y, grid_x] == 1 else 'Black'

    print("Laser is stable, processing the cell.")
    print(f"Laser Pointing at grid cell: ({grid_x}, {grid_y}) - {color}")
    print("Box Size:", box_size)
    print("Grid Shape:", baseline_grid.shape)
    print("Transformation Matrix (M):\n", M)
    print("Grid:\n", baseline_grid)

    print("Extracted 3x3 binary matrix:")
    print(roi_binary)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(roi_binary, cmap='gray')
    plt.title("Extracted 3x3 Binary Matrix")

    plt.subplot(1, 2, 2)
    plt.imshow(baseline_grid, cmap='gray')
    plt.title("Full Grid with Binary States")
    plt.colorbar()
    plt.scatter(grid_x, grid_y, color='red')
    plt.text(grid_x, grid_y - 0.5, 'Laser Position', color='red', fontsize=12, ha='center', va='center')
    plt.figtext(0.5, 0.01, f"Kernel Code: {flattened_str}", ha='center', fontsize=20, bbox={"facecolor": "lightblue", "alpha": 0.5, "pad": 5})
    plt.show()

def main():
    cap = cv2.VideoCapture(0)
    grid_shape = (10, 10)
    dst_size = (800, 800)  

    laser_stable_start_time = None      
    previous_laser_cell = None
    baseline_grid = None
    box_size = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners = detect_corners(frame)
        if corners is not None:
            for corner in corners:
                cv2.circle(frame, tuple(corner), 5, (0, 255, 0), -1)
            cv2.polylines(frame, [corners.reshape((-1, 1, 2)).astype(np.int32)], isClosed=True, color=(0, 255, 255),
                          thickness=2)
            
            ordered_corners = order_points(corners)
            warped_image, M = get_perspective_transform(frame, ordered_corners, dst_size)

            # Capture the baseline grid the first time
            if baseline_grid is None:
                baseline_grid, box_size = capture_baseline_grid(warped_image, grid_shape)

            # Draw grid using baseline grid
            draw_grid(warped_image, grid_shape, box_size)

            laser_pos = detect_laser_pointer(frame)
            if laser_pos:
                transformed_laser = cv2.perspectiveTransform(np.array([[laser_pos]], dtype="float32"), M)[0][0]
                transformed_laser = tuple(map(int, transformed_laser))

                debug_laser_pointer(warped_image, transformed_laser, box_size, grid_shape)

                laser_stable_start_time, previous_laser_cell = handle_laser_stability(
                    transformed_laser, grid_shape, box_size, baseline_grid, M,
                    laser_stable_start_time, previous_laser_cell)

            cv2.imshow('Warped Image', warped_image)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
