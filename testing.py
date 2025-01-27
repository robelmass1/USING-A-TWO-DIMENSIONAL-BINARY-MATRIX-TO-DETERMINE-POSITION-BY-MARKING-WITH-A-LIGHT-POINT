import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

def perspective_transform(image, laser_position, cell_size=27, grid_size=5):
    laser_x, laser_y = laser_position

    half_size = cell_size // 2
    src_points = np.array([
        [laser_x - (half_size * grid_size), laser_y - (half_size * grid_size)],
        [laser_x + (half_size * grid_size), laser_y - (half_size * grid_size)],
        [laser_x - (half_size * grid_size), laser_y + (half_size * grid_size)],
        [laser_x + (half_size * grid_size), laser_y + (half_size * grid_size)]
    ], dtype="float32")

    dst_points = np.array([
        [0, 0],
        [grid_size * cell_size, 0],
        [0, grid_size * cell_size],
        [grid_size * cell_size, grid_size * cell_size]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, matrix, (grid_size * cell_size, grid_size * cell_size))
    return warped

def extract_5x5_matrix(frame, laser_position):
    box_size = 25
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)

    box_pnt1 = (laser_position[0] - (box_size * 3), laser_position[1] - (box_size * 3))
    box_pnt2 = (laser_position[0] + (box_size * 3), laser_position[1] + (box_size * 3))

    warped_image = perspective_transform(binary_image, laser_position, cell_size=25, grid_size=5)
    downsampled_matrix = cv2.resize(warped_image, (5, 5), interpolation=cv2.INTER_AREA)
    _, final_matrix = cv2.threshold(downsampled_matrix, 0.5, 1, cv2.THRESH_BINARY)
    flattened_matrix = final_matrix.flatten()

    return binary_image, final_matrix, warped_image, flattened_matrix, box_pnt1, box_pnt2

def detect_laser(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))  # Detect red laser

    moments = cv2.moments(mask)
    if moments['m00'] > 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return (cx, cy)
    return None

def stabilize_laser_position(current_position, last_position, laser_stable_count, threshold=10, stability_range=5):
    if last_position and abs(current_position[0] - last_position[0]) < stability_range and abs(current_position[1] - last_position[1]) < stability_range:
        laser_stable_count += 1
    else:
        laser_stable_count = 0

    return laser_stable_count

def visualize_results(original_detected, binary_image, extracted_matrix, warped_image, flattened_matrix, laser_position):
    
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    
    axs[0].imshow(original_detected[..., ::-1])  # Convert BGR to RGB
    axs[0].set_title("Original Image")
    axs[0].scatter([laser_position[0]], [laser_position[1]], color='red', label='Laser Position')
    axs[0].legend()
    
    axs[1].imshow(binary_image, cmap='gray')
    axs[1].set_title("Binary Image")
    axs[1].scatter([laser_position[0]], [laser_position[1]], color='red', label='Laser Position')
    axs[1].legend()
    
    axs[2].imshow(warped_image, cmap='gray', interpolation='none')
    axs[2].set_title("After perspective transform")
    
    axs[3].imshow(extracted_matrix, cmap='gray', interpolation='none')
    axs[3].set_title("Flattened 5x5 Matrix")

    plt.figtext(0.5, 0.01, f"Flattened Matrix: {flattened_matrix}", ha='center', fontsize=20, bbox={"facecolor": "lightblue", "alpha": 0.5, "pad": 5})
    plt.show()

def main():
    cap = cv2.VideoCapture(0)
    laser_stable_count = 0
    last_position = None
    orig_frame = None

    for i in range(0, 30):
        _, orig_frame = cap.read()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_position = detect_laser(frame)
        if current_position:
            laser_stable_count = stabilize_laser_position(current_position, last_position, laser_stable_count)
            last_position = current_position

            if laser_stable_count > 10:
                binary_image, extracted_matrix, warped_image, flattened_matrix, box_pnt1, box_pnt2 = extract_5x5_matrix(orig_frame, current_position)
                cv2.rectangle(frame, box_pnt1, box_pnt2, (255, 0, 0), 4)
                visualize_results(frame, binary_image, extracted_matrix, warped_image, flattened_matrix, current_position)

                laser_stable_count = 0

        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
