import cv2
import numpy as np

from utils import filer_image_color, display_image, line_from_points, find_x_for_y


def draw_lanes(img: np.array, lines: np.array, height: int, height_threshold: int) -> np.array:
    """
    Define logic for deciding if line is a street lane.
    Draw left and right street lane.
    :param img: bgr image
    :param lines: contains output of probabilistic hue method
    :param height: vertical coordinate describing end of region of interest
    :param height_threshold: vertical coordinate describing beginning of region of interest

    :return img_copy, start: image with drawn lanes, coordinates of lanes
    """
    img_copy = img.copy()
    start = []
    if lines is not None:
        line_coefficients = []
        line_candidates = [line_array.ravel() for line_array in lines]
        # Exclude lines that were extracted on top of a picture
        line_candidates = list(filter(lambda x: x[1] > height_threshold or x[3] > height_threshold, line_candidates))

        for line in line_candidates:
            x1 = (line[0], line[1])
            x2 = (line[2], line[3])
            # Find function equation coefficients
            a, b = line_from_points(x1=x1, x2=x2)
            line_coefficients.append((a, b))

        # Sort lines based on their point slope
        line_candidates_sorted = sorted(line_coefficients, key=lambda x: x[1])
        left_line, right_line = line_candidates_sorted[0], line_candidates_sorted[-1]
        for line in (left_line, right_line):
            pt1 = int(find_x_for_y(slope=line[0], point_slope=line[1], y=height)), height
            pt2 = int(find_x_for_y(slope=line[0], point_slope=line[1], y=height_threshold)), height_threshold
            cv2.line(img_copy, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
            start.append(pt1)
            start.append(pt2)
        start = sorted(start, key=lambda x: x[0])
    return img_copy, start


def get_lanes(img: np.array) -> np.array:
    """
    Filter yellow left and white right lane.
    Apply edge detection.
    Get lines.

    :param img: bgr image
    :return: lane candidates
    """
    # img_kmeans = cluster_image_colors(img=img, n_clusters=7)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_yellow = filer_image_color(img_hsv=img_hsv,
                                            # Yellow color range
                                            color_hsv_lower=[90, 90, 200],
                                            color_hsv_upper=[110, 130, 240])
    img_white = filer_image_color(img_hsv=img_hsv,
                                           # Yellow color range
                                           color_hsv_lower=[55, 20, 190],
                                           color_hsv_upper=[80, 40, 255])
    # Erode white pixels
    kernel = np.ones((3, 3), np.uint8)
    img_white = cv2.morphologyEx(img_white, cv2.MORPH_OPEN, kernel=kernel)
    # img_white = cv2.erode(img_white, kernel=kernel, iterations=2)

    img_final = cv2.add(img_yellow, img_white)

    img_edges = cv2.Canny(img_final, 50, 200)

    # Hough Line keeps tracks of the intersection between curves of every point in image.
    # https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    linesP = cv2.HoughLinesP(image=img_edges,
                             rho=1,
                             theta=np.pi / 180,
                             threshold=100,  # How many points must coincide (to create line)
                             minLineLength=100,
                             maxLineGap=50)
    return linesP


def main():
    img_name = 'street.png'
    img = cv2.imread(img_name)
    img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
    height, width = img.shape[:2]
    height_threshold = (height // 4) * 3
    lanes = get_lanes(img=img)
    img_lanes, points = draw_lanes(img=img, lines=lanes, height=height, height_threshold=height_threshold)
    display_image(img=img_lanes, name=img_name, color_callback=True)


if __name__ == "__main__":
    main()
