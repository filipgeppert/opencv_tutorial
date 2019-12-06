import cv2
import os
import numpy as np
import math


def normalize_image(img: np.array):
    frame_gray: np.array = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Reduce noise coming from consecutive frames
    frame_roi: np.array = cv2.blur(frame_gray, ksize=(21, 21))
    return frame_roi

def cluster_image_colors(img: np.array, n_clusters: int):
    # Convert to float32, because this is what cv2 kmeans expects
    img_kmeans: np.array = np.float32(img.reshape((-1, 3)))
    ret, labels, centers = cv2.kmeans(data=img_kmeans,
                                      K=n_clusters,
                                      bestLabels=None,
                                      criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                      attempts=10,
                                      flags=cv2.KMEANS_RANDOM_CENTERS)

    # Replace pixels in clusters with their center values. Return to proper data format.
    centers = np.uint8(centers)
    results = centers[labels.ravel()]
    result_img = results.reshape(img.shape)
    return result_img


def filer_image_color(img_hsv: np.array, color_hsv_lower: list, color_hsv_upper: list):
    lower = np.array(color_hsv_lower, dtype="uint64")
    upper = np.array(color_hsv_upper, dtype="uint64")
    mask = cv2.inRange(src=img_hsv, lowerb=lower, upperb=upper)
    output = cv2.bitwise_and(src1=img_hsv, src2=img_hsv, mask=mask)
    return output


def display_image(img: np.array, name: str = "image", color_callback: bool = False):

    def show_hsv_color(event, x, y, flags, param):
        if event == cv2.EVENT_FLAG_LBUTTON:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            print(hsv[y, x])

    if color_callback:
        cv2.namedWindow(name)
        cv2.setMouseCallback(name, show_hsv_color, img)
        cv2.imshow(name, img)
    else:
        cv2.imshow(name, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def line_from_points(x1: tuple, x2: tuple):
    # y = ax + b
    slope = (x1[1] - x2[1]) / (x1[0] - x2[0])
    b = x1[1] - slope*x1[0]
    # (slope, zero point)
    return slope, b


def find_x_for_y(slope: float, point_slope: float, y: int):
    return (y-point_slope) * (1/slope)
