import cv2
import os
import numpy as np


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


def display_image(img: np.array, name: str, color_callback: bool = False):

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


def main():
    img_name = 'highway.jpg'
    img = cv2.imread(img_name)
    img = cv2.resize(img, dsize=(0, 0), fx=0.2, fy=0.2)
    img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_kmeans = cluster_image_colors(img=img, n_clusters=7)
    img_kmeans_hsv = cv2.cvtColor(img_kmeans, cv2.COLOR_RGB2HSV)
    img_filtered_yellow = filer_image_color(img_hsv=img_kmeans_hsv,
                                            # Yellow color range
                                            color_hsv_lower=[90, 165, 220],
                                            color_hsv_upper=[110, 185, 240])
    img_filtered_white = filer_image_color(img_hsv=img_kmeans_hsv,
                                           # Yellow color range
                                            color_hsv_lower=[15, 8, 200],
                                            color_hsv_upper=[30, 20, 225])
    # Erode white pixels
    kernel = np.ones((3, 3), np.uint8)
    img_erosion_white = cv2.morphologyEx(img_filtered_white, cv2.MORPH_OPEN, kernel=kernel)
    img_final = img_filtered_yellow + img_erosion_white
    display_image(img=img_final, name=img_name, color_callback=True)


if __name__ == "__main__":
    main()