import cv2
import numpy as np

from car_line_detector import get_lanes, draw_lanes
from utils import normalize_image


def main():
    template = cv2.imread('lights.png', 0)
    t_w, t_h = template.shape[::-1]
    img_name = 'street_2.mp4'
    cap = cv2.VideoCapture(img_name)
    # Read until video is completed
    is_first_frame = True
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_normalized = normalize_image(img=frame)
        if ret:
            h, w = frame_normalized.shape
            height_threshold = (h // 3) * 2
            # Detect lanes
            lanes = get_lanes(img=frame)
            frame, points = draw_lanes(img=frame, lines=lanes, height=h, height_threshold=height_threshold)
            # Assume first frame to be background
            if is_first_frame:
                roi_start, roi_end = points[0], points[-1]
                frame_roi = frame_normalized[-height_threshold:, roi_start[0]:roi_end[0]]
                first_frame = frame_roi.copy()
                is_first_frame = False
            else:
                frame_roi = frame_normalized[-height_threshold:, roi_start[0]:roi_end[0]]
                frame_color_roi = frame[-height_threshold:, roi_start[0]:roi_end[0]]
                # Mask roi
                mask = np.zeros(frame.shape, dtype=np.uint8)
                roi_corners = np.array([points])
                cv2.fillPoly(mask, roi_corners, (255, 255, 255))
                # Calculate the difference
                frame_difference = cv2.absdiff(first_frame, frame_roi)
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask_gray = mask_gray[-height_threshold:, roi_start[0]:roi_end[0]]
                frame_difference_masked = cv2.bitwise_and(src1=frame_difference, src2=mask_gray)
                _, thresh = cv2.threshold(frame_difference_masked, thresh=110, maxval=255, type=cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
                contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))
                if contours_sorted:
                    area = cv2.contourArea(contours_sorted[-1])
                    if area > 12_000:
                        # display_image(thresh)
                        bounding_rectangle = cv2.boundingRect(contours_sorted[-1])
                        x, y, w, h = bounding_rectangle
                        cv2.rectangle(frame_color_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        pass
                # Try to match street lights
                cv2.imshow("Frame", frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        # Break the loop
        else:
            break

    # Release the video capture object and close all frames
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
