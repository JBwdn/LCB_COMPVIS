#!/Users/Jake/anaconda3/bin/python

import numpy as np
import cv2


def prepare_frame(frame):
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(grey_frame, (7, 7), 0)
    return blur_frame


def find_circles(prep_frame):
    circle_map = cv2.HoughCircles(
        prep_frame,
        cv2.HOUGH_GRADIENT,
        1,
        100,
        param1=50,
        param2=30,
        minRadius=1,
        maxRadius=100
        )
    return circle_map


def annotate_circles(clean_frame, circle_map):
    labled_frame = clean_frame.copy()
    if circle_map is not None:
        circle_map = np.round(circle_map[0, :]).astype("int")
        for (x, y, r) in circle_map:
            cv2.circle(
                labled_frame,
                (x, y),
                r,
                (0, 128, 255),
                3
                )
            cv2.rectangle(
                labled_frame,
                (x - r, y - r),
                (x + r, y + r),
                (0, 128, 255),
                3
                )
    return labled_frame


def map_circles(clean_frame):
    prep_frame = prepare_frame(clean_frame)
    circle_map = find_circles(prep_frame)
    return annotate_circles(clean_frame, circle_map)


def main(path):
    movie = cv2.VideoCapture(path)
    while movie.isOpened():
        retval, frame = movie.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or frame is None:
            movie.release()
            break
        else:
            labled_frame = map_circles(frame)
            cv2.imshow("OUTPUT", labled_frame)
    return


if __name__ == "__main__":
    main("videos/ball_test1.avi")
