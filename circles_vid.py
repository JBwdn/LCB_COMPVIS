#!/Users/Jake/anaconda3/bin/python

import numpy as np
import cv2


def prepare_frame(frame):
    output = frame.copy()
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(grey_frame, (7, 7), 0)
    return blur_frame, output


def find_circles(frame, output):
    circles_frame = cv2.HoughCircles(
        frame,
        cv2.HOUGH_GRADIENT,
        1,
        100,
        param1=50,
        param2=30,
        minRadius=1,
        maxRadius=100
        )
    if circles_frame is not None:
        circles_frame = np.round(circles_frame[0, :]).astype("int")
        for (x, y, r) in circles_frame:
            cv2.circle(
                output,
                (x, y),
                r,
                (0, 128, 255),
                3
                )
            cv2.rectangle(
                output,
                (x - r, y - r),
                (x + r, y + r),
                (0, 128, 255),
                3
                )
    return output


def track_circles(film):
    frame, output = prepare_frame(film)
    find_circles(frame, output)
    return output


def main(path):
    movie = cv2.VideoCapture(path)
    while movie.isOpened():
        retval, frame = movie.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or frame is None:
            movie.release()
            break
        else:
            output = track_circles(frame)
            cv2.imshow("OUTPUT", output)
    return


if __name__ == "__main__":
    main("videos/ball_test1.avi")
