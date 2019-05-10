#!/Users/Jake/anaconda3/bin/python

import numpy as np
import cv2


def prepare_frame(film):
    retval, frame = film.read()
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
        param1=30,
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
                (0, 255, 0),
                4
                )
    return output


def track_circles(film):
    while film.isOpened():
        frame, output = prepare_frame(film)
        find_circles(frame, output)
        cv2.imshow("OUTPUT", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main(path):
    movie = cv2.VideoCapture(path)
    track_circles(movie)
    movie.release()
    return


if __name__ == "__main__":
    main("videos/ball_test1.avi")
