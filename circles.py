#!/Users/Jake/anaconda3/bin/python

import numpy as np
import cv2


# Load the image, clone it for output & convert it to grayscale:
image = cv2.imread("images/ball_test2.jpg")
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect circles in image:
blur = cv2.GaussianBlur(gray, (3, 3), 0)
circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    1,  # Inverse ratio of acc to image resolution (2 = 1/2)
    100,  # Min distance between the center of circles
    param1=75,  # Upper threshold passed to canny edge detector
    param2=75,  # Accumulator threshold
    minRadius=1,
    maxRadius=500
    )

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(
            output,
            (x, y),
            r,
            (0, 255, 0),
            4
            )

    cv2.imshow("output", np.hstack([image, output]))
    cv2.waitKey(0)
