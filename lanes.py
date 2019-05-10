#!/Users/Jake/anaconda3/bin/python
import cv2
import numpy as np
from PIL import Image


def reduce_resolution(IMG_PATH):
    IMG = Image.open(IMG_PATH)
    IMG = IMG.resize(
        (int(IMG.size[0] / 2), int(IMG.size[1] / 2)),
        Image.ANTIALIAS
    )
    return IMG


def canny(image):
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    canny = cv2.Canny(blur, 50, 50)
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[
        (200, height),
        (1100, height),
        (550, 250)
    ]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


image_PIL = reduce_resolution('test_image2.png')
image_opencv = cv2.cvtColor(np.array(image_PIL), cv2.COLOR_RGB2BGR)
lane_image = np.copy(image_opencv)
canny = canny(lane_image)
cropped_image = region_of_interest(canny)

lines = cv2.HoughLinesP(
    cropped_image,
    2,
    np.pi/180,
    3,
    np.array([]),
    minLineLength=40,
    maxLineGap=9
)

line_image = display_lines(lane_image, lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)


cv2.imshow("Result", combo_image)
cv2.waitKey(0)
