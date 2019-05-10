import numpy as np
import cv2

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

def roi(img,vertices):
    # blank mask:
    mask = np.zeros_like(img)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, 255)

    # returning the image only where mask pixels are nonzero
    masked = cv2.bitwise_and(img, mask)
    return masked

cv2.imshow("result", roi(lane_image, np.array([[(200, height), (1100, height), (550, 250)]]))
cv2.waitKey(0)
