#!/Users/Jake/anaconda3/bin/python

import cv2

# Load NN models:
face_cascade = cv2.CascadeClassifier(
    'models/haarcascade_frontalface_default.xml'
    )
eye_cascade = cv2.CascadeClassifier(
    'models/haarcascade_eye.xml'
    )

# Load image and prepare:
img = cv2.imread('images/faces_test1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in img:
faces = face_cascade.detectMultiScale(gray, 1.05, 5)

# Loop over faces and find any eyes:
for (fx, fy, fw, fh) in faces:
    cv2.rectangle(
        img,
        (fx, fy),
        (fx+fw, fy+fh),
        (255, 0, 0),
        2
        )
    # Search each face area for eyes:
    face_color = img[fy:fy+fh, fx:fx+fw]
    face_gray = gray[fy:fy+fh, fx:fx+fw]
    eyes = eye_cascade.detectMultiScale(face_gray, 1.05, 5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(
            face_color,
            (ex, ey),
            (ex+ew, ey+eh),
            (0, 255, 0),
            2
            )

# Show annotated image, quit on key press:
cv2.imshow('Image', img)
cv2.waitKey(0)
