#!/Users/Jake/anaconda3/bin/python
# TODO: stack figure and face annotations for output
import cv2
import numpy as np


def load_models():
    face_cascade = cv2.CascadeClassifier(
        "models/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(
        "models/haarcascade_eye.xml")
    figure_cascade = cv2.CascadeClassifier(
        "models/haarcascade_fullbody.xml")
    return face_cascade, eye_cascade, figure_cascade


def prepare_frame(frame):
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return grey_frame


def map_faces(frame, face_cascade, eye_cascade):
    labled_frame = frame.copy()
    grey_frame = prepare_frame(frame)
    face_map = face_cascade.detectMultiScale(grey_frame, 1.3, 5)
    for (fx, fy, fw, fh) in face_map:
        cv2.rectangle(
            labled_frame,
            (fx, fy),
            (fx+fw, fy+fh),
            (255, 0, 0),
            2
            )
        face_colour = labled_frame[fy:fy+fh, fx:fx+fw]
        face_gray = grey_frame[fy:fy+fh, fx:fx+fw]
        eyes = eye_cascade.detectMultiScale(face_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                face_colour,
                (ex, ey),
                (ex+ew, ey+eh),
                (0, 255, 0),
                2
                )
    return labled_frame


def map_figures(frame, figure_cascade):
    labled_frame = frame.copy()
    grey_frame = prepare_frame(frame)
    figure_map = figure_cascade.detectMultiScale(grey_frame, 1.3, 5)
    for (fx, fy, fw, fh) in figure_map:
        cv2.rectangle(
            labled_frame,
            (fx, fy),
            (fx+fw, fy+fh),
            (255, 0, 0),
            2
            )
    return labled_frame


def main(path):
    face_cascade, eye_cascade, figure_cascade = load_models()
    movie = cv2.VideoCapture(path)
    while movie.isOpened():
        retval, frame = movie.read()
        if (cv2.waitKey(1) & 0xFF == ord("q")) or frame is None:
            movie.release()
            break
        else:
            A = map_figures(
                frame,
                figure_cascade)
            B = map_faces(
                frame,
                face_cascade,
                eye_cascade)
            labled_frame = np.hstack([A, B])
            cv2.imshow("OUTPUT", labled_frame)
    return


if __name__ == "__main__":
    main("videos/runners_test1.mp4")
