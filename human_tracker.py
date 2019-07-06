#!/Users/Jake/anaconda3/bin/python
# TODO: Function to search for faces only within detected figures
import cv2
from sys import argv

def load_models():
    face_cascade = cv2.CascadeClassifier(
        "models/haarcascade_frontalface_default.xml")
    figure_cascade = cv2.CascadeClassifier(
        "models/haarcascade_fullbody.xml")
    return face_cascade, figure_cascade


def prepare_frame(frame):
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return grey_frame


def map_faces(frame, face_cascade, scaleFactor):
    grey_frame = prepare_frame(frame)
    face_map = face_cascade.detectMultiScale(grey_frame, scaleFactor, 5)
    return face_map


def map_figures(frame, figure_cascade, scaleFactor):
    grey_frame = prepare_frame(frame)
    figure_map = figure_cascade.detectMultiScale(grey_frame, scaleFactor, 5)
    return figure_map


def annotate_features(clean_frame, map_list):
    labled_frame = clean_frame.copy()
    for feature_map in map_list:
        if feature_map is not None:
            for (fx, fy, fw, fh) in feature_map:
                cv2.rectangle(
                    labled_frame,
                    (fx, fy),
                    (fx+fw, fy+fh),
                    (160, 32, 240),
                    1)
    return labled_frame


def main(path, scaleFactor):
    face_cascade, figure_cascade = load_models()
    movie = cv2.VideoCapture(path)
    while movie.isOpened():
        retval, frame = movie.read()
        if (cv2.waitKey(1) & 0xFF == ord("q")) or frame is None:
            movie.release()
            break
        else:
            figures = map_figures(frame, figure_cascade, scaleFactor)
            faces = map_faces(frame, face_cascade, scaleFactor)
            labled_frame = annotate_features(frame, [figures, faces])
            cv2.imshow("OUTPUT", labled_frame)


if __name__ == "__main__":
    if len(argv) == 1:
        main("videos/runners_test1.mp4", 1.10)
    if len(argv) == 2:
        print(f"Tracking figures in: {argv[1]}")
        main(argv[1], 1.10)
    if len(argv) == 3:
        print(f"Tracking figures in: {argv[1]} With a scale factor of: {argv[2]}")
        main(argv[1], float(argv[2]))
