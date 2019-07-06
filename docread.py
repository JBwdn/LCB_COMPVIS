#!/Users/Jake/anaconda3/bin/python

import cv2
import pytesseract
import argparse
import os
from PIL import Image
# Process command line arguments:
ap = argparse.ArgumentParser()
ap.add_argument("-i",
                "--image",
                required=True,
                help="Path to image to be parsed"
                )
ap.add_argument("-p",
                "--preprocess",
                type=str,
                default="thresh",
                help="Type of preprocessing used")
ap.add_argument("-e",
                "--engine",
                type=str,
                default="tesseract",
                help="OCR engine")
args = vars(ap.parse_args())

# Class to hold, process and parse the image:
class document:
    def __init__(self):
        self.path = args["image"]
        self.image = cv2.imread(self.path)
        self.grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)


    def preprocess(self):
        if args["preprocess"] == "thresh":
            self.processed = cv2.threshold(
                    self.grey,
                    0,
                    255,
                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        elif args["preprocess"] == "blur":
            self.processed = cv2.medianBlur(self.grey, 3)


    def show_image(self):
        cv2.imshow("TEST", self.processed)
        cv2.waitKey(0)


    def parse_text(self):
        if args["engine"] == "tesseract":
            self.tesseract()
        elif args["engine"] == "NN":
            pass


    def tesseract(self):
        # Write image to temporary file:
        fn = "{}.png".format(os.getpid())
        cv2.imwrite(fn, self.processed)
        # Parse text using Tesseract bindings:
        content = pytesseract.image_to_string(Image.open(fn))
        os.remove(fn)
        print(content)

# Main function:
def main():
    doc = document()
    doc.preprocess()
    doc.parse_text()
    doc.show_image()

if __name__ == "__main__":
    main()
