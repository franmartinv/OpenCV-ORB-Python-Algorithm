# Python Artificial Vision

import cv2 # import OpenCV library
import numpy as np

def open_camera(ref_image):
    # I use ORB algorithm since SURF is no longer in this version of OpenCV
    orb = cv2.ORB_create()

    kp_ref, des_ref = ORB_detectAndCompute(orb,ref_image)

    capture = cv2.VideoCapture(0)

    while (capture.isOpened()):
        # Read camera fotogram
        ret, frame = capture.read()

        # Convert camera fotogram to gray-scale
        gray_capture = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect key points with ORB and calculate descriptors for gray image
        kp, des = ORB_detectAndCompute(orb,gray_capture)

        # Draw the key points on the image
        frame = cv2.drawKeypoints(frame, kp, None, (0, 0, 255), 4)

        cv2.imshow('ORB',frame)

        if (cv2.waitKey(1) == ord('s')):    # pulsar "s" para salir
            break

    capture.release()
    cv2.destroyAllWindows()
    
def load_ref_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def ORB_detectAndCompute(orb,ref_image):
    # Detect Key Points with Reference Image
    kp = orb.detect(ref_image, None)

    # Calculate reference image descriptors with ORB
    kp, des = orb.compute(ref_image, kp)

    return kp, des


def main():
    path = ""
    ref_image = load_ref_image(path)

    open_camera(ref_image)

main()
