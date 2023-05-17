# Python Artificial Vision

import cv2 # import OpenCV library
import numpy as np
from pathlib import Path, PureWindowsPath
from matplotlib import pyplot as plt

def open_camera(gray_ref_image,ref_image):
    # I use ORB algorithm since SURF is no longer in this version of OpenCV
    orb = cv2.ORB_create()

    # I get keypoints and descriptors from the reference image
    kp_ref, des_ref = ORB_detectAndCompute(orb,gray_ref_image)

    # Matcher object creation
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

    capture = cv2.VideoCapture(0)

    while (capture.isOpened()):
        # Read camera fotogram
        ret, frame = capture.read()

        # Convert frame to grayscale
        gray_capture = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect key points with ORB and calculate descriptors for gray image
        kp, des = ORB_detectAndCompute(orb,gray_capture)    

        # Calculate the coincidences
        matches = matcher.match(des_ref, des)

        # Draw the key points on the image
        #frame = cv2.drawKeypoints(frame, kp, None, (0, 0, 255), 4)

        # Order the matches with their distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw the coincidence points
        #final_img = cv2.drawMatches(ref_image, kp_ref, gray_capture, kp, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        final_img = cv2.drawMatches(ref_image, kp_ref, frame, kp, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow('Matches',final_img)

        if (cv2.waitKey(1) == ord('s')):    # pulsar "s" para salir
            break

    capture.release()
    cv2.destroyAllWindows()
    
def load_ref_image(path):
    img = cv2.imread(path)
    resize_img = cv2.resize(img,(0, 0),fx=0.4, fy=0.35, interpolation = cv2.INTER_AREA)
    gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    gray_resize_img = cv2.resize(gray_img,(0, 0),fx=0.4, fy=0.35, interpolation = cv2.INTER_AREA)
    return gray_resize_img, resize_img

def ORB_detectAndCompute(orb,image):
    # Detect Key Points with Reference Image
    kp = orb.detect(image, None)

    # Calculate reference image descriptors
    kp, des = orb.compute(image, kp)

    return kp, des

def main():
    path = ""
    gray_ref_image, ref_image = load_ref_image(path)

    open_camera(gray_ref_image,ref_image)

main()
