import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original Image', image)
cv2.waitKey(0)

mask = np.zeros(image.shape[:2], dtype='uint8')
(cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
cv2.rectangle(mask,(cX - 200, cY - 200), (cX + 200, cY + 200), 255, -1)
cv2.imshow('Mask', mask)
cv2.waitKey(0)

masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow('Masked Applied to Image', masked)
cv2.waitKey(0)