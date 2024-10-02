# Import the required packages
import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()

# We add 'video_path' argument using add_argument() including a help.
ap.add_argument("-v","--video", help="path to the video file")
ap.add_argument("-k", "--k", required=False, help="k value")
args = vars(ap.parse_args())

# Create a VideoCapture object. In this case, the argument is the video file name:
capture = cv2.VideoCapture(args['video'])
k = float(args['k'])
 
# Check if the video is opened successfully
if capture.isOpened() is False:
    print("Error opening the video file!")
 
# Read until video is completed, or 'q' is pressed
while capture.isOpened():
    # Capture frame-by-frame from the video file
    ret, frame = capture.read()

    if ret is True:
        # Convert the frame from the video file to grayscale:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Histogram is computed
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        # Histograma acumulado
        cumulative_hist = np.cumsum(hist)
        
        #obtenemos las dimensiones de la imagen
        (M, N) = gray_frame.shape
        
        # Factor de cambio
        dx = (k-1)/(M*N)
        
        # Construimos un vector X y Y para almacenar los valores precalculados
        y2 = np.array([np.round(cumulative_hist[i] * dx)for i in range(256)], dtype='uint8')
        
        #
        image_equalized = y2[gray_frame]
        
        cv2.imshow('Original frame from the video file', image_equalized)
        
        #cv2.imshow('Original frame from the video file', frame)
 
        # Press q on keyboard to exit the program
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
 
# Release everything
capture.release()
cv2.destroyAllWindows()
