# Import the required packages
import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()

# We add 'video_path' argument using add_argument() including a help.
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-k", "--k", required=True, help="k value")
ap.add_argument("-b", "--min", required=True, help="a_min value")
ap.add_argument("-a", "--max", required=True, help="a_max value")
args = vars(ap.parse_args())

# Create a VideoCapture object. In this case, the argument is the video file name:
capture = cv2.VideoCapture(args['video'])
k = float(args['k'])

# Valores para ajustar el contraste
amin = float(args['min'])
amax = float(args['max'])

# Check if the video is opened successfully
if not capture.isOpened():
    print("Error opening the video file!")
    exit()

# Read until video is completed, or 'q' is pressed
while capture.isOpened():
    # Capture frame-by-frame from the video file
    ret, frame = capture.read()

    if ret:
        # Convert the frame from the video file to grayscale:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        ### Auto Contraste ###
        # Obtenemos el valor máximo y mínimo de la imagen
        ahigh = gray_frame.max()
        alow = gray_frame.min()
        
        # Valor de cambio del contraste
        dx_contrast = (amax - amin) / (ahigh - alow)
        
        # Construimos un vector Y para almacenar los valores pre-calculados para el contraste
        y_contrast = np.array([amin + (i - alow) * dx_contrast for i in range(256)], dtype='uint8')
        
        # Aplicamos la función de auto contraste
        image_contrast = y_contrast[gray_frame]
        
        ### Ecualización del histograma ###
        # Calcular el histograma de la imagen original
        hist = cv2.calcHist([image_contrast], [0], None, [256], [0, 256])
        # Histograma acumulado
        cumulative_hist = np.cumsum(hist)
        
        # Obtenemos las dimensiones de la imagen
        (M, N) = image_contrast.shape
        
        # Factor de cambio para la ecualización
        dx_equalization = (k - 1) / (M * N)
        
        # Construimos un vector para la ecualización del histograma
        y_equalized = np.array([np.round(cumulative_hist[i] * dx_equalization) for i in range(256)], dtype='uint8')
        
        # Aplicamos la ecualización
        image_equalized = y_equalized[image_contrast]

        # Mostrar la imagen de auto contraste
        #cv2.imshow('Auto Contraste', image_contrast)
        
        # Mostrar la imagen ecualizada
        cv2.imshow('Histograma Ecualizado', image_equalized)
 
        # Press 'q' on keyboard to exit the program
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break
 
# Release everything
capture.release()
cv2.destroyAllWindows()
