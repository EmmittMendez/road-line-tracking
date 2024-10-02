# Import the required packages
import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()

# We add 'video_path' argument using add_argument() including a help.
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-k", "--k", required=True, help="k value")
ap.add_argument("-b", "--low", required=True, help="a_low value")   # Valor porcentual de a_low
ap.add_argument("-a", "--high", required=True, help="a_high value")  # Valor porcentual de a_high
ap.add_argument("-m", "--min", required=True, help="a_min value")   # Valor a min
ap.add_argument("-n", "--max", required=True, help="a_max value")   # Valor a max
args = vars(ap.parse_args())

# Create a VideoCapture object. In this case, the argument is the video file name:
capture = cv2.VideoCapture(args['video'])
k = float(args['k'])

# Valores para ajustar el contraste
alow = float(args['low'])
ahigh = float(args['high'])

# Valores del nuevo rango de contraste
amin = float(args['min'])
amax = float(args['max'])

# Check if the video is opened successfully
if not capture.isOpened():
    print("Error opening the video file!")
    exit()


# Función para calcular el contraste ajustado
def ajustar_contraste(gray_frame, alow, ahigh, amin, amax):
    # Calcular histograma
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    cumulative_hist = np.cumsum(hist)
    M, N = gray_frame.shape

    # Calcular a'low y a'high basados en los porcentajes dados
    multlow = int(M * N * alow)
    multhigh = int(M * N * (1 - ahigh))

    # Obtener los límites de intensidad (rango restringido)
    alow_candidates = [i for i in range(256) if cumulative_hist[i] >= multlow]
    ahigh_candidates = [i for i in range(256) if cumulative_hist[i] <= multhigh]

    # Si no hay candidatos válidos, usar 0 para alowp y 255 para ahighp por defecto
    alowp = min(alow_candidates) if alow_candidates else 0
    ahighp = max(ahigh_candidates) if ahigh_candidates else 255

    # Calcular la escala de mapeo
    if ahighp != alowp:
        dx = (amax - amin) / (ahighp - alowp)
    else:
        dx = 1  # Evitar división por cero si ahighp == alowp

    # Crear una tabla de mapeo con valores ajustados
    table_map = np.array([amin if i <= alowp else amax if i >= ahighp else amin + ((i - alowp) * dx) for i in range(256)], dtype='uint8')

    # Aplicar el mapeo a la imagen
    return table_map[gray_frame]


# Read until video is completed, or 'q' is pressed
while capture.isOpened():
    # Capture frame-by-frame from the video file
    ret, frame = capture.read()

    if ret:
        # Convert the frame from the video file to grayscale:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Ajustar el contraste de la imagen en escala de grises
        image_contrast = ajustar_contraste(gray_frame, alow, ahigh, amin, amax)
        # Mostrar la imagen ecualizada
        cv2.imshow('Histograma Ecualizado', image_contrast)
 
        # Press 'q' on keyboard to exit the program
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break
 
# Release everything
capture.release()
cv2.destroyAllWindows()
