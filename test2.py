# Import the required packages
import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()

# We add 'video_path' argument using add_argument() including a help.
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-k", "--k", required=True, help="k value")     # Valor para la ecualización
ap.add_argument("-g", "--gamma", required=True, help="gamma value")  # Valor para la corrección gamma
ap.add_argument("-b", "--low", required=True, help="a_low value")   # Valor porcentual de a_low
ap.add_argument("-a", "--high", required=True, help="a_high value")  # Valor porcentual de a_high
ap.add_argument("-m", "--min", required=True, help="a_min value")   # Valor a min
ap.add_argument("-n", "--max", required=True, help="a_max value")   # Valor a max
args = vars(ap.parse_args())

# Create a VideoCapture object. In this case, the argument is the video file name:
capture = cv2.VideoCapture(args['video'])
k = float(args['k'])
gamma = float(args["gamma"])

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
    
# Funcion para cambiar a formato hsv
def hsv(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    low_values = np.array([0, 0, 0])
    high_values = np.array([179, 255, 255])
    
    mask = cv2.inRange(hsv_frame, low_values, high_values)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result

# Función para aplicar la corrección gamma
def gamma_correction(image, gamma_value):
    x = np.linspace(0, 255, 256)
    y2 = np.array([((i / 255.0) ** gamma_value) * 255 for i in x], dtype='uint8')
    gamma = lambda m: y2[m]
    image_RGB_gamma = y2[image]
    return image_RGB_gamma
# def gamma_correction(image, gamma):
#     # Aplicar la corrección gamma
#     gamma_corrected = np.array(255*(image / 255) ** gamma, dtype='uint8')
#     return gamma_corrected


# Función para ecualizar el histograma
def equalize_histogram(gray_frame, k):
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
    
    return image_equalized

# Función para calcular el contraste ajustado
def ajustar_contraste(gray_frame, alow, ahigh, amin, amax):
    # Calcular histograma
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    cumulative_hist = np.cumsum(hist)
    # Obtenemos las dimensiones de la imagen
    (M, N) = gray_frame.shape

    # Obtenemos los valores de las condiciones para a'low y a'high
    multlow = int(M*N*alow)
    multhigh = int(M*N*(1-ahigh))

    print(multlow, multhigh)

    print(cumulative_hist)

    # Obtenemos a'low y a'high  (Rango de contraste restringido)
    alowp = min([i for i in range(256) if cumulative_hist[i] >= multlow])
    ahighp = max([i for i in range(256) if cumulative_hist[i] <= multhigh])

    dx = (amax - amin)/(ahighp - alowp)

    print("amax, amin, ahighp, alowp: ")
    print(amax, amin, ahighp, alowp)
    print("dx: ")
    print(dx)

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
        
        hsv_frame = hsv(frame)
        cv2.imshow('Video en formato HSV', hsv_frame)
        
        gamma_image = gamma_correction(gray_frame, gamma)        
        cv2.imshow('Video con correccion gamma', gamma_image)
        
        equalized_image = equalize_histogram(gamma_image, k)        
        # equalized_image = equalize_histogram(gray_frame, k)
        cv2.imshow('Video ecualizado', equalized_image)
        # Ajustar el contraste de la imagen en escala de grises
        image_contrast = ajustar_contraste(equalized_image, alow, ahigh, amin, amax)
        # Mostrar la imagen ecualizada
        cv2.imshow('Video con auto contraste restringido', image_contrast)
 
        # Press 'q' on keyboard to exit the program
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break
 
# Release everything
capture.release()
cv2.destroyAllWindows()
