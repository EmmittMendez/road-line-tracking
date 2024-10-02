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


def mascara(frame):
    mask = np.zeros(frame.shape[:2], dtype='uint8')
    
    # Calcular el centro de la imagen
    (cX, cY) = (frame.shape[1] // 2, frame.shape[0] // 2)
    
    #width = frame.shape[1]
    # widthl = 220
    # widthr = 600
    # heightp = -90
    # heightb = 220
    widthl = 400
    widthr = 600
    heightp = -79
    heightb = 300
    
    # Dibujar un rectángulo en el centro de la máscara
    # cv2.rectangle(mask, (cX - width//2, (cY+40) - heightp//2), (cX + width//2, cY + heightb//2), 255, -1)
    cv2.rectangle(mask, (cX - widthl//2, (cY) - heightp//2), (cX + widthr//2, cY + heightb//2), 255, -1)
    # cv2.rectangle(mask, (0, cY - heightp // 2), (width, cY), 255, -1)
    
    # Aplicar la máscara a la imagen
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)
    
    return masked_image
    
#Función para aplicar la corrección gamma
def gamma_correction(b,g,r, gamma):
    
    # Aplicar la corrección gamma a cada canal
    b_corrected = np.array(255 * (b / 255) ** gamma, dtype='uint8')
    g_corrected = np.array(255 * (g / 255) ** gamma, dtype='uint8')
    r_corrected = np.array(255 * (r / 255) ** gamma, dtype='uint8')
    
    # Combinar los canales corregidos de nuevo en una imagen BGR
    gamma_corrected = cv2.merge([b_corrected, g_corrected, r_corrected])
    
    return gamma_corrected


# Función para ecualizar el histograma
def equalize_histogram_hsv(frame, k):
    # Convertir la imagen de BGR a HSV
    image_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Trabajar con el canal V (Value)
    h, s, v = cv2.split(image_HSV)
    #v_channel = image_HSV[:, :, 2]
    
    # Calcular histograma
    hist = cv2.calcHist([v], [0], None, [256], [0, 256])
    cumulative_hist = np.cumsum(hist)
    
    # Obtenemos las dimensiones de la imagen
    (M, N) = v.shape
    
    # Factor de cambio
    dx = (k - 1) / (M * N)
    
    # Construimos un vector Y para almacenar los valores precalculados
    y2 = np.array([np.round(cumulative_hist[i] * dx) for i in range(256)], dtype='uint8')
    
    # Aplicar la ecualización al canal V
    v_channel_equalized = y2[v]
    
    # Reemplazar el canal V ecualizado en la imagen HSV
    image_HSV[:, :, 2] = v_channel_equalized
    
    # Convertir la imagen de HSV de vuelta a BGR
    result = cv2.cvtColor(image_HSV, cv2.COLOR_HSV2BGR)
    
    return result

# Función para calcular el auto contraste restringido
def ajustar_contraste_hsv(frame, alow, ahigh, amin, amax):

    # Convertir la imagen de BGR a HSV
    image_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Trabajar con el canal V (Value)
    h, s, v = cv2.split(image_HSV)
    
    # Calcular histograma
    hist = cv2.calcHist([v], [0], None, [256], [0, 256])
    cumulative_hist = np.cumsum(hist)
    
    # Obtenemos las dimensiones de la imagen
    (M, N) = v.shape
    
    # Obtenemos los valores de las condiciones para a'low y a'high
    multlow = int(M * N * alow)
    multhigh = int(M * N * (1 - ahigh))
    
    # Obtenemos a'low y a'high  (Rango de contraste restringido)
    alowp = min([i for i in range(256) if cumulative_hist[i] >= multlow])
    ahighp = max([i for i in range(256) if cumulative_hist[i] <= multhigh])
    
    dx = (amax - amin) / (ahighp - alowp)
    
    # Crear una tabla de mapeo con valores ajustados
    table_map = np.array([amin if i <= alowp else amax if i >= ahighp else amin + ((i - alowp) * dx) for i in range(256)], dtype='uint8')
    
    # Aplicar el mapeo al canal V
    v_correct = table_map[v]
    
    # Reemplazar el canal V ajustado en la imagen HSV
    image_HSV = cv2.merge([h, s, v_correct])
    
    # Convertir la imagen de HSV de vuelta a BGR
    result = cv2.cvtColor(image_HSV, cv2.COLOR_HSV2BGR)
    
    return result


# Read until video is completed, or 'q' is pressed
while capture.isOpened():
    # Capture frame-by-frame from the video file
    ret, frame = capture.read()

    if ret:
        frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
        
        b,g,r = cv2.split(frame) 
        
        gamma_frame = gamma_correction(b,g,r, gamma)        
        cv2.imshow('Video con correccion gamma', gamma_frame)
        
        
        
        equalized_image = equalize_histogram_hsv(gamma_frame, k)        
        # equalized_image = equalize_histogram(gray_frame, k)
        cv2.imshow('Video ecualizado', equalized_image)
        # Ajustar el contraste de la imagen en escala de grises
        image_contrast = ajustar_contraste_hsv(equalized_image, alow, ahigh, amin, amax)
        
        gray_frame = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2GRAY)
        # Mostrar la imagen ecualizada
        masked_frame = mascara(gray_frame)
        cv2.imshow('Video con auto contraste restringido', masked_frame)
        masked_frame2 = mascara(frame)
        cv2.imshow('Video con mascara', masked_frame2)
        
 
        # Press 'q' on keyboard to exit the program
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break
 
# Release everything
capture.release()
cv2.destroyAllWindows()
