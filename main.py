# Import the required packages
import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()

# We add 'video_path' argument using add_argument() including a help.
ap.add_argument("-v", "--video", required=True, help="path to the video file")
ap.add_argument("-o", "--output", required=False, help="path to the output video file")
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
    
# Obtener el tamaño de los frames y el FPS del video de entrada
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # Redimensionado por 0.4
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = capture.get(cv2.CAP_PROP_FPS)

# Crear el objeto VideoWriter para guardar el video procesado
output_video = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height), False)


#Funcion para crear la mascara
def mascara(frame):
    mask = np.zeros(frame.shape[:2], dtype='uint8')
    
    # Calcular el centro de la imagen
    (cX, cY) = (frame.shape[1] // 2, frame.shape[0] // 2)

    #Dimensiones de la mascara
    widthl = 400
    widthr = 600
    heightp = -79
    heightb = 300
    
    # Dibujamos la mascara
    cv2.rectangle(mask, (cX - widthl//2, (cY) - heightp//2), (cX + widthr//2, cY + heightb//2), 255, -1)
    
    # Aplicamos la mascara a la imagen
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)
    
    return masked_image
    
#Función para aplicar la correccion gamma
def gamma_correction(b,g,r, gamma):
    
    # Aplicar la corrección gamma a cada canal
    b_corrected = np.array(255 * (b / 255) ** gamma, dtype='uint8')
    g_corrected = np.array(255 * (g / 255) ** gamma, dtype='uint8')
    r_corrected = np.array(255 * (r / 255) ** gamma, dtype='uint8')
    
    # Combinar los canales corregidos de nuevo en una imagen BGR
    gamma_corrected = cv2.merge([b_corrected, g_corrected, r_corrected])
    
    return gamma_corrected


# Función para ecualizar el histograma en formato HSV
def equalize_histogram_hsv(frame, k):
    # Convertir la imagen de BGR a HSV
    image_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Separamos los canales de la imagen
    h, s, v = cv2.split(image_HSV)
    
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
    v_equalized = y2[v]
    
    image_HSV = cv2.merge([h, s, v_equalized])
    
    # Convertir la imagen de HSV de vuelta a BGR
    result = cv2.cvtColor(image_HSV, cv2.COLOR_HSV2BGR)
    
    return result

# Función para calcular el auto contraste restringido en formato HSV
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
    
    # Factor de cambio
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

# Programa principal
while capture.isOpened():
    # Capture frame-by-frame from the video file
    ret, frame = capture.read()

    if ret:
        # Redimensionamos el video
        frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
        cv2.imshow('Video original', frame)
        
        # Separar los canales del video
        b,g,r = cv2.split(frame) 
        
        # Aplicar la corrección gamma
        gamma_frame = gamma_correction(b,g,r, gamma)        
        cv2.imshow('Video con correccion gamma', gamma_frame)
        
        # Aplicamos y mostramos la ecualización del histograma
        equalized_image = equalize_histogram_hsv(gamma_frame, k)        
        cv2.imshow('Video ecualizado', equalized_image)
        
        # Aplicamos el auto contraste restringido
        image_contrast = ajustar_contraste_hsv(equalized_image, alow, ahigh, amin, amax)
        # Convertir el video contrastado a escala de grises
        gray_frame = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2GRAY)
        
        # Aplicamos y mostramos la mascara al video
        masked_frame = mascara(gray_frame)
        cv2.imshow('Video final', masked_frame)
        
        #Guadamos el video procesado
        output_video.write(masked_frame)
        
        # Press 'q' on keyboard to exit the program
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break
 
# Release everything
capture.release()
cv2.destroyAllWindows()
