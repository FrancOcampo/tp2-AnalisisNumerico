import numpy as np
import matplotlib.pyplot as plt
import cv2

def sobel(image):
    # Definir las mascaras de Sobel
    Gx = np.array([[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]], dtype=np.float32)
    
    Gy = np.array([[-1, -2, -1], 
                   [0, 0, 0], 
                   [1, 2, 1]], dtype=np.float32)

    # Obtener las dimensiones de la imagen
    filas, columnas = image.shape

    # Inicializar las matrices de gradiente
    grad_x = np.zeros((filas, columnas), dtype=np.float32)
    grad_y = np.zeros((filas, columnas), dtype=np.float32)
    grad = np.zeros((filas, columnas), dtype=np.float32)

    # Aplicar los kernels a la imagen
    for i in range(1, filas-1):
        for j in range(1, columnas-1):
            region = image[i-1:i+2, j-1:j+2]
            grad_x[i, j] = np.sum(Gx * region)
            grad_y[i, j] = np.sum(Gy * region)

    # Calcular la magnitud del gradiente
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad = np.clip(grad, 0, 255)  # Limitar los valores entre 0 y 255
    grad = grad.astype(np.uint8)
    
    return grad

def mostrar_imagen(imagenes, titulos):
    for i, image in enumerate(imagenes):
        plt.subplot(1, len(imagenes), i+1)
        plt.imshow(image)
        # plt.imshow(image, cmap='gray') #con gris
        plt.title(titulos[i])
        plt.axis('off')
    plt.show()

def mostrar_imagenHSV(imagen):

    cv2.imshow('Imagen HSV', imagen)

    # Esperar a que se presione una tecla y luego cerrar las ventanas
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def comparar_resultados(imagen, resultado):
    # Mostrar la imagen original y la imagen resultante
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Original')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
    plt.title('Área Amarilla Resaltada')

    plt.show()

def gaussian_lowpass(filas, columnas, d0):
        # Calcular el centro de la imagen
        fila_central = filas // 2
        columna_central = columnas // 2
        lpf = np.zeros((filas, columnas))
        for i in range(filas):
            for j in range(columnas):
                distancia = np.sqrt((i - fila_central)**2 + (j - columna_central)**2)
                lpf[i, j] = np.exp(-distancia**2 / (2 * d0**2))
        return lpf

def gaussian_highpass(filas, columnas, d0):
    # Calcular el centro de la imagen
    fila_central = filas // 2
    columna_central = columnas // 2
    hpf = np.zeros((filas, columnas))
    for i in range(filas):
        for j in range(columnas):
            distancia = np.sqrt((i - fila_central)**2 + (j - columna_central)**2)
            hpf[i, j] = 1 - np.exp(-distancia**2 / (2 * d0**2))
    return hpf

def analizar_imagen(ruta_imagen):

    imagen = cv2.imread(ruta_imagen)

    # Convertir imagen a espacio hsv 
    hsv_imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Definir limites para el espacio H para el valor amarillo
    limite_inf_amarillo = np.array([20, 75, 75]) 
    limite_sup_amarillo = np.array([40, 255, 255]) 

    # Extraer canales H, S y V
    h = hsv_imagen[:, :, 0]
    s = hsv_imagen[:, :, 1]
    v = hsv_imagen[:, :, 2]

    # Crear mascara 
    mascara = cv2.inRange(hsv_imagen, limite_inf_amarillo, limite_sup_amarillo)

    # Aplicar mascara al canal de saturacion 
    mascara_s = cv2.bitwise_and(s, s, mask=mascara)

    # Detectar bordes
    bordes = sobel(mascara_s)

    # Crear un elemento estructurante
    kernel = np.ones((7,7), np.uint8)

    # Dilatar los bordes
    bordes_dilatados = cv2.dilate(bordes, kernel, iterations=1)

    # Cerrar los bordes
    bordes_cerrados = cv2.morphologyEx(bordes_dilatados, cv2.MORPH_CLOSE, kernel)

    # Encontrar los contornos
    contornos, _ = cv2.findContours(bordes_cerrados.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Rellenar el área
    area_rellenada = cv2.drawContours(bordes_cerrados, contornos, -1, 255, thickness=cv2.FILLED)

    # Calcular el area amarilla 
    area_amarilla = np.sum(area_rellenada == 255)
    area_total = imagen.shape[0] * imagen.shape[1]
    proporcion_area_amarilla = area_amarilla / area_total
    print(f"Área total amarilla en imagen 1: {area_amarilla}, que es el {round(proporcion_area_amarilla*100,2)}% del área total")

    # Integrar el canal de valor
    intensidad_amarilla = np.sum(v[mascara == 255])
    print(f"Intensidad total del amarillo en imagen 1: {intensidad_amarilla}")

    # Crear el resultado 
    resultado_mascara = cv2.bitwise_and(imagen, imagen, mask=mascara)
    imagen_con_mascara = cv2.bitwise_and(imagen, imagen, mask=area_rellenada)

    # Mostrar resultados
    mostrar_imagen([mascara_s, bordes, bordes_cerrados], ["Mascara en Saturacion", "Bordes", "Bordes cerrados"])
    comparar_resultados(imagen, imagen_con_mascara)

    return area_amarilla, imagen

def plot_spectrum(transformada):
    # Obtén las frecuencias correspondientes a los valores de la transformada de Fourier
    freqs = np.fft.fftfreq(len(transformada))

    # Calcula la magnitud de la transformada de Fourier
    magnitudes = np.abs(transformada)

    # Traza la magnitud en función de la frecuencia
    plt.plot(freqs, magnitudes)
    plt.xlabel('Frecuencia')
    plt.ylabel('Magnitud')
    plt.title('Espectro de la transformada de Fourier')
    plt.show()

def consignaF(imagen):

    gray_image1=cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Otra forma de suavizar la imagen es utilizando un filtro pasabajos en la transformada de fourier de las 3 matrices 
    # Filtro al canal de saturacion----------------

    transformada = np.fft.fft2(gray_image1) #aplicamos la transformada de fourier a la matriz de saturacion
    transformada_centrada = np.fft.fftshift(transformada) #centramos la transformada
    espectro_magnitud = 20 * np.log(np.abs(transformada_centrada)) #obtenemos el espectro de magnitud
    filas, columnas = gray_image1.shape #obtenemos las dimensiones de la matriz
    crow, ccol = filas // 2, columnas // 2 #obtenemos el centro de la matriz

    # New code to display the magnitude spectrum
    plt.figure(figsize=(6, 6))
    plt.imshow(espectro_magnitud, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.colorbar()
    plt.show()
    # Crear un filtro pasa-bajo (LPF) en el centro, las bajas frecuencias son donde el color es más uniforme.
    # Parámetros del filtro Gaussiano

    # Usa la función para trazar el espectro de la transformada de Fourier
    plot_spectrum(transformada_centrada)

    d0=30

    imagen_hp=gaussian_highpass(filas, columnas, d0)


    # Aplicar el filtro en el dominio de la frecuencia
    transformada_centrada *= imagen_hp


    # # Crear un filtro pasa-bajo (LPF) en el centro, las bajas frecuencias son donde el color es mas uniforme.
    # mascara = np.ones((filas, columnas), np.uint8)
    # mascara[crow+30:crow-30, ccol+30:ccol-30] = 0

    # # Aplicar el filtro en el dominio de la frecuencia
    # transformada_centrada *= mascara

    # Invertir la Transformada de Fourier
    f_ishift = np.fft.ifftshift(transformada_centrada)
    resultado = np.fft.ifft2(f_ishift)
    resultado = np.abs(resultado)
    resultado = resultado.astype(np.uint8)

    # # Mostrar la imagen filtrada
    # plt.imshow(img_back, cmap='gray')
    # plt.title('Imagen Filtrada con HPF')
    # plt.show()
    # Mostrar la imagen filtrada
    comparar_resultados(imagen, resultado)

def main():
    # Analizar la primera imagen
    area_amarilla1, imagen1 = analizar_imagen("im1_tp2.jpg")
    area_amarilla2, imagen2 = analizar_imagen("im2_tp2.jpg")
    
    if area_amarilla1 > area_amarilla2:
        print("La primera imagen tiene un área amarilla más grande.")
    else:
        print("La segunda imagen tiene un área amarilla más grande.")

    # consignaF(imagen)

main()
