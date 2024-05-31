import numpy as np
import matplotlib.pyplot as plt
import cv2

import funciones

def alternativa1_fourier(imagen, mascara, numero_imagen):
    # convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # aplicar la transformada de Fourier
    transformada_f = np.fft.fft2(imagen_gris)

    transformada_f_desplazada = np.fft.fftshift(transformada_f)

    espectro_magnitud = 20 * np.log(np.abs(transformada_f_desplazada))

    filas, columnas = imagen_gris.shape
    centro_fila, centro_columna = filas // 2, columnas // 2

    # Crear un filtro pasa-bajo (LPF) en el centro, las bajas frecuencias son donde el color es más uniforme.
    mascara_filtro = np.ones((filas, columnas), np.uint8)
    mascara_filtro[0:centro_fila + 30, centro_columna + 30: columnas] = 0

    # Aplicar el filtro en el dominio de la frecuencia
    transformada_f_desplazada *= mascara_filtro

    # Invertir la Transformada de Fourier
    transformada_inversa = np.fft.ifftshift(transformada_f_desplazada)
    imagen_inversa = np.fft.ifft2(transformada_inversa)
    imagen_inversa = np.abs(imagen_inversa)

    # Mostrar la imagen filtrada
    plt.imshow(imagen_inversa, cmap='gray')
    plt.title('Imagen Filtrada con HPF')
    plt.show()

    # Mejorar la máscara usando la información de TF
    mascara_mejorada = cv2.bitwise_and(mascara_filtro, mascara_filtro, mask=(imagen_inversa > np.mean(imagen_inversa)).astype(np.uint8))
    mascara_mejorada = mascara_mejorada.astype(np.uint8) * 255

    funciones.mostrar_imagen([mascara, mascara_mejorada], ['Máscara original', 'Máscara mejorada'])

    # Calcular el área del color amarillo mejorada
    area_amarilla = np.sum(mascara_mejorada == 255)
    area_total = imagen.shape[0] * imagen.shape[1]
    ratio_area_amarilla = area_amarilla / area_total
    print('\n')
    print(f"Area total amarilla de la imagen {numero_imagen}: {area_amarilla}, es decir {round(ratio_area_amarilla*100,2)}% del área total")

    # mostrar el espectro de magnitud
    plt.imshow(espectro_magnitud, cmap='gray')
    plt.title(f'Imagen en el espectro de magnitud {numero_imagen}')
    plt.show()

    # Asegurarse de que la máscara mejorada esté en el formato correcto
    mascara_mejorada = mascara_mejorada.astype(np.uint8)

    # Multiplicar la máscara mejorada con la imagen original
    # result = cv2.bitwise_and(image1, image1, mask=improved_mask)
    resultado = cv2.bitwise_and(imagen, imagen, mask=mascara)

    # Mostrar la imagen original y la imagen resultante
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Original')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
    plt.title('Área Amarilla Resaltada')

    plt.show()

def alternativa2_medianBlur(ruta_imagen, numero_imagen):

    imagen = cv2.imread(ruta_imagen)

    # Convertir imagen a espacio hsv 
    hsv_imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Definir limites para el espacio H para el valor amarillo
    limite_inf_amarillo = np.array([20, 75, 75]) 
    limite_sup_amarillo = np.array([40, 255, 255]) 

    # Extraer canales H, S y V
    h = hsv_imagen[:, :, 0]
    s = hsv_imagen[:, :, 1]
    v = hsv_imagen[:, :, 2]

    imagen_gris_suavizada= cv2.medianBlur(imagen_gris, 7, 0)

    funciones.mostrar_imagen_gris([imagen_gris, imagen_gris_suavizada], ["Imagen en escala de grises", "Imagen suavizada"])

    # Crear mascara 
    mascara = cv2.inRange(hsv_imagen, limite_inf_amarillo, limite_sup_amarillo)

    # Aplicar mascara al canal de saturacion 
    mascara_s = cv2.bitwise_and(imagen_gris_suavizada, imagen_gris_suavizada, mask=mascara)

    # Detectar bordes
    bordes = funciones.sobel(mascara_s)

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
    print(f"Área total amarilla en imagen {numero_imagen}: {area_amarilla}, que es el {round(proporcion_area_amarilla*100,2)}% del área total")

    # Integrar el canal de valor
    intensidad_amarilla = np.sum(v[mascara == 255])
    print(f"Intensidad total del amarillo en imagen {numero_imagen}: {intensidad_amarilla}")

    # Crear el resultado 
    resultado_mascara = cv2.bitwise_and(imagen, imagen, mask=mascara)
    imagen_con_mascara = cv2.bitwise_and(imagen, imagen, mask=area_rellenada)

    # Mostrar resultados
    funciones.mostrar_imagen([mascara_s, bordes, bordes_cerrados], ["Mascara en Saturacion", "Bordes", "Bordes cerrados"])
    funciones.comparar_resultados(imagen, imagen_con_mascara)

    return area_amarilla, imagen, mascara

def alternativa3(ruta_imagen, numero_imagen):
    imagen = cv2.imread(ruta_imagen)

    # Convertir imagen a espacio hsv 
    hsv_imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Definir limites para el espacio H para el valor amarillo
    limite_inf_amarillo = np.array([20, 70, 70]) 
    limite_sup_amarillo = np.array([40, 255, 255])

    # Create a mask for the yellow color
    mask = cv2.inRange(hsv_imagen, limite_inf_amarillo, limite_sup_amarillo)

    # Apply edge detection to the mask
    edges = cv2.Canny(mask, 250, 300)

    # Blur the edges
    blurred_edges = cv2.GaussianBlur(edges, (5, 5), 0)

    # Apply the blurred edges as a mask to the original image
    res = cv2.bitwise_and(imagen, imagen, mask=blurred_edges)

    funciones.mostrar_imagen([imagen, res], ['Imagen original', 'Imagen con bordes amarillos'])   
    funciones.mostrar_imagen([edges, blurred_edges], ['Bordes', 'Bordes suavizados'])

    # Crear un elemento estructurante
    kernel = np.ones((7,7), np.uint8)

    # Dilatar los bordes
    bordes_dilatados = cv2.dilate(blurred_edges, kernel, iterations=1)

    # Cerrar los bordes
    bordes_cerrados = cv2.morphologyEx(bordes_dilatados, cv2.MORPH_CLOSE, kernel)

    # Encontrar los contornos
    contornos, _ = cv2.findContours(bordes_cerrados.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Rellenar el área
    area_rellenada = cv2.drawContours(bordes_cerrados, contornos, -1, 255, thickness=cv2.FILLED)

    funciones.mostrar_imagen([bordes_cerrados, area_rellenada], ['Bordes cerrados', 'Área rellenada'])    

    # Calcular el area amarilla 
    area_amarilla = np.sum(area_rellenada == 255)
    area_total = imagen.shape[0] * imagen.shape[1]
    proporcion_area_amarilla = area_amarilla / area_total
    print(f"Área total amarilla en imagen {numero_imagen}: {area_amarilla}, que es el {round(proporcion_area_amarilla*100,2)}% del área total")

    # Count the number of yellow pixels
    yellow_pixels = np.count_nonzero(blurred_edges)

    # Calculate the total number of pixels
    total_pixels = imagen.shape[0] * imagen.shape[1]

    # Calculate the percentage of yellow pixels
    yellow_percentage = (yellow_pixels / total_pixels) * 100

    print(f"Number of yellow pixels in image {numero_imagen}: {yellow_pixels}")
    print(f"Percentage of yellow pixels in image {numero_imagen}: {yellow_percentage}%")


