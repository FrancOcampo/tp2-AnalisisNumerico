import numpy as np
import matplotlib.pyplot as plt
import cv2

# # Rutas completas a las imágenes (ajusta estas rutas según tu sistema)
# ruta_imagen1 = 'C:/Users/Miguel/Desktop/anumerico/AnalisisNumerico/tp2-AnalisisNumerico/im1_tp2.jpg'
# ruta_imagen2 = 'C:/Users/Miguel/Desktop/anumerico/AnalisisNumerico/tp2-AnalisisNumerico/im2_tp2.jpg'

# Leer las imágenes
imagen1 = cv2.imread("im1_tp2.jpg")
imagen2 = cv2.imread("im2_tp2.jpg")

# Convertir las imágenes a HSV
hsv_imagen1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2HSV)
hsv_imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2HSV)

# Umbralizar las imágenes
limite_inferior_amarillo = np.array([20, 75, 75])  # límite inferior para el amarillo
limite_superior_amarillo = np.array([30, 255, 255])  # límite superior para el amarillo

# Crear la máscara para el color amarillo
mascara1 = cv2.inRange(hsv_imagen1, limite_inferior_amarillo, limite_superior_amarillo)
mascara2 = cv2.inRange(hsv_imagen2, limite_inferior_amarillo, limite_superior_amarillo)

# Extraer el canal de saturación
saturacion1 = hsv_imagen1[:, :, 1]
saturacion2 = hsv_imagen2[:, :, 1]

# Aplicar la máscara
mascarilla1 = cv2.bitwise_and(saturacion1, saturacion1, mask=mascara1)
mascarilla2 = cv2.bitwise_and(saturacion2, saturacion2, mask=mascara2)

# Detectar los bordes
bordes1 = cv2.Sobel(mascara1, cv2.CV_64F, 1, 1, ksize=7)
bordes2 = cv2.Sobel(mascara2, cv2.CV_64F, 1, 1, ksize=7)  # impar para asegurar el píxel central

# Crear un elemento estructurante
kernel = np.ones((7,7), np.uint8)

# Dilatar los bordes
bordes_dilatados1 = cv2.dilate(bordes1, kernel, iterations=1)
bordes_dilatados2 = cv2.dilate(bordes2, kernel, iterations=1)

# Cerrar los bordes
bordes_cerrados1 = cv2.morphologyEx(bordes_dilatados1, cv2.MORPH_CLOSE, kernel)
bordes_cerrados2 = cv2.morphologyEx(bordes_dilatados2, cv2.MORPH_CLOSE, kernel)

# Encontrar los contornos
contornos1, _ = cv2.findContours(bordes_cerrados1.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contornos2, _ = cv2.findContours(bordes_cerrados2.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Rellenar el área
area_rellenada1 = cv2.drawContours(bordes_cerrados1, contornos1, -1, (255, 255, 255), thickness=cv2.FILLED)
area_rellenada2 = cv2.drawContours(bordes_cerrados2, contornos2, -1, (255, 255, 255), thickness=cv2.FILLED)

# Calcular el área amarilla
area_amarilla1 = np.sum(area_rellenada1 == 255)
area_total1 = imagen1.shape[0] * imagen1.shape[1]
proporcion_area_amarilla1 = area_amarilla1 / area_total1

area_amarilla2 = np.sum(area_rellenada2 == 255)
area_total2 = imagen2.shape[0] * imagen2.shape[1]
proporcion_area_amarilla2 = area_amarilla2 / area_total2

print(f"Área total amarilla en imagen 1: {area_amarilla1}, que es el {round(proporcion_area_amarilla1*100,2)}% del área total")
print(f"Área total amarilla en imagen 2: {area_amarilla2}, que es el {round(proporcion_area_amarilla2*100,2)}% del área total")

# Extraer el canal de valor
canal_valor1 = hsv_imagen1[:, :, 2]
canal_valor2 = hsv_imagen2[:, :, 2]

# Integrar el canal de valor
intensidad_amarilla1 = np.sum(canal_valor1[mascara1 == 255])
intensidad_amarilla2 = np.sum(canal_valor2[mascara2 == 255])

print(f"Intensidad total del amarillo en imagen 1: {intensidad_amarilla1}")
print(f"Intensidad total del amarillo en imagen 2: {intensidad_amarilla2}")

# Otra técnica: Transformada de Fourier
# Convertir la imagen a escala de grises
imagen_gris1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
imagen_gris2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)

# Aplicar la transformada de Fourier
transformada1 = np.fft.fft2(imagen_gris1)
transformada2 = np.fft.fft2(imagen_gris2)

transformada_shifted1 = np.fft.fftshift(transformada1)
transformada_shifted2 = np.fft.fftshift(transformada2)

espectro_magnitud1 = 20 * np.log(np.abs(transformada_shifted1))
espectro_magnitud2 = 20 * np.log(np.abs(transformada_shifted2))

filas, columnas = imagen_gris1.shape
fila_central, columna_central = filas // 2, columnas // 2

# Crear un filtro pasa-bajo (LPF) en el centro, las bajas frecuencias son donde el color es más uniforme.
mascara_lpf = np.ones((filas, columnas), np.uint8)
mascara_lpf[fila_central + 30:fila_central - 30, columna_central + 30:columna_central - 30] = 0

# Aplicar el filtro en el dominio de la frecuencia
transformada_shifted1 *= mascara_lpf
transformada_shifted2 *= mascara_lpf

# Invertir la Transformada de Fourier
transformada_invertida1 = np.fft.ifftshift(transformada_shifted1)
transformada_invertida2 = np.fft.ifftshift(transformada_shifted2)

imagen_filtrada1 = np.fft.ifft2(transformada_invertida1)
imagen_filtrada1 = np.abs(imagen_filtrada1)

imagen_filtrada2 = np.fft.ifft2(transformada_invertida2)
imagen_filtrada2 = np.abs(imagen_filtrada2)

# Mostrar la imagen filtrada 1
plt.imshow(imagen_filtrada1, cmap='gray')
plt.title('Imagen 1 Filtrada con HPF')
plt.show()

# Mejorar la máscara usando la información de la Transformada de Fourier 
mascara_mejorada = cv2.bitwise_and(mascara_lpf, mascara_lpf, mask=(imagen_filtrada1 > np.mean(imagen_filtrada1)).astype(np.uint8))

# Calcular el área del color amarillo mejorada 
area_amarilla_mejorada = np.sum(mascara_mejorada == 255)
proporcion_area_mejorada = area_amarilla_mejorada / area_total1

print('\n')
print(f"Área total amarilla en imagen 1: {area_amarilla1}, que es el {round(proporcion_area_amarilla1*100,2)}% del área total")
print(f"Área total amarilla en imagen 2: {area_amarilla2}, que es el {round(proporcion_area_amarilla2*100,2)}% del área total")

# Mostrar el espectro de magnitud 1 
plt.imshow(espectro_magnitud1, cmap='gray')
plt.title('Espectro de Magnitud Imagen 1')
plt.show()

# Asegurarse de que la máscara mejorada esté en formato correcto
mascara_mejorada = mascara_mejorada.astype(np.uint8)

# Multiplicar la máscara mejorada con la imagen original 
resultado = cv2.bitwise_and(imagen1, imagen1, mask=mascara1)

resultado2 = cv2.bitwise_and(imagen2, imagen2, mask=mascara2)
# Mostrar la imagen original y la imagen resultante 1
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(imagen1, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
plt.title('Área Amarilla Resaltada')
plt.show()

# Mostrar la imagen filtrada 2
plt.imshow(imagen_filtrada2, cmap='gray')
plt.title('Imagen 2 Filtrada con HPF')
plt.show()

# Mostrar el espectro de magnitud 2 
plt.imshow(espectro_magnitud2, cmap='gray')
plt.title('Espectro de Magnitud Imagen 2')
plt.show()

# Mostrar la imagen original y la imagen resultante 1
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(imagen2, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(resultado2, cv2.COLOR_BGR2RGB))
plt.title('Área Amarilla Resaltada')
plt.show()