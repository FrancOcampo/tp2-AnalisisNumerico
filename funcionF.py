import numpy as np
import matplotlib.pyplot as plt
import cv2

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
    mascara_filtro[centro_fila + 30:centro_fila - 30, centro_columna + 30:centro_columna - 30] = 0

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

# def alternative2_