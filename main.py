import numpy as np
import matplotlib.pyplot as plt
import cv2

import funcionF

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

def analizar_imagen(ruta_imagen, numero_imagen):

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
    print(f"Área total amarilla en imagen {numero_imagen}: {area_amarilla}, que es el {round(proporcion_area_amarilla*100,2)}% del área total")

    # Integrar el canal de valor
    intensidad_amarilla = np.sum(v[mascara == 255])
    print(f"Intensidad total del amarillo en imagen {numero_imagen}: {intensidad_amarilla}")

    # Crear el resultado 
    resultado_mascara = cv2.bitwise_and(imagen, imagen, mask=mascara)
    imagen_con_mascara = cv2.bitwise_and(imagen, imagen, mask=area_rellenada)

    # Mostrar resultados
    mostrar_imagen([mascara_s, bordes, bordes_cerrados], ["Mascara en Saturacion", "Bordes", "Bordes cerrados"])
    comparar_resultados(imagen, imagen_con_mascara)

    return area_amarilla, imagen, mascara

def main():
    # Analizar la primera imagen
    area_amarilla1, imagen1, mascara1 = analizar_imagen("im1_tp2.jpg",1)
    area_amarilla2, imagen2, mascara2 = analizar_imagen("im2_tp2.jpg",2)
    
    if area_amarilla1 > area_amarilla2:
        print("La primera imagen tiene un área amarilla más grande.")
    else:
        print("La segunda imagen tiene un área amarilla más grande.")

    funcionF.alternativa1_fourier(imagen1, mascara1)
    funcionF.alternativa1_fourier(imagen2, mascara2)

main()
