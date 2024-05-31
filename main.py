import funciones
import funcionF

def main():
    # # Analizar la primera imagen
    area_amarilla1, imagen1, mascara1 = funciones.analizar_imagen("im1_tp2.jpg", 1)
    area_amarilla2, imagen2, mascara2 = funciones.analizar_imagen("im2_tp2.jpg", 2)
    
    if area_amarilla1 > area_amarilla2:
        print("La primera imagen tiene un área amarilla más grande.")
    else:
        print("La segunda imagen tiene un área amarilla más grande.")

    funcionF.alternativa1_fourier(imagen1, mascara1, 1)
    funcionF.alternativa1_fourier(imagen2, mascara2, 2)

    funcionF.alternativa2_medianBlur("im1_tp2.jpg", 1)
    funcionF.alternativa2_medianBlur("im2_tp2.jpg", 2)

    funcionF.alternativa3("im2_tp2.jpg", 2)
    funcionF.alternativa3("im1_tp2.jpg", 1)

main()
