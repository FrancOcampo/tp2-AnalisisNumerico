# Python
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Your code goes here
image1 = cv2.imread('im1_tp2.jpg')
image2 = cv2.imread('im2_tp2.jpg')

# Convert the image to hsv
hsv_im1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
hsv_im2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

# ----------------------------------------------------------------
# #Splitting the channels
# h1, s1, v1 = cv2.split(hsv_im1)
# h2, s2, v2 = cv2.split(hsv_im2)

# #showing the images
#     #image1
# cv2.imshow('hue 1', h1)
# cv2.imshow('saturation 1', s1)
# cv2.imshow('value 1', v1)
#     #image2
# cv2.imshow('hue 2', h2)
# cv2.imshow('saturation 2', s2)
# cv2.imshow('value 2', v2)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ----------------------------------------------------------------
# threshold the images
lower_yellow = np.array([20, 75, 75]) #lower bound for yellow 20 from 0 to 180, 100 from 0 to 255, 100 from 0 to 255, 100 from 0 to 255
upper_yellow = np.array([30, 255, 255]) #upper bound for yellow 30 from 0 to 180, 255 from 0 to 255, 255 from 0 to 255, 255 from 0 to 255

#mask the yellow color
mask1 = cv2.inRange(hsv_im1, lower_yellow, upper_yellow)
mask2 = cv2.inRange(hsv_im2, lower_yellow, upper_yellow)

#3)----extract the saturation channel
saturation1 = hsv_im1[:,:,1]
saturation2 = hsv_im2[:,:,1]

#apply the mask
masked1 = cv2.bitwise_and(saturation1, saturation1, mask = mask1)
masked2 = cv2.bitwise_and(saturation2, saturation2, mask = mask2)



#detect the edges
edges1= cv2.Sobel(masked1, cv2.CV_64F, 1, 1, ksize=7)
edges2= cv2.Sobel(masked2, cv2.CV_64F, 1, 1, ksize=7)# odd to ensure the center pixel



#4)----create structuring element
kernel = np.ones((7,7), np.uint8)

#dilate the edges
dilated1 = cv2.dilate(edges1, kernel, iterations=1)
dilated2 = cv2.dilate(edges2, kernel, iterations=1)


#Close the edges
closed1 = cv2.morphologyEx(dilated1, cv2.MORPH_CLOSE, kernel)
closed2 = cv2.morphologyEx(dilated2, cv2.MORPH_CLOSE, kernel)


# cv2.imshow('masked1', closed1)
# cv2.imshow('masked2', closed2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#5)----find the contours
contours1, _ = cv2.findContours(closed1.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(closed2.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#fill the area
filled_image1 = cv2.drawContours(closed1, contours1, -1, (255, 255, 255), thickness=cv2.FILLED)
filled_image2 = cv2.drawContours(closed2, contours2, -1, (255, 255, 255), thickness=cv2.FILLED)

# cv2.imshow('masked1', filled_image1)
# cv2.imshow('masked2', filled_image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#calculate the yellow area
yellow_area1 = np.sum(filled_image1 == 255)
total_area1 = image1.shape[0] * image1.shape[1]
yellow_area_ratio1 = yellow_area1 / total_area1

yellow_area2 = np.sum(filled_image2 == 255)
total_area2 = image2.shape[0] * image2.shape[1]
yellow_area_ratio2 = yellow_area2 / total_area2

print(f"total yellow area in image 1: {yellow_area1}, it is its {round(yellow_area_ratio1*100,2)}% of the total area")
print(f"total yellow area in image 2: {yellow_area2}, it is its {round(yellow_area_ratio2*100,2)}% of the total area")

#6)----extract the value channel
value1_channel = hsv_im1[:,:,2]
value2_channel = hsv_im2[:,:,2]

#integrate the value channel
yellow_intensity1= np.sum(value1_channel[mask1==255])
yellow_intensity2= np.sum(value2_channel[mask2==255])

print(f"Total Intensity of yellow in image 1: {yellow_intensity1}")
print(f"Total Intensity of yellow in image 2: {yellow_intensity2}")

#7)----other Technique FOURIER TRANSFORM
#convert the image to gray
gray_image1=cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2=cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#apply fourier transform
f_transform1 = np.fft.fft2(gray_image1)
f_transform2 = np.fft.fft2(gray_image2)

f_transform1_shifted = np.fft.fftshift(f_transform1)
f_transform2_shifted = np.fft.fftshift(f_transform2)

magnitude_spectrum1 = 20*np.log(np.abs(f_transform1_shifted))
magnitude_spectrum2 = 20*np.log(np.abs(f_transform2_shifted))

rows, cols = gray_image1.shape
crow, ccol = rows // 2, cols // 2

# Crear un filtro pasa-bajo (LPF) en el centro, las bajas frecuencias son donde el color es mas uniforme.
mask = np.ones((rows, cols), np.uint8)
mask[crow+30:crow-30, ccol+30:ccol-30] = 0

# Aplicar el filtro en el dominio de la frecuencia
f_transform1_shifted *= mask

# Invertir la Transformada de Fourier
f_ishift = np.fft.ifftshift(f_transform1_shifted)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Mostrar la imagen filtrada
plt.imshow(img_back, cmap='gray')
plt.title('Imagen Filtrada con HPF')
plt.show()

# Mejorar la máscara usando la información de TF
improved_mask = cv2.bitwise_and(mask, mask, mask=(img_back > np.mean(img_back)).astype(np.uint8))

# Calcular el área del color amarillo mejorada
yellow_area = np.sum(improved_mask == 255)
total_area = image1.shape[0] * image1.shape[1]
yellow_area_ratio = yellow_area / total_area
print('\n')
print(f"total yellow area in image 1: {yellow_area1}, it is its {round(yellow_area_ratio1*100,2)}% of the total area")
print(f"total yellow area in image 2: {yellow_area2}, it is its {round(yellow_area_ratio2*100,2)}% of the total area")


#show the magnitude spectrum
plt.imshow(magnitude_spectrum1, cmap='gray')
plt.title('Magnitude Spectrum Image 1')
plt.show()

# Asegurarse de que la máscara mejorada esté en formato correcto
improved_mask = improved_mask.astype(np.uint8)

# Multiplicar la máscara mejorada con la imagen original
# result = cv2.bitwise_and(image1, image1, mask=improved_mask)
result = cv2.bitwise_and(image1, image1, mask=mask1)

# Mostrar la imagen original y la imagen resultante
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Área Amarilla Resaltada')

plt.show()
