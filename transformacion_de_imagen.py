import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
imagen = cv2.imread('nueva.png')

# Obtener las dimensiones de la imagen
alto, ancho = imagen.shape[:2]

# Definir los parámetros de las transformaciones
# Rotación (en grados)
angulo_rotacion = 45
# Escalamiento (factor)
factor_escala = 2
# Cizallamiento (shearing)
cizallamiento_x = 0.5
cizallamiento_y = 0.2
# Translación (desplazamiento en píxeles)
desplazamiento_x = 50
desplazamiento_y = 30

# Aplicar la rotación
matriz_rotacion = cv2.getRotationMatrix2D((ancho / 2, alto / 2), angulo_rotacion, 1)
imagen_rotada = cv2.warpAffine(imagen, matriz_rotacion, (ancho, alto))

# Aplicar el escalamiento
imagen_escalada = cv2.resize(imagen, (int(ancho * factor_escala), int(alto * factor_escala)))

# Aplicar el cizallamiento
matriz_cizallamiento = np.float32([[1, cizallamiento_x, 0], [cizallamiento_y, 1, 0]])
imagen_cizallada = cv2.warpAffine(imagen, matriz_cizallamiento, (ancho, alto))

# Aplicar la translación
matriz_translacion = np.float32([[1, 0, desplazamiento_x], [0, 1, desplazamiento_y]])
imagen_transladada = cv2.warpAffine(imagen, matriz_translacion, (ancho, alto))

# Crear una imagen con transparencia (canal alpha)
image_with_alpha = cv2.cvtColor(imagen, cv2.COLOR_BGR2BGRA)

# Configurar la transparencia (canal alpha)
alpha = 50  # Puedes ajustar el nivel de transparencia (0 a 255)
image_with_alpha[:, :, 3] = alpha

 # Mostrar la imagen con transparencia utilizando matplotlib
plt.imshow(cv2.cvtColor(image_with_alpha, cv2.COLOR_BGRA2RGBA))
plt.title('Imagen con transparencia')
plt.axis('off')
plt.show()

 # Definir un ángulo de rotación personalizado (en grados)
angulo_rotacion = 45

    # Calcular la matriz de rotación con el ángulo personalizado
rotacion_matrix = cv2.getRotationMatrix2D((imagen.shape[1] // 2, imagen.shape[0] // 2), angulo_rotacion, 1)

    # Aplicar la rotación usando la matriz de rotación
image_rotacion = cv2.warpAffine(imagen, rotacion_matrix, (imagen.shape[2], imagen.shape[0]))

    # Mostrar la imagen con rotación utilizando matplotlib
plt.imshow(cv2.cvtColor(image_rotacion, cv2.COLOR_BGR2RGB))
plt.title('Imagen con rotación')
plt.axis('off')
plt.show()

# Mostrar las imágenes transformadas
cv2.imshow('Imagen Rotada', imagen_rotada)
cv2.imshow('Imagen Escalada', imagen_escalada)
cv2.imshow('Imagen Cizallada', imagen_cizallada)
cv2.imshow('Imagen Transladada', imagen_transladada)

cv2.waitKey(0)
cv2.destroyAllWindows()
