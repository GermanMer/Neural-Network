###!!! Por una cuestión de caracteres especiales, para ejecutarlo ir a la carpeta D:/RN/

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Cargar el modelo previamente entrenado (opcional si no se guardó el modelo)
model = load_model('modelo_mnist.h5')

# Cargar la imagen de prueba (reemplaza 'imagen_prueba_2.jpg' por la ruta de tu imagen)
imagen = cv2.imread('imagen_prueba_2.jpg')

# Redimensionar la imagen a 28x28 píxeles
imagen = cv2.resize(imagen, (28, 28))

# Convertir la imagen a escala de grises
imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Normalizar los valores de píxeles al rango [0, 1]
imagen = imagen.astype('float32') / 255

# Agregar una dimensión de lote (batch dimension) para que sea compatible con el modelo
imagen = np.expand_dims(imagen, axis=0)

# Realizar una predicción con el modelo
predictions = model.predict(imagen)

# Las predicciones son un arreglo de probabilidades para cada clase
# Puedes obtener la clase con la mayor probabilidad utilizando argmax
predicted_class = np.argmax(predictions)

print(f'Clase predicha: {predicted_class}')

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
