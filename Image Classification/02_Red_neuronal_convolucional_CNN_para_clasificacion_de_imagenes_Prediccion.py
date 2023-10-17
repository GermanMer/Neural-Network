#Utilizamos la biblioteca TensorFlow y su interfaz de alto nivel Keras para crear una red neuronal convolucional (CNN) para la clasificación de imágenes.
#En este caso, usaremos el conjunto de datos MNIST, que consiste en imágenes de dígitos escritos a mano. El objetivo es clasificar cada imagen en uno de los 10 dígitos (0 al 9).
#Este código carga el conjunto de datos MNIST, crea una CNN simple, compila el modelo y lo entrena durante 5 épocas. Después, evalúa la precisión del modelo en el conjunto de prueba.

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2

# Cargar el conjunto de datos MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocesar los datos
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Crear el modelo CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Precisión en el conjunto de prueba: {test_acc}')

# Cargar una imagen de prueba (asegúrate de que tenga las mismas dimensiones que las imágenes de entrenamiento)
image_to_predict = cv2.imread('imagen_prueba_2.jpg', cv2.IMREAD_GRAYSCALE)  # Cargar en escala de grises
image_to_predict = cv2.resize(image_to_predict, (28, 28))  # Redimensionar a 28x28
image_to_predict = image_to_predict.astype('float32') / 255  # Normalizar
image_to_predict = np.expand_dims(image_to_predict, axis=0)  # Agregar una dimensión de lote

# Realizar una predicción
predictions = model.predict(image_to_predict)

# Las predicciones son un arreglo de probabilidades para cada clase
# Puedes obtener la clase con la mayor probabilidad utilizando argmax
predicted_class = np.argmax(predictions)

print(f'Clase predicha: {predicted_class}')

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
