#Utilizamos la biblioteca TensorFlow y su interfaz de alto nivel Keras para crear una red neuronal convolucional (CNN) para la clasificación de imágenes.
#En este caso, usaremos el conjunto de datos MNIST, que consiste en imágenes de dígitos escritos a mano. El objetivo es clasificar cada imagen en uno de los 10 dígitos (0 al 9).
#Este código carga el conjunto de datos MNIST, crea una CNN simple, compila el modelo y lo entrena durante 5 épocas. Después, evalúa la precisión del modelo en el conjunto de prueba.

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Cargar el conjunto de datos MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocesar los datos
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalizar los valores de píxeles al rango [0, 1]
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Codificar las etiquetas en formato one-hot
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

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
