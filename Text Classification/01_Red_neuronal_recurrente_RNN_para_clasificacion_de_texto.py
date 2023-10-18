#Red neuronal recurrente (RNN) con una capa de Embedding (para convertir secuencias de palabras en vectores densos) y una capa LSTM (Long Short-Term Memory) que es especialmente efectiva para procesar secuencias. Específicamente, se trata de una RNN unidireccional, ya que la capa LSTM procesa las secuencias de texto en una sola dirección, de izquierda a derecha.
#En este ejemplo, usaremos el conjunto de datos IMDB, que contiene revisiones de películas etiquetadas como positivas o negativas. El objetivo es clasificar las revisiones en estas dos categorías.
#Este ejemplo utiliza una capa de Embedding para convertir las secuencias de palabras en vectores de palabras densos, seguida de una capa LSTM para modelar las dependencias temporales en las secuencias de palabras. El modelo se compila para la clasificación binaria y se entrena en el conjunto de datos IMDB.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Cargar el conjunto de datos IMDB
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Preprocesar los datos
max_review_length = 200  # Longitud máxima de la revisión
train_data = pad_sequences(train_data, maxlen=max_review_length, padding='post', truncating='post')
test_data = pad_sequences(test_data, maxlen=max_review_length, padding='post', truncating='post')

# Construir el modelo de red neuronal
model = keras.Sequential()
model.add(Embedding(10000, 128, input_length=max_review_length)) #Capa de Embedding, para convertir secuencias de palabras en vectores densos. Los vectores densos son representaciones numéricas de las palabras que capturan relaciones semánticas entre ellas. En el ejemplo, se utiliza una capa de Embedding para convertir las palabras en vectores de 128 dimensiones.
model.add(LSTM(64)) #Capa LSTM, se utiliza para modelar las dependencias temporales en las secuencias de palabras en las revisiones de películas. La capa LSTM puede capturar relaciones a largo plazo en las secuencias, lo que la hace adecuada para tareas de procesamiento de lenguaje natural (NLP) como clasificación de texto.
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_data, train_labels, epochs=3, validation_data=(test_data, test_labels))

# Evaluar el modelo
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'Precisión en el conjunto de prueba: {test_acc:.2f}')

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
