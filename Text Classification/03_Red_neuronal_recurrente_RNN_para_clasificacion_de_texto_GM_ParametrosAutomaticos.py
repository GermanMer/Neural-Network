#Red neuronal recurrente (RNN) con una capa de Embedding (para convertir secuencias de palabras en vectores densos) y una capa LSTM (Long Short-Term Memory) que es especialmente efectiva para procesar secuencias. Específicamente, se trata de una RNN unidireccional, ya que la capa LSTM procesa las secuencias de texto en una sola dirección, de izquierda a derecha.
#En este ejemplo, usaremos el conjunto de datos IMDB, que contiene revisiones de películas etiquetadas como positivas o negativas. El objetivo es clasificar las revisiones en estas dos categorías.
#Este ejemplo utiliza una capa de Embedding para convertir las secuencias de palabras en vectores de palabras densos, seguida de una capa LSTM para modelar las dependencias temporales en las secuencias de palabras. El modelo se compila para la clasificación binaria y se entrena en el conjunto de datos IMDB.
#ESTE CÓDIGO ADEMÁS HACE UNA BUSQUEDA AUTOMÁTICA DE LOS MEJORES HIPERPARÁMETROS PARA EL MODELO

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
from keras_tuner.tuners import RandomSearch

# Cargar el conjunto de datos IMDB
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Preprocesar los datos
max_review_length = 200  # Longitud máxima de la revisión
train_data = pad_sequences(train_data, maxlen=max_review_length, padding='post', truncating='post')
test_data = pad_sequences(test_data, maxlen=max_review_length, padding='post', truncating='post')

# Dividir los datos de entrenamiento en entrenamiento y validación
x_train, x_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Función que define el modelo
def build_model(hp):
    model = keras.Sequential()
    model.add(Embedding(10000, hp.Int('embedding_dim', min_value=32, max_value=256, step=32), input_length=max_review_length))
    model.add(LSTM(hp.Int('lstm_units', min_value=32, max_value=128, step=32)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Configurar el sintonizador
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,  # Número de combinaciones de hiperparámetros a probar
    directory='my_dir',  # Directorio donde se guardarán los registros del sintonizador
    project_name='imdb_tuning'  # Nombre del proyecto de sintonización
)

# Realizar la búsqueda de hiperparámetros
tuner.search(x_train, y_train, epochs=3, validation_data=(x_val, y_val))

# Obtener los mejores hiperparámetros
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Imprimir los mejores hiperparámetros
print("Mejores hiperparámetros encontrados:")
print(f"Embedding Dim: {best_hps.get('embedding_dim')}")
print(f"LSTM Units: {best_hps.get('lstm_units')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")

# Construir y compilar el modelo con los mejores hiperparámetros
model = tuner.hypermodel.build(best_hps)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate')),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo con los mejores hiperparámetros
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'Precisión en el conjunto de prueba: {test_acc:.2f}')

# Guardar el modelo entrenado
model.save('modelo_imdb.h5')

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
