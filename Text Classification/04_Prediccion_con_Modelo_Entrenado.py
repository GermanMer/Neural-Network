from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Cargar el modelo previamente entrenado
model = keras.models.load_model('modelo_imdb.h5')

# Preparar la revisión para la predicción (reemplaza con tu propio texto)
review = "Esta película es genial. Me encantó."

# Tokenizar y preprocesar la revisión
max_review_length = 200
tokenizer = Tokenizer(num_words=10000)
review_sequence = tokenizer.texts_to_sequences([review])
review_data = pad_sequences(review_sequence, maxlen=max_review_length, padding='post', truncating='post')

# Realizar una predicción
predicted_sentiment = model.predict(review_data)

# La predicción es un valor entre 0 y 1, donde 0 es negativo y 1 es positivo
print(f'Predicción de sentimiento: {predicted_sentiment[0, 0]:.2f}')

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
