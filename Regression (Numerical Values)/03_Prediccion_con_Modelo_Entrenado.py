import tensorflow as tf
from tensorflow import keras
import numpy as np

# Cargar el modelo guardado
loaded_model = tf.keras.models.load_model('modelo_regresion.h5')

# Nuevos datos
habitaciones_nuevas = np.array([7.5, 8.5, 9.5, 10.5], dtype=float)

# Realizar una predicción con el modelo guardado
predicciones = loaded_model.predict(habitaciones_nuevas)

# Imprimir las predicciones
for i in range(len(habitaciones_nuevas)):
    print(f'Número de habitaciones: {habitaciones_nuevas[i]} => Precio estimado: {predicciones[i][0]}')

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
