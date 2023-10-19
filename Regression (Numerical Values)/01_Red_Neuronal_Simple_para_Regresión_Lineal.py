#Red neuronal simple con una sola neurona para realizar una regresión lineal.
#Utilizaremos un conjunto de datos de ejemplo para predecir el precio de una casa en función de su número de habitaciones.
#La red se compila con la función de pérdida de error cuadrático medio (MSE) y se entrena utilizando datos de ejemplo de número de habitaciones y precios. Luego, realizamos predicciones de precios para nuevas cantidades de habitaciones.

import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import mean_squared_error

# Generar datos de ejemplo (número de habitaciones y precio)
num_habitaciones = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
precio = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500, 550], dtype=float)

# Crear un modelo de regresión lineal
model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])  # Capa de una sola neurona
])

# Compilar el modelo
model.compile(optimizer='sgd', loss='mean_squared_error')  # Optimizador de descenso de gradiente estocástico

# Entrenar el modelo
model.fit(num_habitaciones, precio, epochs=1000)

# Realizar una predicción
habitaciones_nuevas = np.array([7.5, 8.5, 9.5, 10.5], dtype=float)
predicciones = model.predict(habitaciones_nuevas)

# Imprimir las predicciones
for i in range(len(habitaciones_nuevas)):
    print(f'Número de habitaciones: {habitaciones_nuevas[i]} => Precio estimado: {predicciones[i][0]}')

# Calcular el Error Cuadrático Medio (MSE)
mse = mean_squared_error(precio, model.predict(num_habitaciones))
print(f'Error Cuadrático Medio (MSE): {mse:.2f}')

# Agregar una espera para que la ventana no se cierre automáticamente en Windows
input("Presiona Enter para salir...")
