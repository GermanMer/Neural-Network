#Red neuronal recurrente (RNN) para la predicción de series temporales.
#Utilizaremos una serie temporal ficticia para predecir valores futuros.
#Modelo RNN con dos capas LSTM, que se entrena para predecir valores futuros en la serie temporal y luego realiza predicciones en el conjunto de prueba y visualiza los resultados.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Generar una serie temporal ficticia
def generate_time_series():
    t = np.linspace(0, 30, 300)
    x = np.sin(t) + np.random.normal(0, 0.1, 300)
    return t, x

t, x = generate_time_series()

# Dividir la serie temporal en conjuntos de entrenamiento y prueba
split_time = 200
x_train, t_train = x[:split_time], t[:split_time]
x_test, t_test = x[split_time:], t[split_time:]

# Preparar los datos en ventanas de tiempo
window_size = 20
batch_size = 32

def create_time_series_dataset(series, window_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

train_dataset = create_time_series_dataset(x_train, window_size, batch_size)
test_dataset = create_time_series_dataset(x_test, window_size, batch_size)

# Construir un modelo RNN para la predicción de series temporales
model = keras.Sequential([
    keras.layers.LSTM(100, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(100, return_sequences=True),
    keras.layers.Dense(1)
])

# Compilar el modelo con métrica MSE
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# Entrenar el modelo
model.fit(train_dataset, epochs=10)

# Evaluar el modelo en el conjunto de prueba
mse = model.evaluate(test_dataset)

# Realizar predicciones en el conjunto de prueba
forecast = []
for time in range(len(x) - window_size):
    forecast.append(model.predict(x[time:time + window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]

print(f'Error Cuadrático Medio (MSE) en el conjunto de prueba: {mse[1]:.4f}')

# Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.plot(t_test, x_test, label='Serie Temporal Real')
plt.plot(t_test, results, label='Predicciones', color='red')
plt.legend()
plt.show()
