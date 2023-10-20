#Red neuronal autoencoder para la detección de anomalías en datos.
#En este ejemplo, primero generamos datos normales y luego introducimos datos con anomalías. Luego, creamos un autoencoder simple con capas de codificación y decodificación. Entrenamos el autoencoder utilizando los datos combinados.
#Después de entrenar el autoencoder, calculamos el error de reconstrucción (Error Cuadrático Medio, MSE) para cada muestra. Finalmente, establecemos un umbral para detectar anomalías, en este caso, el 5% superior de los errores. Las muestras con errores de reconstrucción por encima de ese umbral se consideran anomalías.

import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generar datos normales (sin anomalías)
data_normal = np.random.normal(0, 1, (1000, 10))

# Introducir anomalías
data_anomaly = np.random.normal(10, 2, (20, 10))

# Combinar datos normales y datos con anomalías
data = np.vstack((data_normal, data_anomaly))
np.random.shuffle(data)

# Crear etiquetas (0 para datos normales, 1 para anomalías)
labels = np.zeros(len(data))
labels[len(data_normal):] = 1

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Construir un autoencoder más complejo
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss='mse')

# Entrenar el autoencoder
model.fit(x_train, x_train, epochs=50, batch_size=32, validation_data=(x_test, x_test))

# Realizar predicciones y calcular el error de reconstrucción
train_predictions = model.predict(x_train)
test_predictions = model.predict(x_test)
mse_train = np.mean(np.square(x_train - train_predictions), axis=1)
mse_test = np.mean(np.square(x_test - test_predictions), axis=1)

# Calcular la métrica AUC-ROC en el conjunto de prueba
auc_roc = roc_auc_score(y_test, mse_test)

# Establecer un umbral para detectar anomalías
threshold = np.percentile(mse_test, 95)  # Por ejemplo, el 5% superior de los errores

# Identificar anomalías en el conjunto de prueba
anomalies = x_test[mse_test > threshold]

print(f'Número de anomalías detectadas: {len(anomalies)}')
print(f'AUC-ROC en el conjunto de prueba: {auc_roc:.2f}')

# Gráfico de dispersión con anomalías resaltadas en rojo
plt.scatter(range(len(mse_test)), mse_test, c=['g' if mse < threshold else 'r' for mse in mse_test])
plt.xlabel('Índice de Muestra')
plt.ylabel('Error de Reconstrucción (MSE)')
plt.title('Detección de Anomalías')
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Anomalías')],
           loc='upper right')
plt.show()
