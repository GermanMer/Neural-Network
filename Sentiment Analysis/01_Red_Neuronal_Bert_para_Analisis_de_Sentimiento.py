#Este ejemplo utiliza el modelo BERT pre-entrenado "bert-base-uncased" para realizar el análisis de sentimiento en los textos de ejemplo.
#BERT es un tipo de red neuronal profunda llamada "Transformer". Los Transformers son una arquitectura de red neuronal que ha demostrado un gran éxito en diversas tareas de procesamiento de lenguaje natural, incluido el análisis de sentimiento. BERT (Bidirectional Encoder Representations from Transformers) es una variante de los modelos Transformer que se pre-entrena en una gran cantidad de datos de texto y, posteriormente, se ajusta para tareas específicas, como el análisis de sentimiento.

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Crear un conjunto de datos ficticio
data = {
    'texto': [
        "Esta película es genial",
        "No me gustó esta película",
        "Una obra maestra del cine",
        "La peor película que he visto",
        "Una película asombrosa",
        "No la recomendaría a nadie",
        "Una película increíble",
        "No recomiendo esto en absoluto",
        "Una joya del cine",
        "Muy mala película",
        "Maravillosa película",
        "Horrible experiencia"
    ],
    'etiqueta': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

# Crear un DataFrame de pandas
df = pd.DataFrame(data)

# Guardar el DataFrame en un archivo CSV
df.to_csv('tu_conjunto_de_datos.csv', index=False)

# Cargar el conjunto de datos desde el archivo CSV
data = pd.read_csv('tu_conjunto_de_datos.csv')  # Reemplaza 'tu_conjunto_de_datos.csv' por el nombre de tu archivo

# Textos y etiquetas
textos = data['texto'].tolist()
etiquetas = data['etiqueta'].tolist()

# Cargar el modelo pre-entrenado de BERT
modelo_bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # 2 etiquetas: negativo y positivo
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Preprocesar y tokenizar los textos
secuencias = tokenizer(textos, padding='max_length', max_length=128, truncation=True, return_tensors='pt')

# Convertir etiquetas a tensores de torch
etiquetas = torch.tensor(etiquetas)

# Configurar el optimizador
optimizer = AdamW(modelo_bert.parameters(), lr=1e-5)

# Entrenar el modelo
modelo_bert.train()
for epoch in range(3):
    optimizer.zero_grad()
    outputs = modelo_bert(**secuencias, labels=etiquetas)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Época {epoch + 1}, Pérdida: {loss.item()}")

# Realizar predicciones en nuevos textos
nuevos_textos = ["Esta película es asombrosa", "No recomendaría esta película"]
secuencias_nuevas = tokenizer(nuevos_textos, padding='max_length', max_length=128, truncation=True, return_tensors='pt')

modelo_bert.eval()
with torch.no_grad():
    outputs = modelo_bert(**secuencias_nuevas)
    logits = outputs.logits
    probabilidades = torch.softmax(logits, dim=1)

etiquetas_predichas = torch.argmax(probabilidades, dim=1)
etiquetas_predichas = etiquetas_predichas.tolist()

# Decodificar las etiquetas predichas
sentimientos = ["positivo" if etiqueta == 1 else "negativo" for etiqueta in etiquetas_predichas]

# Resultados
for texto, sentimiento in zip(nuevos_textos, sentimientos):
    print(f"Texto: {texto} - Sentimiento: {sentimiento}")
