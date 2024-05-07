import pandas as pd
from bertopic import BERTopic

# Cargar el modelo BERTopic preentrenado
topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")

# Obtener información sobre los temas del modelo
topic_info = topic_model.get_topic_info()

# Extraer las clasificaciones únicas
classifications = topic_info['Name'].tolist()

# Convertir la lista de clasificaciones en un DataFrame
classifications_df = pd.DataFrame({"classification": classifications})

# Guardar el DataFrame en un archivo CSV
classifications_df.to_csv("clasificaciones.csv", index=False)
