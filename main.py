import csv
from transformers import pipeline

# Cargar el modelo pre-entrenado para análisis de sentimientos
analizador_sentimientos = pipeline("sentiment-analysis")

# Textos de ejemplo junto con los nombres de las personas
datos = [
    {"Nombre": "Usuario1", "Oración": "Me encanta este producto, es increíble."},
    {"Nombre": "Usuario2", "Oración": "No me gusta el servicio, es muy lento."},
    {"Nombre": "Usuario3", "Oración": "La película fue bastante aburrida."},
    {"Nombre": "Usuario4", "Oración": "¡Qué maravillosa sorpresa!"},
    {"Nombre": "Usuario5", "Oración": "Estoy muy decepcionado por el mal servicio recibido."}
]

# Definir umbral para considerar neutral
umbral_neutral = 0.2

# Crear o abrir el archivo CSV para escritura
with open('analisis_sentimientos.csv', mode='w', newline='', encoding='utf-8') as archivo_csv:
    campos = ['Nombre', 'Oración', 'Sentimiento']
    escritor_csv = csv.DictWriter(archivo_csv, fieldnames=campos)
    escritor_csv.writeheader()

    # Analizar sentimientos para cada oración
    for dato in datos:
        resultado = analizador_sentimientos(dato["Oración"])
        etiqueta_sentimiento = resultado[0]['label']
        probabilidad = resultado[0]['score']

        if etiqueta_sentimiento == 'POSITIVE':
            sentimiento = 'POSITIVO'
        elif etiqueta_sentimiento == 'NEGATIVE':
            sentimiento = 'NEGATIVO'
        elif probabilidad < umbral_neutral:
            sentimiento = 'NEUTRAL'
        else:
            sentimiento = 'INDEFINIDO (probabilidad: {})'.format(probabilidad)

        # Escribir en el archivo CSV
        escritor_csv.writerow({'Nombre': dato['Nombre'], 'Oración': dato['Oración'], 'Sentimiento': sentimiento})

print("Los resultados se han guardado en 'analisis_sentimientos.csv'.")
