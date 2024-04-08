import csv
from gensim import corpora, models
from textblob import TextBlob

# Textos de ejemplo junto con los nombres de las personas
datos = [
    {"Oración": "Me encanta este producto, es increíble."},
    {"Oración": "No me gusta el servicio, es muy lento."},
    {"Oración": "La película fue bastante aburrida."},
    {"Oración": "¡Qué maravillosa sorpresa!"},
    {"Oración": "Estoy muy decepcionado por el mal servicio recibido."}
]

# Preprocesamiento de texto
texto_preprocesado = []
for dato in datos:
    texto_preprocesado.append(TextBlob(dato["Oración"]).words.lower())

# Crear diccionario y corpus de palabras
diccionario = corpora.Dictionary(texto_preprocesado)
corpus = [diccionario.doc2bow(text) for text in texto_preprocesado]

# Entrenar modelo LDA
modelo_lda = models.LdaModel(corpus, num_topics=1, id2word=diccionario)

# Obtener el tema principal
tema_principal = modelo_lda.print_topics(num_topics=1, num_words=3)[0][1]

# Crear o abrir el archivo CSV para escritura
with open('temas_principales.csv', mode='w', newline='', encoding='utf-8') as archivo_csv:
    campos = ['Oración', 'Tema', 'Porcentaje de favorabilidad']
    escritor_csv = csv.DictWriter(archivo_csv, fieldnames=campos)
    escritor_csv.writeheader()

    # Escribir en el archivo CSV
    for dato in datos:
        porcentaje_favorabilidad = TextBlob(dato["Oración"]).sentiment.polarity
        escritor_csv.writerow({'Oración': dato['Oración'], 'Tema': tema_principal, 'Porcentaje de favorabilidad': porcentaje_favorabilidad})

print("Los resultados se han guardado en 'temas_principales.csv'.")
