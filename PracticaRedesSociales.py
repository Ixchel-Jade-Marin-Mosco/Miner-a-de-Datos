import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

df = pd.read_csv("practica_mineria_textos_redes_sociales.csv")

df = df.dropna()
df["Texto_Limpio"] = df["Texto_Publicacion"].str.lower()

texto = " ".join(df["Texto_Limpio"])
palabras = re.findall(r'\w+', texto)
frecuencias = Counter(palabras)
print("\nTOP 10 PALABRAS")
for palabra, cantidad in frecuencias.most_common(10):
    print(palabra, ":", cantidad)

positivas = ["bueno", "excelente", "genial", "rápido"]
negativas = ["malo", "error", "terrible", "lento"]

def sentimiento(texto):
    texto = texto.lower()

    for palabra in positivas:
        if palabra in texto:
            return "positivo"

    for palabra in negativas:
        if palabra in texto:
            return "negativo"
    return "neutral"
df["Sentimiento_Calculado"] = df["Texto_Publicacion"].apply(sentimiento)

df["Sentimiento_Calculado"].value_counts().plot(kind="bar")
plt.title("Sentimientos")
plt.show()

print("\nANÁLISIS POR TEMA")
print(pd.crosstab(df["Tema_Referencia"],
                  df["Sentimiento_Calculado"]))

hashtags = []
for texto in df["Hashtags"]:
    hashtags.extend(re.findall(r"#\w+", str(texto)))

print("\nTOP 10 HASHTAGS")
for hashtag, cantidad in Counter(hashtags).most_common(10):
    print(hashtag, ":", cantidad)

aciertos = (
    df["Sentimiento_Calculado"]
    == df["Sentimiento_Referencia"]
).sum()

precision = aciertos / len(df) * 100
print("\nPrecisión:", round(precision, 2), "%")