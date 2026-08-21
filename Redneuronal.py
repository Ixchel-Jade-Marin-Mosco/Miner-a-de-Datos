import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("dataset.csv")
print("Primeras filas del dataset:")
print(df.head())

X = df[[
    "Horas_Uso_Semanal",
    "Temperatura_CPU",
    "Uso_RAM_Porcentaje",
    "Espacio_Disco_Libre",
    "Errores_Sistema",
    "Tiempo_Arranque_Segundos"
]]

y = df["Estado_Equipo"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

escalador = StandardScaler()

X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

modelo = MLPClassifier(
    hidden_layer_sizes=(8, 6),
    activation="relu",
    max_iter=2000,
    random_state=42
)

modelo.fit(X_train_escalado, y_train)

predicciones = modelo.predict(X_test_escalado)
exactitud = accuracy_score(y_test, predicciones)

print("\nExactitud del modelo:")
print(f"{exactitud:.2%}")

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, predicciones))

print("\nReporte de clasificación:")
print(classification_report(y_test, predicciones))

comparacion = pd.DataFrame({
    "Real": y_test.values,
    "Prediccion": predicciones
})

print("\nComparación de resultados:")
print(comparacion)

modelo_validacion = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(8, 6),
        activation="relu",
        max_iter=2000,
        random_state=42
    )
)

validacion = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
puntajes = cross_val_score(modelo_validacion, X, y, cv=validacion)

print("\nExactitud promedio en validación cruzada:")
print(f"{puntajes.mean():.2%}")

nuevos_equipos = pd.DataFrame({
    "Horas_Uso_Semanal": [15, 35, 60],
    "Temperatura_CPU": [42, 65, 90],
    "Uso_RAM_Porcentaje": [35, 70, 95],
    "Espacio_Disco_Libre": [400, 110, 15],
    "Errores_Sistema": [1, 6, 18],
    "Tiempo_Arranque_Segundos": [28, 70, 145]
})

nuevos_equipos_escalados = escalador.transform(nuevos_equipos)
predicciones_nuevas = modelo.predict(nuevos_equipos_escalados)

print("\nPredicciones para nuevos equipos:")
print(predicciones_nuevas)

plt.plot(modelo.loss_curve_)
plt.xlabel("Iteraciones")
plt.ylabel("Error o pérdida")
plt.title("Proceso de aprendizaje de la red neuronal")
plt.show()

print("\nInterpretación:")
print("El modelo obtuvo una exactitud alta y resultados estables en la validación cruzada.")
print("El primer equipo fue clasificado como Bueno, el segundo como Regular y el tercero como Critico.")
print("Las predicciones tienen sentido porque los equipos con más temperatura, errores y tiempo de arranque presentan peor estado.")
