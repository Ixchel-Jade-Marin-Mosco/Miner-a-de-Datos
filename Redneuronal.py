import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("clientes .csv")
print("Primeras filas del dataset:")
print(df.head())
#print("\nInformación del dataset:")
#print(df.info())

X = df[[
    "Compras_Mensuales",
    "Gasto_Mensual_MXN",
    "Visitas_Web_Mensuales",
    "Satisfaccion"
]]

y = df["Nivel_Consumo_Referencia"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

escalador = StandardScaler()

X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

modelo = MLPClassifier(
    hidden_layer_sizes=(8, 6),
    activation="relu",
    max_iter=1000,
    random_state=42
)

modelo.fit(X_train_escalado, y_train)

predicciones = modelo.predict(X_test_escalado)

exactitud = accuracy_score(y_test, predicciones)

print("\nExactitud del modelo:")
print(exactitud)
##Preguntas

1. ¿Qué predicción obtuvo cada cliente?
 R= Cliente 1 = bajo   Cliente 2 = alto  Cliente 3 =medio
2. ¿La predicción tiene sentido?
R= Si el primer cliente tiene pocas compras, un gasto mensual bajo, pocas visitas a la página y una satisfacción baja, por lo que se clasifica como bajo. Los clientes 2 y 3 presentan un gasto elevado, más compras y mayor actividad por lo que se clasifica como alto.
3. ¿Qué variable parece influir más?
La variable que parece influir más es Gasto_Mensual_MXN, 
4. ¿Qué pasa si aumentas el gasto mensual?
R=el modelo tiende a clasificar al cliente en un nivel de consumo más alto
5. ¿Qué pasa si disminuyes la satisfacción?
R=Ya no se obtiene un nivel de consumo alto y puede  reducirse ya que la satisfacción es una de las variables que el modelo considera para realizar la clasificación