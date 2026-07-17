import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression

df = pd.read_csv("dataset_sucursales_mensual.csv") 

print("Medidas descriptivas:")
print(df.describe(include="all"))

variables = ["Publicidad","Clientes","Satisfaccion","Descuento","Ventas_Totales"]

print("\nCorrelaciones con Ventas_Totales:")
for var in variables[:-1]:
    corr = df[var].corr(df["Ventas_Totales"])
    print(f"{var} ↔ Ventas_Totales: {corr:.3f}")

corr_matrix = df[variables].corr()

fig, ax = plt.subplots(figsize=(8,6))
cax = ax.matshow(corr_matrix, cmap="coolwarm")
plt.xticks(range(len(variables)), variables, rotation=90)
plt.yticks(range(len(variables)), variables)
fig.colorbar(cax)
plt.title("Matriz de correlación", pad=20)
plt.show()

X = df[["Publicidad","Clientes","Satisfaccion","Descuento"]]
y = df["Cumplio_Meta"].map({"Sí":1,"No":0})  # Convertir a binario

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
print("\nÁrbol de decisión:")
print(classification_report(y_test, tree.predict(X_test)))

plt.figure(figsize=(12,8))
plot_tree(tree, feature_names=X.columns, class_names=["No","Sí"], filled=True)
plt.show()

nb = GaussianNB()
nb.fit(X_train, y_train)
print("\nNaive Bayes:")
print(classification_report(y_test, nb.predict(X_test)))

logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)
print("\nRegresión Logística:")
print(classification_report(y_test, logreg.predict(X_test)))

X_lin = df[["Publicidad"]]
y_lin = df["Ventas_Totales"]

lin_model = LinearRegression()
lin_model.fit(X_lin, y_lin)

print("\nRegresión Lineal Simple (Publicidad → Ventas_Totales)")
print("Coeficiente:", lin_model.coef_[0])
print("Intercepto:", lin_model.intercept_)

plt.scatter(X_lin, y_lin, color="blue")
plt.plot(X_lin, lin_model.predict(X_lin), color="red")
plt.xlabel("Publicidad")
plt.ylabel("Ventas Totales")
plt.title("Regresión Lineal Simple")
plt.show()

X_multi = df[["Publicidad","Clientes","Satisfaccion","Descuento"]]
y_multi = df["Ventas_Totales"]

multi_model = LinearRegression()
multi_model.fit(X_multi, y_multi)

print("\nRegresión Múltiple (Todas las variables → Ventas_Totales)")
print("Coeficientes:", multi_model.coef_)
print("Intercepto:", multi_model.intercept_)
