import pandas as pd

df = pd.read_csv("dataset.csv")

print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())

print("Duplicados:", df.duplicated().sum())

print("Promedio de compras:")
print(df["Compras_Mensuales"].mean())

print("Promedio de gasto:")
print(df["Gasto_Mensual_MXN"].mean())

print("Promedio de satisfacción:")
print(df["Satisfaccion"].mean())

print(df.groupby("Sucursal")["Gasto_Mensual_MXN"].mean())

print(df.groupby("Nivel_Consumo_Referencia")[[
    "Compras_Mensuales",
    "Gasto_Mensual_MXN",
    "Visitas_Web_Mensuales",
    "Satisfaccion"
]].mean())
print("\n--- Análisis de Categorías Favoritas ---")
print(df["Categoria_Favorita"].value_counts())

print("\n--- Análisis de Canal de Origen ---")
print(df["Canal_Origen"].value_counts())

print("\n--- Gasto Mensual Promedio por Categoría Favorita ---")
print(df.groupby("Categoria_Favorita")["Gasto_Mensual_MXN"].mean())

print("\n--- Matriz de Correlación de Características Numéricas ---")

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
print(df[numerical_cols].corr())

print("\n--- Satisfacción Promedio por Sentimiento de Referencia ---")
print(df.groupby("Sentimiento_Referencia")["Satisfaccion"].mean())