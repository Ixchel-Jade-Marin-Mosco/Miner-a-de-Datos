import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("dataset_clientes.csv")

X = df[["Sucursal", "Gasto_Mensual_MXN", "Compras_Mensuales"]]

X = X.dropna()

X = pd.get_dummies(X, columns=["Sucursal"], drop_first=True)

escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

modelo = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = modelo.fit_predict(X_escalado)

df_limpio = X.copy()
df_limpio["Cluster"] = clusters

print(df_limpio)
print(df_limpio.groupby("Cluster").mean())
