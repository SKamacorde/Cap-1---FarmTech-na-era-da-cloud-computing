# make_plots.py
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# ---- Carregar dataset
df = pd.read_csv("crop_yield.csv")
df.columns = [c.strip() for c in df.columns]

# ---- Resolver nomes de colunas (fuzzy)
def find_col(candidates, patterns, required=True):
    # devolve o primeiro nome de coluna que "bate" com os padrões
    for col in candidates:
        low = col.lower()
        if all(p in low for p in patterns):
            return col
    if required:
        raise ValueError(f"Não encontrei coluna com padrões: {patterns}")
    return None

cols = {}
cands = list(df.columns)

cols["crop"] = find_col(cands, ["crop"])
cols["yield"] = find_col(cands, ["yield"])

# precip, specific humidity, relative humidity, temperature (nomes longos)
cols["precip"] = find_col(cands, ["precip"])  # e.g. "Precipitation (mm day-1)"
cols["shum"]   = find_col(cands, ["specific", "humidity"])  # "Specific Humidity at 2 Meters (g/kg)"
cols["rhum"]   = find_col(cands, ["relative", "humidity"])  # "Relative Humidity at 2 Meters (%)"
cols["temp"]   = find_col(cands, ["temperature"])           # "Temperature at 2 Meters (C)"

# ---- EDA

# 1) Histograma do Yield
plt.figure(figsize=(8,6))
plt.hist(df[cols["yield"]], bins=30)
plt.title("Distribuição do Rendimento (Yield)")
plt.xlabel("Rendimento (ton/ha)")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "eda_histogram_yield.png"))
plt.close()

# 2) Boxplot do Yield por Cultura
plt.figure(figsize=(10,6))
order = df.groupby(cols["crop"])[cols["yield"]].median().sort_values(ascending=False).index
# usar matplotlib puro para evitar depender de seaborn
# converter categorias para índice numérico
xlabels = list(order)
pos = range(len(xlabels))
data_by_crop = [df.loc[df[cols["crop"]]==lbl, cols["yield"]].values for lbl in xlabels]
plt.boxplot(data_by_crop, showfliers=True)
plt.xticks([p+1 for p in pos], xlabels, rotation=45, ha="right")
plt.ylabel("Rendimento (ton/ha)")
plt.title("Boxplot do Rendimento por Cultura")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "eda_boxplot_culturas.png"))
plt.close()

# 3) Heatmap de Correlação (matplotlib)
num_df = df[[cols["precip"], cols["shum"], cols["rhum"], cols["temp"], cols["yield"]]].copy()
corr = num_df.corr(numeric_only=True)

plt.figure(figsize=(8,6))
plt.imshow(corr, aspect="auto")
plt.colorbar()
plt.title("Mapa de Correlação das Variáveis")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "eda_heatmap.png"))
plt.close()

# ---- Clusterização (K-Means)
# Usaremos todas as numéricas exceto a categórica 'crop'
X = df[[cols["precip"], cols["shum"], cols["rhum"], cols["temp"]]].copy()
X = X.replace([np.inf, -np.inf], np.nan).dropna()

# 4) Método do Cotovelo
inertia = []
K_RANGE = range(1, 10)
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    km.fit(X)
    inertia.append(km.inertia_)

plt.figure(figsize=(8,6))
plt.plot(list(K_RANGE), inertia, marker="o")
plt.title("Método do Cotovelo - KMeans (Dataset Completo)")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Inércia")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "cluster_elbow.png"))
plt.close()

# 5) Scatter Clusterizado (Temperatura vs Yield)
# Precisamos alinhar Yield com X após dropna()
df_clean = df.loc[X.index, [cols["temp"], cols["yield"]]].copy()
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
labels = kmeans.fit_predict(X)
df_clean["Cluster"] = labels

plt.figure(figsize=(8,6))
plt.scatter(df_clean[cols["temp"]], df_clean[cols["yield"]], c=df_clean["Cluster"], s=10)
plt.title("Clusters - Temperatura vs Rendimento (Completo)")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Rendimento (ton/ha)")
cb = plt.colorbar()
cb.set_label("Cluster")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "cluster_scatter.png"))
plt.close()

print("Arquivos salvos em ./assets/:")
print(" - eda_histogram_yield.png")
print(" - eda_boxplot_culturas.png")
print(" - eda_heatmap.png")
print(" - cluster_elbow.png")
print(" - cluster_scatter.png")