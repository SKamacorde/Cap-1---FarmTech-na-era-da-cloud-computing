# ===== Artefatos extras: CSVs + gráficos (rodar no final do notebook) =====
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
ASSETS = Path("assets"); ASSETS.mkdir(exist_ok=True)

# ---------- garante df carregado e colunas detectadas ----------
def find_col(cands, patterns, required=True):
    for col in cands:
        low = col.lower()
        if all(p in low for p in patterns):
            return col
    if required:
        raise ValueError(f"Coluna não encontrada para padrões: {patterns}")
    return None

try:
    df  # se já existir no notebook, seguimos
except NameError:
    from pathlib import Path
    CSV_PATH = Path("crop_yield.csv")
    assert CSV_PATH.exists(), "Coloque crop_yield.csv na raiz do projeto."
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

cands   = list(df.columns)
col_crop = find_col(cands, ["crop"])
col_yield= find_col(cands, ["yield"])
col_prec = find_col(cands, ["precip"])
col_shum = find_col(cands, ["specific","humidity"])
col_rhum = find_col(cands, ["relative","humidity"])
col_temp = find_col(cands, ["temperature"])

df = df[[col_crop, col_prec, col_shum, col_rhum, col_temp, col_yield]].replace([np.inf,-np.inf], np.nan).dropna()

# ---------- CSV: resumo de clusters ----------
Xc = df[[col_prec, col_shum, col_rhum, col_temp]].copy()
kmeans = KMeans(n_clusters=4, random_state=SEED, n_init=10, max_iter=300)
labels = kmeans.fit_predict(Xc)
cluster_summary = (
    df.assign(Cluster=labels)
      .groupby(["Cluster", col_crop])[[col_yield]]
      .agg(["mean","median","count"])
      .round(3)
)
cluster_summary.to_csv(ASSETS / "cluster_summary.csv")

# ---------- Modelagem (5 modelos) + métricas ----------
FEATURES = [col_prec, col_shum, col_rhum, col_temp]
TARGET   = col_yield
X = df[FEATURES].values
y = df[TARGET].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

def eval_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))  # compatível com qualquer sklearn
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2

# modelos
lr = LinearRegression(); lr.fit(Xtr, ytr)
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
lasso = Lasso(alpha=0.0001, random_state=SEED, max_iter=50000); lasso.fit(Xtr_s, ytr)
ridge = Ridge(alpha=1.0, random_state=SEED); ridge.fit(Xtr_s, ytr)
rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1); rf.fit(Xtr, ytr)
gbr = GradientBoostingRegressor(random_state=SEED); gbr.fit(Xtr, ytr)

rows = []
for name, model, Xte_use in [
    ("Linear Regression", lr,  Xte),
    ("Lasso Regression",   lasso, Xte_s),
    ("Ridge Regression",   ridge, Xte_s),
    ("Random Forest Regressor", rf,  Xte),
    ("Gradient Boosting Regressor", gbr, Xte),
]:
    pred = model.predict(Xte_use)
    rows.append((name, *eval_metrics(yte, pred)))

df_metrics = pd.DataFrame(rows, columns=["Modelo","MAE","RMSE","R2"]).sort_values("R2", ascending=False)
df_metrics_round = df_metrics.copy()
df_metrics_round[["MAE","RMSE","R2"]] = df_metrics_round[["MAE","RMSE","R2"]].round(4)
df_metrics_round.to_csv(ASSETS / "model_metrics.csv", index=False)
print(df_metrics_round)

# ---------- PNG: resíduos do melhor modelo ----------
best_name = df_metrics.iloc[0]["Modelo"]
name_to_model = {
    "Linear Regression": lr,
    "Lasso Regression": lasso,
    "Ridge Regression": ridge,
    "Random Forest Regressor": rf,
    "Gradient Boosting Regressor": gbr,
}
Xte_map = {"Linear Regression": Xte, "Lasso Regression": Xte_s, "Ridge Regression": Xte_s,
           "Random Forest Regressor": Xte, "Gradient Boosting Regressor": Xte}
best_model = name_to_model[best_name]
y_pred = best_model.predict(Xte_map[best_name])
res = yte - y_pred

plt.figure(figsize=(8,6))
plt.scatter(y_pred, res, s=12)
plt.axhline(0, linestyle="--")
plt.title(f"Resíduos — {best_name}")
plt.xlabel("Predito"); plt.ylabel("Resíduo (y_true - y_pred)")
plt.tight_layout(); plt.savefig(ASSETS / "residuos_best_model.png"); plt.show()

# ---------- PNG: barplot de R² ----------
plt.figure(figsize=(8,5))
plt.bar(df_metrics["Modelo"], df_metrics["R2"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("R² (holdout)")
plt.title("Comparação de R² entre modelos")
plt.tight_layout(); plt.savefig(ASSETS / "r2_barplot.png"); plt.show()

print("✔ Artefatos criados em ./assets/: cluster_summary.csv, model_metrics.csv, residuos_best_model.png, r2_barplot.png")