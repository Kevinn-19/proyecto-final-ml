"""
Segmentación de clientes de tarjeta de crédito con K-Means y DBSCAN.

Versión en script del notebook notebooks/segmentacion_clientes_tarjetas.ipynb.
Ejecuta las siete fases del trabajo y guarda las gráficas en reports/figures,
las tablas en reports/tablas y los datos procesados en data/processed.

Uso, desde cualquier carpeta
    python scripts/segmentacion_clientes_tarjetas.py

Requiere el archivo data/raw/CC GENERAL.csv y las librerías de requirements.txt.
"""


# ============================================================================
# Segmentación de clientes de tarjeta de crédito con Machine Learning no supervisado
# ============================================================================


# ============================================================================
# Fase 0. Configuración del entorno
# ============================================================================

# Librerías
import os
import sys
import time
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # las gráficas se guardan, no se abren ventanas
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Semilla para que los resultados se puedan reproducir
SEMILLA = 42
np.random.seed(SEMILLA)

# Estilo de las gráficas
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

EN_COLAB = "google.colab" in sys.modules
print("Ejecutando en Google Colab" if EN_COLAB else "Ejecutando en local")


# --- Rutas del proyecto ---

# URL del repositorio en GitHub (solo se usa en Colab). Ejemplo: "https://github.com/usuario/proyecto-final-ml.git"
REPO_URL = "https://github.com/Kevinn-19/proyecto-final-ml.git"

NOMBRE_ARCHIVO = "CC GENERAL.csv"


def encontrar_raiz():
    """Devuelve la carpeta raíz del proyecto."""
    if EN_COLAB and REPO_URL:
        nombre_repo = REPO_URL.rstrip("/").split("/")[-1].removesuffix(".git")
        destino = Path("/content") / nombre_repo
        if not destino.exists():
            subprocess.run(["git", "clone", REPO_URL, str(destino)], check=True)
        return destino
    actual = Path(__file__).resolve().parent
    for carpeta in [actual, *actual.parents]:
        if (carpeta / "data").is_dir() and (carpeta / "notebooks").is_dir():
            return carpeta
    return actual


RAIZ = encontrar_raiz()
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROCESADOS = RAIZ / "data" / "processed"
DIR_FIGURAS = RAIZ / "reports" / "figures"
DIR_TABLAS = RAIZ / "reports" / "tablas"
for carpeta in [DIR_RAW, DIR_PROCESADOS, DIR_FIGURAS, DIR_TABLAS]:
    carpeta.mkdir(parents=True, exist_ok=True)

RUTA_DATOS = Path(os.environ.get("CC_DATA_PATH", DIR_RAW / NOMBRE_ARCHIVO))

if not RUTA_DATOS.exists():
    if EN_COLAB:
        from google.colab import files
        print(f"No se encontró '{NOMBRE_ARCHIVO}'. Súbelo desde tu computador:")
        subidos = files.upload()
        nombre_subido = next(iter(subidos))
        RUTA_DATOS = DIR_RAW / NOMBRE_ARCHIVO
        RUTA_DATOS.write_bytes(subidos[nombre_subido])
    else:
        raise FileNotFoundError(
            f"No se encontró el dataset en {RUTA_DATOS}.\n"
            "Descárgalo de https://www.kaggle.com/datasets/arjunbhasin2013/ccdata "
            "y guarda 'CC GENERAL.csv' en la carpeta data/raw/."
        )

print("Raíz del proyecto:", RAIZ)
print("Dataset:", RUTA_DATOS)


def guardar_figura(nombre):
    """Guarda la figura actual en reports/figures para usarla en el informe."""
    plt.savefig(DIR_FIGURAS / f"{nombre}.png", bbox_inches="tight")


def guardar_tabla(tabla, nombre):
    """Guarda una tabla en reports/tablas para usarla en el informe."""
    tabla.to_csv(DIR_TABLAS / f"{nombre}.csv", encoding="utf-8-sig")


# ============================================================================
# Fase 1. Descripción del problema
# ============================================================================


# ============================================================================
# Fase 2. Fuente de datos
# ============================================================================

df = pd.read_csv(RUTA_DATOS)

print(f"Registros: {df.shape[0]:,}")
print(f"Variables: {df.shape[1]}")
print(df.head())


# --- Diccionario de variables ---

DICCIONARIO = {
    "CUST_ID": "Identificador del titular (no se usa en el modelo)",
    "BALANCE": "Saldo pendiente en la cuenta",
    "BALANCE_FREQUENCY": "Frecuencia con que se actualiza el saldo (0 a 1)",
    "PURCHASES": "Monto total de compras",
    "ONEOFF_PURCHASES": "Monto de compras hechas en un solo pago",
    "INSTALLMENTS_PURCHASES": "Monto de compras hechas a cuotas",
    "CASH_ADVANCE": "Monto de avances en efectivo",
    "PURCHASES_FREQUENCY": "Frecuencia de compras (0 a 1)",
    "ONEOFF_PURCHASES_FREQUENCY": "Frecuencia de compras en un solo pago (0 a 1)",
    "PURCHASES_INSTALLMENTS_FREQUENCY": "Frecuencia de compras a cuotas (0 a 1)",
    "CASH_ADVANCE_FREQUENCY": "Frecuencia de avances en efectivo (0 a 1)",
    "CASH_ADVANCE_TRX": "Número de transacciones de avance en efectivo",
    "PURCHASES_TRX": "Número de transacciones de compra",
    "CREDIT_LIMIT": "Cupo de la tarjeta",
    "PAYMENTS": "Monto total pagado",
    "MINIMUM_PAYMENTS": "Monto de pagos mínimos realizados",
    "PRC_FULL_PAYMENT": "Porcentaje de meses en que se pagó el total (0 a 1)",
    "TENURE": "Antigüedad del servicio de tarjeta (meses)",
}

tabla_diccionario = pd.DataFrame({
    "Variable": df.columns,
    "Tipo": df.dtypes.astype(str).values,
    "Descripción": [DICCIONARIO.get(c, "") for c in df.columns],
})
guardar_tabla(tabla_diccionario.set_index("Variable"), "diccionario_variables")
print(tabla_diccionario)


# ============================================================================
# Fase 3. Análisis exploratorio de datos (EDA)
# ============================================================================

COLUMNA_ID = "CUST_ID"
numericas = df.select_dtypes(include="number").columns.tolist()

print(f"Registros: {len(df):,} | Variables numéricas: {len(numericas)}")

descriptivas = df[numericas].describe().T
descriptivas["asimetría"] = df[numericas].skew()
guardar_tabla(descriptivas, "estadisticas_descriptivas")
print(descriptivas)


# --- 3.2 Valores faltantes y registros duplicados ---

faltantes = pd.DataFrame({
    "faltantes": df.isna().sum(),
    "porcentaje": (df.isna().mean() * 100).round(2),
})
faltantes = faltantes[faltantes["faltantes"] > 0].sort_values("faltantes", ascending=False)
guardar_tabla(faltantes, "valores_faltantes")

dup_filas = df.duplicated().sum()
dup_id = df[COLUMNA_ID].duplicated().sum() if COLUMNA_ID in df else 0
dup_sin_id = df.drop(columns=[COLUMNA_ID], errors="ignore").duplicated().sum()

print(f"Filas duplicadas completas: {dup_filas}")
print(f"IDs de cliente repetidos: {dup_id}")
print(f"Filas iguales ignorando el ID: {dup_sin_id}")
print()

# ¿Los faltantes de MINIMUM_PAYMENTS tienen alguna explicación?
falta_min = df["MINIMUM_PAYMENTS"].isna()
print(f"Clientes sin MINIMUM_PAYMENTS: {falta_min.sum()}")
print(f"  de ellos con PAYMENTS = 0: {(falta_min & (df['PAYMENTS'] == 0)).sum()}")
print(f"Clientes con PAYMENTS = 0 en todo el dataset: {(df['PAYMENTS'] == 0).sum()}")
print()
print("Variables con valores faltantes:")
print(faltantes)


# --- 3.3 Valores atípicos ---

def contar_atipicos(serie):
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    return ((serie < q1 - 1.5 * iqr) | (serie > q3 + 1.5 * iqr)).sum()


atipicos = pd.DataFrame({
    "atípicos": df[numericas].apply(contar_atipicos),
})
atipicos["porcentaje"] = (atipicos["atípicos"] / len(df) * 100).round(2)
atipicos = atipicos.sort_values("atípicos", ascending=False)
guardar_tabla(atipicos, "valores_atipicos")
print(atipicos)

n_cols = 4
n_filas = int(np.ceil(len(numericas) / n_cols))
fig, ejes = plt.subplots(n_filas, n_cols, figsize=(16, 3.2 * n_filas))
for eje, col in zip(ejes.flat, numericas):
    sns.boxplot(x=df[col], ax=eje, color="#5B8DB8", fliersize=2)
    eje.set_title(col, fontsize=9)
    eje.set_xlabel("")
for eje in list(ejes.flat)[len(numericas):]:
    eje.set_visible(False)
fig.suptitle("Diagramas de caja de las variables numéricas", fontsize=13)
plt.tight_layout()
guardar_figura("eda_01_diagramas_caja")
plt.close()


# --- 3.4 Distribución de las variables (histogramas) ---

fig, ejes = plt.subplots(n_filas, n_cols, figsize=(16, 3.2 * n_filas))
for eje, col in zip(ejes.flat, numericas):
    sns.histplot(df[col].dropna(), bins=40, ax=eje, color="#5B8DB8")
    eje.set_title(f"{col}\n(asimetría = {df[col].skew():.2f})", fontsize=9)
    eje.set_xlabel("")
    eje.set_ylabel("")
for eje in list(ejes.flat)[len(numericas):]:
    eje.set_visible(False)
fig.suptitle("Histogramas de las variables numéricas", fontsize=13)
plt.tight_layout()
guardar_figura("eda_02_histogramas")
plt.close()


# --- 3.5 Relaciones entre variables (matriz de correlación) ---

correlacion = df[numericas].corr()

plt.figure(figsize=(13, 10))
mascara = np.triu(np.ones_like(correlacion, dtype=bool))
sns.heatmap(correlacion, mask=mascara, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, annot_kws={"size": 7}, linewidths=0.5)
plt.title("Matriz de correlación de Pearson", fontsize=13)
plt.tight_layout()
guardar_figura("eda_03_matriz_correlacion")
plt.close()

# Pares de variables con correlación alta
pares = (
    correlacion.where(np.triu(np.ones(correlacion.shape, dtype=bool), k=1))
    .stack()
    .rename("r")
    .reset_index()
    .rename(columns={"level_0": "variable_1", "level_1": "variable_2"})
)
pares_altos = pares[pares["r"].abs() >= 0.7].sort_values("r", key=abs, ascending=False)
guardar_tabla(pares_altos.set_index(["variable_1", "variable_2"]), "correlaciones_altas")
print("Pares con |r| >= 0.7:")
print(pares_altos)


# --- 3.6 Diagramas de dispersión más relevantes ---

PARES_DISPERSION = [
    ("PURCHASES", "PAYMENTS"),
    ("BALANCE", "CREDIT_LIMIT"),
    ("BALANCE", "CASH_ADVANCE"),
    ("PURCHASES", "ONEOFF_PURCHASES"),
]

fig, ejes = plt.subplots(1, len(PARES_DISPERSION), figsize=(18, 4.2))
for eje, (x, y) in zip(ejes, PARES_DISPERSION):
    eje.scatter(np.log1p(df[x]), np.log1p(df[y]), s=4, alpha=0.3, color="#5B8DB8")
    r = df[[x, y]].corr().iloc[0, 1]
    eje.set_xlabel(f"log(1 + {x})")
    eje.set_ylabel(f"log(1 + {y})")
    eje.set_title(f"{x} vs {y} (r = {r:.2f})", fontsize=10)
plt.tight_layout()
guardar_figura("eda_04_dispersion")
plt.close()


# --- Interpretación del EDA ---

# El análisis escrito de esta parte está en el notebook y en el informe.


# ============================================================================
# Fase 4. Preprocesamiento
# ============================================================================

datos = df.copy()
registro = []  # bitácora de lo que se hace en cada paso

# 4.1 Eliminar el identificador
datos = datos.drop(columns=[COLUMNA_ID], errors="ignore")
registro.append(("Eliminar identificador", f"Se elimina {COLUMNA_ID}", len(datos)))

# 4.2 Valores faltantes
# CREDIT_LIMIT: muy pocos faltantes -> se elimina la fila
n_antes = len(datos)
datos = datos.dropna(subset=["CREDIT_LIMIT"])
registro.append(("Faltantes en CREDIT_LIMIT", f"Se eliminan {n_antes - len(datos)} filas", len(datos)))

# MINIMUM_PAYMENTS: en el EDA se ve que los clientes con PAYMENTS = 0 son los que no tienen
# MINIMUM_PAYMENTS. Si el cliente no pagó nada, su pago mínimo realizado es 0 (no un valor típico).
faltante_min = datos["MINIMUM_PAYMENTS"].isna()
sin_pagos = faltante_min & (datos["PAYMENTS"] == 0)
datos.loc[sin_pagos, "MINIMUM_PAYMENTS"] = 0
registro.append(("Faltantes en MINIMUM_PAYMENTS (sin pagos)",
                 f"{sin_pagos.sum()} clientes con PAYMENTS = 0: se imputa 0", len(datos)))

# El resto de faltantes se imputa con la mediana (robusta ante atípicos)
resto = datos["MINIMUM_PAYMENTS"].isna()
mediana_pagos_min = datos.loc[~faltante_min, "MINIMUM_PAYMENTS"].median()
datos.loc[resto, "MINIMUM_PAYMENTS"] = mediana_pagos_min
registro.append(("Faltantes en MINIMUM_PAYMENTS (resto)",
                 f"Se imputan {resto.sum()} valores con la mediana ({mediana_pagos_min:,.2f})", len(datos)))

# Cualquier otro faltante restante se imputa con la mediana de su columna
otros = datos.columns[datos.isna().any()].tolist()
for col in otros:
    datos[col] = datos[col].fillna(datos[col].median())
if otros:
    registro.append(("Otros faltantes", f"Imputados con la mediana: {otros}", len(datos)))

# 4.3 Duplicados
n_antes = len(datos)
datos = datos.drop_duplicates()
registro.append(("Duplicados", f"Se eliminan {n_antes - len(datos)} filas", len(datos)))

# 4.4 Variables categóricas
categoricas = datos.select_dtypes(exclude="number").columns.tolist()
registro.append(("Variables categóricas",
                 "No hay; no se requiere codificación" if not categoricas else f"Codificar: {categoricas}",
                 len(datos)))

datos = datos.reset_index(drop=True)
tabla_registro = pd.DataFrame(registro, columns=["Paso", "Acción", "Registros resultantes"])
guardar_tabla(tabla_registro.set_index("Paso"), "preprocesamiento_bitacora")
print(tabla_registro)


# --- 4.5 Transformación logarítmica ---

VARIABLES_LOG = [
    "BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE", "CASH_ADVANCE_TRX", "PURCHASES_TRX", "CREDIT_LIMIT",
    "PAYMENTS", "MINIMUM_PAYMENTS",
]
VARIABLES_LOG = [c for c in VARIABLES_LOG if c in datos.columns]

datos_log = datos.copy()
datos_log[VARIABLES_LOG] = np.log1p(datos_log[VARIABLES_LOG])

comparacion_asimetria = pd.DataFrame({
    "asimetría antes": datos[VARIABLES_LOG].skew(),
    "asimetría después": datos_log[VARIABLES_LOG].skew(),
})
guardar_tabla(comparacion_asimetria, "asimetria_antes_despues_log")
print(comparacion_asimetria)


# --- 4.6 Selección de variables ---

UMBRAL_CONSTANTE = 0.80
UMBRAL_CORRELACION = 0.80

# Orden de prioridad: primero las variables más generales del comportamiento del cliente
PRIORIDAD = [
    "BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS",
    "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "BALANCE_FREQUENCY", "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES", "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX", "TENURE",
]
PRIORIDAD = [c for c in PRIORIDAD if c in datos_log.columns]
PRIORIDAD += [c for c in datos_log.columns if c not in PRIORIDAD]

decisiones = []
seleccionadas = []
corr_log = datos_log.corr()

for col in PRIORIDAD:
    proporcion_moda = datos_log[col].value_counts(normalize=True).iloc[0]
    if proporcion_moda > UMBRAL_CONSTANTE:
        decisiones.append((col, "Descartada", f"Casi constante: {proporcion_moda:.0%} de los clientes tiene el mismo valor"))
        continue
    redundante_con = [s for s in seleccionadas if abs(corr_log.loc[col, s]) >= UMBRAL_CORRELACION]
    if redundante_con:
        s = redundante_con[0]
        decisiones.append((col, "Descartada", f"Redundante con {s} (r = {corr_log.loc[col, s]:.2f})"))
        continue
    seleccionadas.append(col)
    decisiones.append((col, "Seleccionada", ""))

tabla_seleccion = pd.DataFrame(decisiones, columns=["Variable", "Decisión", "Motivo"])
guardar_tabla(tabla_seleccion.set_index("Variable"), "seleccion_variables")
print(f"Variables seleccionadas ({len(seleccionadas)}): {seleccionadas}")
print(tabla_seleccion)


# --- 4.7 Estandarización ---

escalador = StandardScaler()
X = pd.DataFrame(
    escalador.fit_transform(datos_log[seleccionadas]),
    columns=seleccionadas,
)

# Entregable de la Fase 4: dataset preparado
datos.to_csv(DIR_PROCESADOS / "cc_limpio.csv", index=False)
X.to_csv(DIR_PROCESADOS / "cc_preparado_estandarizado.csv", index=False)

print(f"Dataset preparado: {X.shape[0]:,} registros x {X.shape[1]} variables")
print("Guardado en data/processed/cc_preparado_estandarizado.csv")
print(X.describe().T[["mean", "std", "min", "max"]])


# --- 4.8 Reducción a 2 dimensiones con PCA (solo para visualizar) ---

pca = PCA(n_components=2, random_state=SEMILLA)
X_2d = pca.fit_transform(X)
varianza_2d = pca.explained_variance_ratio_.sum()
print(f"Varianza explicada por las 2 primeras componentes: {varianza_2d:.1%}")


# ============================================================================
# Fase 5. Implementación de los dos algoritmos
# ============================================================================

def punto_de_codo(x, y):
    """Índice del punto de máxima curvatura (método de la distancia a la recta).

    Se traza una recta entre el primer y el último punto de la curva y se elige
    el punto que queda más lejos de esa recta.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_n = (x - x.min()) / (x.max() - x.min())
    y_n = (y - y.min()) / (y.max() - y.min())
    inicio = np.array([x_n[0], y_n[0]])
    fin = np.array([x_n[-1], y_n[-1]])
    direccion = (fin - inicio) / np.linalg.norm(fin - inicio)
    puntos = np.column_stack([x_n, y_n]) - inicio
    proyeccion = np.outer(puntos @ direccion, direccion)
    distancias = np.linalg.norm(puntos - proyeccion, axis=1)
    return int(np.argmax(distancias))


def metricas_clustering(X, etiquetas):
    """Calcula las métricas internas. En DBSCAN se excluye el ruido (etiqueta -1)."""
    etiquetas = np.asarray(etiquetas)
    mascara = etiquetas != -1
    n_grupos = len(set(etiquetas[mascara]))
    resultado = {
        "n_clusters": n_grupos,
        "% ruido": (~mascara).mean() * 100,
        "Silhouette": np.nan,
        "Davies-Bouldin": np.nan,
        "Calinski-Harabasz": np.nan,
    }
    if n_grupos >= 2:
        Xm = np.asarray(X)[mascara]
        em = etiquetas[mascara]
        resultado["Silhouette"] = silhouette_score(Xm, em)
        resultado["Davies-Bouldin"] = davies_bouldin_score(Xm, em)
        resultado["Calinski-Harabasz"] = calinski_harabasz_score(Xm, em)
    return resultado


# --- 5.1 K-Means, elección del número de grupos (k) ---

RANGO_K = range(2, 11)
resultados_k = []

for k in RANGO_K:
    inicio = time.perf_counter()
    modelo = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA)
    etiquetas = modelo.fit_predict(X)
    duracion = time.perf_counter() - inicio
    fila = {"k": k, "Inercia": modelo.inertia_, "Tiempo (s)": duracion}
    fila.update(metricas_clustering(X, etiquetas))
    resultados_k.append(fila)

tabla_k = pd.DataFrame(resultados_k).set_index("k").drop(columns=["n_clusters", "% ruido"])
guardar_tabla(tabla_k, "kmeans_busqueda_k")

k_codo = list(RANGO_K)[punto_de_codo(list(RANGO_K), tabla_k["Inercia"])]
k_silhouette = tabla_k["Silhouette"].idxmax()
print(f"k sugerido por el método del codo: {k_codo}")
print(f"k con mayor Silhouette: {k_silhouette}")
print(tabla_k)

fig, (eje1, eje2) = plt.subplots(1, 2, figsize=(13, 4.5))

eje1.plot(tabla_k.index, tabla_k["Inercia"], "o-", color="#5B8DB8")
eje1.axvline(k_codo, ls="--", color="#D1495B", label=f"Codo en k = {k_codo}")
eje1.set_title("Método del codo")
eje1.set_xlabel("Número de grupos (k)")
eje1.set_ylabel("Inercia (suma de distancias al centroide)")
eje1.legend()

eje2.plot(tabla_k.index, tabla_k["Silhouette"], "o-", color="#2E8B57")
eje2.axvline(k_silhouette, ls="--", color="#D1495B", label=f"Máximo en k = {k_silhouette}")
eje2.set_title("Coeficiente Silhouette según k")
eje2.set_xlabel("Número de grupos (k)")
eje2.set_ylabel("Silhouette")
eje2.legend()

plt.tight_layout()
guardar_figura("modelo_01_kmeans_codo_silhouette")
plt.close()


# --- Elección final de k ---

K_ELEGIDO = k_codo  # cambiar aquí si el equipo justifica otro valor

inicio = time.perf_counter()
kmeans = KMeans(n_clusters=K_ELEGIDO, n_init=10, random_state=SEMILLA)
etiquetas_km = kmeans.fit_predict(X)
tiempo_km = time.perf_counter() - inicio

print(f"K-Means con k = {K_ELEGIDO} entrenado en {tiempo_km:.3f} s")
print("Clientes por grupo:")
print(pd.Series(etiquetas_km).value_counts().sort_index().to_string())


# --- 5.2 DBSCAN, elección de eps y min_samples ---

MIN_SAMPLES_BASE = 2 * X.shape[1]

vecinos = NearestNeighbors(n_neighbors=MIN_SAMPLES_BASE).fit(X)
distancias, _ = vecinos.kneighbors(X)
distancia_k = np.sort(distancias[:, -1])

indice_codo = punto_de_codo(np.arange(len(distancia_k)), distancia_k)
eps_codo = float(distancia_k[indice_codo])

plt.figure(figsize=(8, 4.5))
plt.plot(distancia_k, color="#5B8DB8")
plt.axhline(eps_codo, ls="--", color="#D1495B", label=f"eps sugerido = {eps_codo:.2f}")
plt.title(f"Distancia al {MIN_SAMPLES_BASE}-ésimo vecino más cercano (ordenada)")
plt.xlabel("Clientes ordenados por distancia")
plt.ylabel("Distancia")
plt.legend()
plt.tight_layout()
guardar_figura("modelo_02_dbscan_k_distancia")
plt.close()

print(f"min_samples base: {MIN_SAMPLES_BASE}")
print(f"eps sugerido por el codo: {eps_codo:.3f}")


# --- Búsqueda de parámetros de DBSCAN ---

MAX_RUIDO = 15  # porcentaje máximo de ruido aceptado

valores_eps = np.round(eps_codo * np.arange(0.6, 1.45, 0.1), 3)
valores_min_samples = sorted({max(3, MIN_SAMPLES_BASE // 2), MIN_SAMPLES_BASE, int(MIN_SAMPLES_BASE * 1.5)})

resultados_db = []
for eps in valores_eps:
    for ms in valores_min_samples:
        inicio = time.perf_counter()
        etiquetas = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
        duracion = time.perf_counter() - inicio
        fila = {"eps": eps, "min_samples": ms, "Tiempo (s)": duracion}
        fila.update(metricas_clustering(X, etiquetas))
        resultados_db.append(fila)

tabla_db = pd.DataFrame(resultados_db)
guardar_tabla(tabla_db.set_index(["eps", "min_samples"]), "dbscan_busqueda_parametros")

validas = tabla_db[(tabla_db["n_clusters"] >= 2) & (tabla_db["% ruido"] <= MAX_RUIDO)]
if validas.empty:
    print(f"Ninguna combinación cumple (>= 2 grupos y <= {MAX_RUIDO}% de ruido).")
    validas = tabla_db[tabla_db["n_clusters"] >= 2]
    if not validas.empty:
        print("Se elige la de mayor Silhouette entre las que forman al menos 2 grupos.")

if validas.empty:
    print("DBSCAN no logra formar 2 o más grupos con estos parámetros; se usa el eps del codo.")
    mejor = {"eps": eps_codo, "min_samples": MIN_SAMPLES_BASE}
else:
    mejor = validas.sort_values("Silhouette", ascending=False).iloc[0]

EPS_ELEGIDO = float(mejor["eps"])
MIN_SAMPLES_ELEGIDO = int(mejor["min_samples"])
print(f"Parámetros elegidos: eps = {EPS_ELEGIDO:.3f}, min_samples = {MIN_SAMPLES_ELEGIDO}")
print(tabla_db.sort_values("Silhouette", ascending=False).head(10))

inicio = time.perf_counter()
dbscan = DBSCAN(eps=EPS_ELEGIDO, min_samples=MIN_SAMPLES_ELEGIDO)
etiquetas_db = dbscan.fit_predict(X)
tiempo_db = time.perf_counter() - inicio

print(f"DBSCAN entrenado en {tiempo_db:.3f} s")
print("Clientes por grupo (-1 = ruido):")
print(pd.Series(etiquetas_db).value_counts().sort_index().to_string())


# ============================================================================
# Fase 6. Evaluación del desempeño
# ============================================================================

def resumen_modelo(nombre, parametros, etiquetas, tiempo):
    fila = {"Algoritmo": nombre, "Parámetros": parametros}
    fila.update(metricas_clustering(X, etiquetas))
    tamanos = pd.Series(etiquetas[etiquetas != -1]).value_counts()
    fila["Grupo más grande (%)"] = tamanos.max() / len(etiquetas) * 100 if len(tamanos) else np.nan
    fila["Grupo más pequeño (%)"] = tamanos.min() / len(etiquetas) * 100 if len(tamanos) else np.nan
    fila["Tiempo (s)"] = tiempo
    return fila


comparacion = pd.DataFrame([
    resumen_modelo("K-Means", f"k = {K_ELEGIDO}", etiquetas_km, tiempo_km),
    resumen_modelo("DBSCAN", f"eps = {EPS_ELEGIDO:.3f}, min_samples = {MIN_SAMPLES_ELEGIDO}",
                   etiquetas_db, tiempo_db),
]).set_index("Algoritmo")
guardar_tabla(comparacion, "comparacion_algoritmos")
print(comparacion)


# --- Gráfica, comparación de métricas ---

METRICAS = [("Silhouette", "más alto es mejor"),
            ("Davies-Bouldin", "más bajo es mejor"),
            ("Calinski-Harabasz", "más alto es mejor")]
COLORES = ["#5B8DB8", "#E8A33D"]

fig, ejes = plt.subplots(1, 3, figsize=(14, 4))
for eje, (metrica, nota) in zip(ejes, METRICAS):
    valores = comparacion[metrica]
    barras = eje.bar(valores.index, valores.values, color=COLORES)
    eje.bar_label(barras, fmt="%.2f")
    eje.set_title(f"{metrica}\n({nota})", fontsize=10)
fig.suptitle("Comparación de métricas internas", fontsize=13)
plt.tight_layout()
guardar_figura("eval_01_comparacion_metricas")
plt.close()


# --- Gráfica, grupos vistos en 2D (PCA) ---

fig, ejes = plt.subplots(1, 2, figsize=(15, 6))

for eje, (nombre, etiquetas) in zip(ejes, [("K-Means", etiquetas_km), ("DBSCAN", etiquetas_db)]):
    paleta = sns.color_palette("tab10")
    for g in sorted(set(etiquetas)):
        m = etiquetas == g
        if g == -1:
            eje.scatter(X_2d[m, 0], X_2d[m, 1], s=6, c="lightgray", label="Ruido", alpha=0.6)
        else:
            eje.scatter(X_2d[m, 0], X_2d[m, 1], s=6, color=paleta[g % len(paleta)],
                        label=f"Grupo {g}", alpha=0.6)
    eje.set_title(nombre)
    eje.set_xlabel("Componente principal 1")
    eje.set_ylabel("Componente principal 2")
    eje.legend(markerscale=3, fontsize=8, loc="best")

fig.suptitle(f"Grupos proyectados con PCA ({varianza_2d:.0%} de la varianza explicada)", fontsize=13)
plt.tight_layout()
guardar_figura("eval_02_grupos_pca")
plt.close()


# --- Gráfica, tamaño de cada grupo ---

fig, ejes = plt.subplots(1, 2, figsize=(14, 4.5))

for eje, (nombre, etiquetas, color) in zip(
    ejes, [("K-Means", etiquetas_km, COLORES[0]), ("DBSCAN", etiquetas_db, COLORES[1])]
):
    conteo = pd.Series(etiquetas).value_counts().sort_index()
    nombres = ["Ruido" if g == -1 else f"Grupo {g}" for g in conteo.index]
    colores = ["lightgray" if g == -1 else color for g in conteo.index]
    barras = eje.bar(nombres, conteo.values, color=colores)
    eje.bar_label(barras, labels=[f"{v:,}\n({v / len(etiquetas):.0%})" for v in conteo.values], fontsize=8)
    eje.set_title(f"Clientes por grupo con {nombre}")
    eje.set_ylabel("Número de clientes")
    eje.tick_params(axis="x", rotation=45)
    eje.margins(y=0.2)

plt.tight_layout()
guardar_figura("eval_03_tamano_grupos")
plt.close()


# ============================================================================
# Fase 7. Interpretación de los resultados
# ============================================================================

VARIABLES_PERFIL = [
    "BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE",
    "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "CREDIT_LIMIT", "PAYMENTS",
    "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE",
]
VARIABLES_PERFIL = [c for c in VARIABLES_PERFIL if c in datos.columns]


def perfil_grupos(etiquetas, nombre):
    tabla = datos[VARIABLES_PERFIL].copy()
    tabla["Grupo"] = etiquetas
    tabla = tabla[tabla["Grupo"] != -1]
    perfil = tabla.groupby("Grupo").mean()
    perfil.insert(0, "Clientes", tabla["Grupo"].value_counts().sort_index())
    guardar_tabla(perfil, f"perfil_{nombre.lower().replace('-', '')}")
    return perfil


def mapa_perfiles(etiquetas, nombre, archivo):
    z = datos_log[VARIABLES_PERFIL].apply(lambda s: (s - s.mean()) / s.std())
    z["Grupo"] = etiquetas
    z = z[z["Grupo"] != -1].groupby("Grupo").mean()
    plt.figure(figsize=(13, 0.6 * len(z) + 2.5))
    sns.heatmap(z, annot=True, fmt=".1f", cmap="RdBu_r", center=0, linewidths=0.5,
                cbar_kws={"label": "Desviaciones estándar respecto al promedio"})
    plt.title(f"Perfil de los grupos de {nombre}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    guardar_figura(archivo)
    plt.close()
    return z


def describir_grupos(z, n=3):
    """Lista las variables en las que cada grupo está más por encima y más por debajo del promedio."""
    for grupo, fila in z.iterrows():
        altas = fila.sort_values(ascending=False).head(n)
        bajas = fila.sort_values().head(n)
        print(f"Grupo {grupo}:")
        print("   Alto en: " + ", ".join(f"{v} ({fila[v]:+.1f})" for v in altas.index if fila[v] > 0.3))
        print("   Bajo en: " + ", ".join(f"{v} ({fila[v]:+.1f})" for v in bajas.index if fila[v] < -0.3))


# --- 7.1 Perfiles de K-Means ---

perfil_km = perfil_grupos(etiquetas_km, "K-Means")
print(perfil_km)

z_km = mapa_perfiles(etiquetas_km, "K-Means", "interp_01_perfiles_kmeans")
print(describir_grupos(z_km))


# --- 7.2 Perfiles de DBSCAN ---

if len(set(etiquetas_db) - {-1}) >= 1:
    perfil_db = perfil_grupos(etiquetas_db, "DBSCAN")
    print(perfil_db)
    z_db = mapa_perfiles(etiquetas_db, "DBSCAN", "interp_02_perfiles_dbscan")
    describir_grupos(z_db)
else:
    print("DBSCAN no formó grupos (todo es ruido).")


# --- 7.3 ¿Quiénes son los clientes que DBSCAN marca como ruido? ---

es_ruido = etiquetas_db == -1
if es_ruido.any():
    comparacion_ruido = pd.DataFrame({
        "Ruido (promedio)": datos.loc[es_ruido, VARIABLES_PERFIL].mean(),
        "Resto (promedio)": datos.loc[~es_ruido, VARIABLES_PERFIL].mean(),
    })
    comparacion_ruido["Ruido / Resto"] = comparacion_ruido["Ruido (promedio)"] / comparacion_ruido["Resto (promedio)"]
    guardar_tabla(comparacion_ruido, "dbscan_ruido_vs_resto")
    print(f"Clientes marcados como ruido: {es_ruido.sum():,} ({es_ruido.mean():.1%})")
    print(comparacion_ruido)
else:
    print("DBSCAN no marcó ningún cliente como ruido.")


# --- 7.4 Guardar los resultados ---

resultado_final = datos.copy()
resultado_final["grupo_kmeans"] = etiquetas_km
resultado_final["grupo_dbscan"] = etiquetas_db
resultado_final.to_csv(DIR_PROCESADOS / "cc_con_grupos.csv", index=False)

print("Archivos generados:")
for carpeta in [DIR_PROCESADOS, DIR_FIGURAS, DIR_TABLAS]:
    for archivo in sorted(carpeta.glob("*.*")):
        print("  ", archivo.relative_to(RAIZ))


# --- Interpretación y selección del algoritmo ---

# El análisis escrito de esta parte está en el notebook y en el informe.


# ============================================================================
# Referencias
# ============================================================================
