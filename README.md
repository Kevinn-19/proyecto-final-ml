# Segmentación de clientes de tarjeta de crédito con Machine Learning no supervisado

Trabajo final del curso **Introducción a Machine Learning**.

Se aplican dos algoritmos de clustering, **K-Means** y **DBSCAN**, sobre el comportamiento de uso de tarjetas de crédito de unos 9.000 clientes. El objetivo es descubrir perfiles de clientes, evaluar la calidad de los grupos con métricas cuantitativas y elegir el algoritmo más apropiado.

**Pregunta de investigación:** ¿Qué perfiles de comportamiento en el uso de tarjetas de crédito existen entre los titulares del dataset, y cuál de los algoritmos K-Means y DBSCAN los identifica con mayor calidad de agrupamiento?

## Estructura del proyecto

```
├── data/
│   ├── raw/               # Dataset original: CC GENERAL.csv
│   └── processed/         # Datos limpios, estandarizados y con grupos (los genera el notebook)
├── notebooks/
│   └── segmentacion_clientes_tarjetas.ipynb   # Las 7 fases del trabajo
├── scripts/
│   ├── segmentacion_clientes_tarjetas.py      # El mismo análisis del notebook, como script
│   └── diagrama_bloques.py                    # Genera el diagrama de bloques
├── reports/
│   ├── figures/           # Gráficas para el informe (las genera el notebook o el script)
│   └── tablas/            # Tablas en CSV para el informe (las genera el notebook o el script)
├── docs/                  # Informe final en Word y PDF
├── requirements.txt
└── README.md
```

## Dataset

**Credit Card Dataset for Clustering**: Bhasin, A. (2018). Kaggle.
https://www.kaggle.com/datasets/arjunbhasin2013/ccdata

1. Inicia sesión en Kaggle y descarga el dataset.
2. Descomprime y guarda el archivo `CC GENERAL.csv` en `data/raw/`.

## Ejecutar en local

Requiere Python 3.10 o superior.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook notebooks/segmentacion_clientes_tarjetas.ipynb
```

El notebook también se puede abrir con VS Code usando la extensión de Jupyter.

**Como script de Python.** El mismo análisis está en `scripts/segmentacion_clientes_tarjetas.py`. Ejecuta las 7 fases, imprime las tablas en la consola y guarda las gráficas y tablas en `reports/`. Tarda unos 3 minutos.

```powershell
python scripts/segmentacion_clientes_tarjetas.py
```

## Ejecutar en Google Colab

**Opción A: desde GitHub (recomendada)**
1. En Colab: *Archivo → Abrir cuaderno → GitHub* y pega la URL del repositorio.
2. En la celda de rutas, escribe la URL del repositorio en `REPO_URL`. El notebook clona el repo y usa el CSV de `data/raw/`.

**Opción B: subiendo el notebook**
1. Sube el `.ipynb` a Colab.
2. Al ejecutar, si no encuentra el dataset, el notebook pide subir `CC GENERAL.csv`.

Colab ya trae instaladas todas las librerías necesarias.

## Fases del trabajo

| Fase | Contenido | Salida principal |
|---|---|---|
| 1 | Descripción del problema y pregunta de investigación | Texto en el notebook |
| 2 | Fuente de datos y diccionario de variables | `reports/tablas/diccionario_variables.csv` |
| 3 | Análisis exploratorio (EDA) | Diagramas de caja, histogramas, correlación y dispersión |
| 4 | Preprocesamiento | `data/processed/cc_preparado_estandarizado.csv` |
| 5 | K-Means y DBSCAN | Búsqueda de k, de eps y de min_samples |
| 6 | Evaluación | Silhouette, Davies-Bouldin, Calinski-Harabasz, % de ruido, tamaños y tiempo |
| 7 | Interpretación y selección del algoritmo | Perfiles de los grupos y `data/processed/cc_con_grupos.csv` |

## Herramientas

| Herramienta | Uso |
|---|---|
| Python 3 | Lenguaje del proyecto |
| Jupyter / Google Colab | Ejecución del notebook y trabajo colaborativo |
| pandas, numpy | Carga, limpieza y transformación de los datos |
| matplotlib, seaborn | Gráficas |
| scikit-learn | `StandardScaler`, `PCA`, `KMeans`, `DBSCAN`, `NearestNeighbors` y métricas |
| Git / GitHub | Control de versiones del equipo |
