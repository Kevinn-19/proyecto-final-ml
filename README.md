# Segmentación de clientes de tarjeta de crédito con machine learning no supervisado

Trabajo final de Introducción a Machine Learning, Especialización en Inteligencia Artificial, Universidad de Cundinamarca. Kevin Esteban Chiquillo Díaz, noviembre de 2026.

En este proyecto busco perfiles de clientes de tarjeta de crédito a partir de cómo usan la tarjeta. Uso dos algoritmos de agrupamiento, K-Means y DBSCAN, sobre unos 9.000 clientes, comparo la calidad de los grupos con varias métricas y escojo el algoritmo que mejor funciona para el problema.

La pregunta que quiero responder es esta.

> ¿Qué perfiles de comportamiento en el uso de tarjetas de crédito existen entre los titulares del dataset, y cuál de los algoritmos K-Means y DBSCAN los identifica con mayor calidad de agrupamiento?

## Resultado principal

K-Means con cinco grupos fue el algoritmo más apropiado. Los perfiles que encontré son estos.

- Compradores frecuentes de alto gasto: 1.804 clientes (20 %).
- Compradores a cuotas: 2.317 clientes (26 %).
- Revolventes de alto saldo: 1.498 clientes (17 %).
- Usuarios de avances en efectivo: 2.174 clientes (24 %).
- Inactivos o de bajo uso: 1.156 clientes (13 %).

DBSCAN dejó a casi todos los clientes en un mismo grupo, pero me sirvió para ubicar a 141 clientes que no hicieron pagos en seis meses y a un 1,6 % de casos atípicos.

## Estructura del proyecto

```
├── data/
│   ├── raw/               # Dataset original, CC GENERAL.csv
│   └── processed/         # Datos limpios, estandarizados y con grupos
├── notebooks/
│   └── segmentacion_clientes_tarjetas.ipynb   # Las 7 fases del trabajo
├── scripts/
│   ├── segmentacion_clientes_tarjetas.py      # El mismo análisis como script
│   └── diagrama_bloques.py                    # Genera el diagrama de bloques
├── reports/
│   ├── figures/           # Gráficas que usa el informe
│   └── tablas/            # Tablas en CSV que usa el informe
├── docs/                  # Informe final en Word y PDF
├── requirements.txt
└── README.md
```

## Dataset

Uso el Credit Card Dataset for Clustering, que Arjun Bhasin publicó en [Kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) en 2018 con licencia de dominio público. El archivo `CC GENERAL.csv` ya viene incluido en `data/raw`, así que no hace falta descargarlo. Si se quiere bajar de nuevo, hay que iniciar sesión en Kaggle, descargarlo, descomprimirlo y dejar el CSV en esa misma carpeta.

## Cómo ejecutarlo

### En Google Colab

La forma más fácil es abrir el notebook directamente con este [enlace a Colab](https://colab.research.google.com/github/Kevinn-19/proyecto-final-ml/blob/main/notebooks/segmentacion_clientes_tarjetas.ipynb). El notebook clona el repositorio solo y usa el dataset que viene incluido. Colab ya trae todas las librerías que se necesitan.

Otra opción es subir el `.ipynb` a Colab a mano. En ese caso, si no encuentra el dataset, el notebook pide subir `CC GENERAL.csv`.

### En local

Se necesita Python 3.10 o superior.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook notebooks/segmentacion_clientes_tarjetas.ipynb
```

También se puede abrir con VS Code usando la extensión de Jupyter.

### Como script de Python

El mismo análisis está en `scripts/segmentacion_clientes_tarjetas.py`. Corre las siete fases, muestra las tablas en la consola y guarda las gráficas y tablas en `reports`. Tarda unos tres minutos.

```powershell
python scripts/segmentacion_clientes_tarjetas.py
```

## Fases del trabajo

| Fase | Contenido | Salida principal |
|---|---|---|
| 1 | Descripción del problema y pregunta de investigación | Texto en el notebook |
| 2 | Fuente de datos y diccionario de variables | `reports/tablas/diccionario_variables.csv` |
| 3 | Análisis exploratorio | Diagramas de caja, histogramas, correlación y dispersión |
| 4 | Preprocesamiento | `data/processed/cc_preparado_estandarizado.csv` |
| 5 | K-Means y DBSCAN | Búsqueda de k, de eps y de min_samples |
| 6 | Evaluación | Silhouette, Davies-Bouldin, Calinski-Harabasz, porcentaje de ruido, tamaños y tiempo |
| 7 | Interpretación y selección del algoritmo | Perfiles de los grupos y `data/processed/cc_con_grupos.csv` |

## Herramientas

| Herramienta | Para qué la usé |
|---|---|
| Python 3 | Lenguaje del proyecto |
| Jupyter y Google Colab | Ejecutar el notebook |
| pandas y NumPy | Cargar, limpiar y transformar los datos |
| Matplotlib y seaborn | Hacer las gráficas |
| scikit-learn | `StandardScaler`, `PCA`, `KMeans`, `DBSCAN`, `NearestNeighbors` y las métricas |
| Git y GitHub | Guardar el historial de cambios |
