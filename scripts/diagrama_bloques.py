"""Genera el diagrama de bloques de la solución.

Uso (desde la raíz del proyecto):
    python scripts/diagrama_bloques.py

Salidas:
    reports/figures/diagrama_bloques.png
    reports/figures/diagrama_bloques.pdf
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "reports" / "figures"

# Colores por tipo de etapa: (relleno, borde)
DATOS = ("#E3EEF9", "#2F6DB5")
PREPARACION = ("#E2F4F1", "#1F8A78")
KMEANS = ("#FDEBD8", "#D9822B")
DBSCAN = ("#EEE6F7", "#7B4FB0")
EVALUACION = ("#E6F4E2", "#3C8D3A")
RESULTADO = ("#FBE4E6", "#B83B4B")
TEXTO = "#1F2933"
GRIS = "#5F6B7A"


def caja(ax, x, y, ancho, alto, titulo, lineas, colores, herramientas=None):
    """Dibuja un bloque con título, contenido y herramientas usadas."""
    relleno, borde = colores
    ax.add_patch(FancyBboxPatch(
        (x, y), ancho, alto, boxstyle="round,pad=0,rounding_size=1.2",
        facecolor=relleno, edgecolor=borde, linewidth=1.8,
    ))
    ax.text(x + 1.5, y + alto - 1.6, titulo, ha="left", va="top",
            fontsize=11.5, fontweight="bold", color=borde)
    ax.text(x + 1.5, y + alto - 4.4, "\n".join(lineas), ha="left", va="top",
            fontsize=9, color=TEXTO, linespacing=1.45)
    if herramientas:
        ax.text(x + ancho - 1.2, y + 1.0, herramientas, ha="right", va="bottom",
                fontsize=8, color=GRIS, style="italic")


def flecha(ax, x1, y1, x2, y2, etiqueta=None, lado="derecha"):
    """Dibuja una flecha con la salida que pasa de un bloque al siguiente."""
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
        color="#4A5563", linewidth=1.6, shrinkA=0, shrinkB=0,
    ))
    if etiqueta:
        diagonal = abs(x2 - x1) > 1  # en flechas inclinadas se aleja más el texto
        desplazamiento = (4 if diagonal else 1.2) * (1 if lado == "derecha" else -1)
        ax.text((x1 + x2) / 2 + desplazamiento, (y1 + y2) / 2, etiqueta,
                ha="left" if lado == "derecha" else "right", va="center",
                fontsize=8.5, color=GRIS, style="italic")


def fase(ax, y_centro, texto):
    """Etiqueta de la fase del trabajo en la columna izquierda."""
    ax.text(5, y_centro, texto, ha="center", va="center", fontsize=9,
            fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#3E4C59", edgecolor="none"))


def main():
    fig, ax = plt.subplots(figsize=(12, 17))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 150)
    ax.axis("off")

    ax.text(55, 148.5, "Diagrama de bloques de la solución",
            ha="center", va="top", fontsize=17, fontweight="bold", color=TEXTO)
    ax.text(55, 145.2, "Segmentación de clientes de tarjeta de crédito con K-Means y DBSCAN",
            ha="center", va="top", fontsize=11.5, color=GRIS)

    X0, ANCHO = 12, 86          # bloques de ancho completo
    MEDIO = X0 + ANCHO / 2

    # 1. Fuente de datos
    caja(ax, X0, 128, ANCHO, 13, "1. Fuente de datos", [
        "Credit Card Dataset for Clustering (Bhasin, 2018) · Kaggle · licencia CC0",
        "Archivo CC GENERAL.csv: 8.950 titulares × 18 variables, 6 meses de uso, sin etiquetas",
    ], DATOS, "pandas · read_csv")
    fase(ax, 134.5, "Fase 2")
    flecha(ax, MEDIO, 128, MEDIO, 123.5, "DataFrame original (8.950 × 18)")

    # 2. EDA
    caja(ax, X0, 105, ANCHO, 18.5, "2. Análisis exploratorio (EDA)", [
        "• Estadísticas descriptivas y asimetría  • Valores faltantes (313 en MINIMUM_PAYMENTS, 1 en CREDIT_LIMIT)",
        "• Duplicados (0)  • Atípicos con la regla IQR  • Histogramas y diagramas de caja",
        "• Matriz de correlación y dispersión (PURCHASES ~ ONEOFF_PURCHASES, r = 0,92)",
        "Hallazgo: los 240 clientes con PAYMENTS = 0 son los que no tienen MINIMUM_PAYMENTS",
    ], DATOS, "pandas · matplotlib · seaborn")
    fase(ax, 114, "Fase 3")
    flecha(ax, MEDIO, 105, MEDIO, 100.5, "Decisiones de limpieza y transformación")

    # 3. Preprocesamiento
    caja(ax, X0, 77, ANCHO, 23.5, "3. Preprocesamiento", [
        "1) Eliminar el identificador CUST_ID",
        "2) Faltantes: MINIMUM_PAYMENTS = 0 si PAYMENTS = 0; resto con la mediana; quitar 1 fila sin CREDIT_LIMIT",
        "3) Eliminar duplicados  ·  4) Sin variables categóricas (no requiere codificación)",
        "5) Transformación log(1 + x) en montos y conteos para reducir la asimetría",
        "6) Selección de variables: quitar las casi constantes (TENURE) y las redundantes (|r| ≥ 0,80)",
        "7) Estandarización con StandardScaler (media 0, desviación 1)",
    ], PREPARACION, "pandas · numpy · scikit-learn")
    fase(ax, 89, "Fase 4")

    # Bifurcación hacia los dos algoritmos
    ax.text(MEDIO, 74.6, "Dataset preparado: 8.949 clientes × 13 variables (mismas variables para ambos algoritmos)",
            ha="center", va="center", fontsize=8.5, color=GRIS, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#C9D1DA"))
    ANCHO_MEDIO = 41.5
    X_KM, X_DB = X0, X0 + ANCHO - ANCHO_MEDIO
    flecha(ax, MEDIO - 8, 73.3, X_KM + ANCHO_MEDIO / 2, 70)
    flecha(ax, MEDIO + 8, 73.3, X_DB + ANCHO_MEDIO / 2, 70)

    # 4a. K-Means
    caja(ax, X_KM, 51, ANCHO_MEDIO, 19, "4a. K-Means (centroides)", [
        "• Probar k = 2 … 10 (n_init = 10, semilla 42)",
        "• Elegir k con el método del codo",
        "   y el coeficiente Silhouette",
        "• Modelo final: k = 5",
    ], KMEANS, "sklearn.cluster.KMeans")
    # 4b. DBSCAN
    caja(ax, X_DB, 51, ANCHO_MEDIO, 19, "4b. DBSCAN (densidad)", [
        "• min_samples base = 2 × n.º de variables",
        "• eps desde la gráfica de distancia",
        "   al k-ésimo vecino",
        "• Búsqueda de eps y min_samples",
        "• Modelo final: eps = 2,34; min_samples = 39",
    ], DBSCAN, "DBSCAN · NearestNeighbors")
    fase(ax, 60.5, "Fase 5")

    flecha(ax, X_KM + ANCHO_MEDIO / 2, 51, MEDIO - 8, 46.5, "Etiquetas de grupo", lado="izquierda")
    flecha(ax, X_DB + ANCHO_MEDIO / 2, 51, MEDIO + 8, 46.5, "Etiquetas (−1 = ruido)")

    # 5. Evaluación
    caja(ax, X0, 31, ANCHO, 15.5, "5. Evaluación del desempeño", [
        "• Métricas internas: Silhouette, Davies-Bouldin, Calinski-Harabasz",
        "• Porcentaje de ruido, tamaño de cada grupo y tiempo de ejecución",
        "• Gráficas: codo/Silhouette, k-distancia, grupos en 2D con PCA, tamaños y comparación de métricas",
    ], EVALUACION, "sklearn.metrics · PCA · matplotlib")
    fase(ax, 38.75, "Fase 6")
    flecha(ax, MEDIO, 31, MEDIO, 27.5, "Tabla comparativa de métricas")

    # 6. Comparación y selección
    caja(ax, X0, 17, ANCHO, 10.5, "6. Comparación y selección del algoritmo", [
        "K-Means: 5 grupos equilibrados (13 % – 26 %), mejor Calinski-Harabasz y más rápido  →  algoritmo elegido",
        "DBSCAN: un grupo con el 97 % de los clientes; útil como complemento para detectar no pagadores y atípicos",
    ], RESULTADO)
    fase(ax, 22.25, "Fase 7")
    flecha(ax, MEDIO, 17, MEDIO, 13.5, "Perfiles promedio por grupo")

    # 7. Interpretación y uso
    caja(ax, X0, 1, ANCHO, 12.5, "7. Interpretación y uso de los resultados", [
        "Segmentos: compradores frecuentes de alto gasto · compradores a cuotas · revolventes de alto saldo ·",
        "usuarios de avances en efectivo · inactivos   →   campañas por segmento, ajuste de cupos y gestión de riesgo",
    ], RESULTADO, "cc_con_grupos.csv · perfiles por grupo")
    fase(ax, 7.25, "Fase 7")

    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / "diagrama_bloques.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(SALIDA / "diagrama_bloques.pdf", bbox_inches="tight", facecolor="white")
    print("Diagrama guardado en", SALIDA)


if __name__ == "__main__":
    main()
