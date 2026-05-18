# Pronóstico de Demanda para Comedores Universitarios

Un pipeline de machine learning que predice la demanda diaria de comidas en múltiples restaurantes universitarios, permitiendo una mejor asignación de recursos y reduciendo el desperdicio alimentario.

---

## Problema

Los comedores universitarios enfrentan un desafío operativo recurrente: preparar la cantidad correcta de comidas cada día sin sobreproducir ni quedarse cortos. La sobreproducción genera desperdicio; la subproducción afecta la satisfacción estudiantil. Este proyecto construye un pipeline de datos end-to-end que ingesta registros históricos de comidas, limpia y transforma los datos, entrena modelos predictivos y genera pronósticos de demanda a corto plazo.

**Contexto:** Datos provenientes de la UNICAMP (Universidad de Campinas, Brasil) mediante solicitudes públicas SIC — abarcando 3 comedores, 2 servicios de comida diarios y múltiples categorías de menú.

---

## Visión General del Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE DE DATOS                            │
│                                                                     │
│  Datos Crudos       Capa ETL              Capa de Modelos           │
│  ────────────       ─────────             ─────────────────         │
│                                                                     │
│  Excel SIC   ──►  Categorización  ──►  Árbol de Decisión            │
│  (registros)      de Menú              Regresor (R²=0.987)          │
│                   (Python)                                          │
│                        │          ──►  GLM Binomial Negativo        │
│  Excel crudo ──►  Limpieza de          (modelo para datos de        │
│  (operaciones)    Datos (Python + R)    conteo)                     │
│                        │                                            │
│                        │          ──►  Clustering K-Prototipos      │
│                        ▼               (segmentación de demanda)    │
│               CSV Procesado                                         │
│               (2.052 filas)      ──►  Pronósticos de Demanda        │
│                                       (horizonte de 30 días)        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Estructura del Proyecto

```
Demand-Forecasting-ML/
│
├── notebooks/                         # Jupyter notebooks (pipeline ordenado)
│   ├── 01_menu_categorization.ipynb   # ETL: categorizar platos en grupos proteicos
│   ├── 02_data_cleaning.ipynb         # ETL: limpiar, filtrar y codificar variables (Python)
│   ├── 03_exploratory_analysis.ipynb  # EDA: distribuciones, series temporales, boxplots (R)
│   └── 04_demand_forecasting.ipynb    # Modelos: Árbol de Decisión + GLM Binomial Negativo
│
├── r_analysis/                        # Análisis en R Markdown
│   ├── 01_initial_analysis.Rmd        # Análisis exploratorio inicial
│   └── 02_clustering_kprototypes.Rmd  # Clustering no supervisado (K-Prototipos)
│
├── data/
│   ├── raw/                           # Archivos fuente originales (sin modificar)
│   │   ├── Dados_Semprocesar.xlsx
│   │   └── Solicitacoes_SIC_Volumetria_Refeicoes.xlsx
│   └── processed/                     # Salida del pipeline ETL
│       ├── dados-seg.csv              # Dataset de entrenamiento (2.052 filas × 11 cols)
│       ├── dados-previsao.csv         # Input de predicción (30 filas)
│       └── banco_limpo.xlsx           # Dataset de referencia limpio
│
├── reports/                           # Documentación y reportes académicos
│   ├── Parte_I_Projeto_Final.pdf
│   ├── Parte_II_Modelos_e_Predicoes.pdf
│   ├── Detalhes_Arvore_de_Decisao.pdf
│   ├── Limpeza_Banco_de_Dados.pdf
│   ├── Caderno_Unicamp.pdf
│   └── desperdicio_alimentar_e_machine_learning.pdf
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Stack Tecnológico

| Capa | Tecnología |
|---|---|
| Ingesta de datos | `pandas`, `openpyxl`, `gdown` |
| Transformación de datos | `pandas`, `numpy`, R (`tidyverse`, `readxl`) |
| Modelado supervisado | `scikit-learn` (DecisionTreeRegressor) |
| Modelado estadístico | `statsmodels` (GLM Binomial Negativo) |
| Modelado no supervisado | R (`clustMixType` — K-Prototipos) |
| Visualización | `matplotlib`, `seaborn`, R (`ggplot2`) |
| Entorno | Jupyter Notebooks, R Markdown |

---

## Dataset

El dataset procesado cubre **febrero 2024 – agosto 2025** y contiene una fila por servicio de comida, por restaurante y por categoría de menú.

| Columna | Descripción |
|---|---|
| `date` | Fecha del servicio |
| `tipo_refeicao` | Tipo de comida (Almoço / Jantar) |
| `restaurante_utilizado` | Comedor (RU, RA, RS) |
| `categoria_padrao` | Proteína del menú estándar (vacuno, pollo, pescado, cerdo, huevos, feijoada) |
| `categoria_vegano` | Proteína del menú vegano (legumbres, PTS, cereales integrales, vegetales) |
| `dia_semana` | Día de la semana |
| `mes` | Mes |
| `almoco` | Variable binaria — 1 = almuerzo |
| `periodo_semestre` | Período académico (inicio, medio, fin, vacaciones) |
| `feriado_dummy` | Variable de feriado — 1 = día feriado |
| `quantidade` | **Variable objetivo** — número de comidas servidas |

---

## Descripción de los Notebooks

### `01_menu_categorization.ipynb` — ETL: Parseo del Menú
Procesa los nombres crudos de los platos desde los registros SIC y los mapea a categorías proteicas estandarizadas mediante diccionarios de palabras clave. Genera las variables `categoria_padrao` y `categoria_vegano` utilizadas en todo el pipeline.

### `02_data_cleaning.ipynb` — ETL: Preparación de Datos
- Parsea y filtra fechas (desde el 28/02/2024 en adelante)
- Elimina columnas irrelevantes (ensaladas, bebidas, guarniciones)
- Crea variables codificadas en binario (`almoco`, `feriado_dummy`)
- Exporta el dataset de entrenamiento limpio a `data/processed/dados-seg.csv`

### `03_exploratory_analysis.ipynb` — EDA
- Gráficos de series temporales de demanda por comedor y tipo de comida
- Boxplots comparando la demanda por día de la semana, período académico y categoría
- Análisis de distribuciones para detectar estacionalidad y outliers

### `04_demand_forecasting.ipynb` — Modelado
Notebook principal de modelado con dos enfoques complementarios:

**Árbol de Decisión Regresor**
- Codificación one-hot de variables categóricas (41 variables en total tras la codificación)
- División 80/20 entrenamiento/prueba
- `max_depth=20`

**GLM Binomial Negativo**
- Adecuado para datos de conteo con sobredispersión
- Fórmula: `quantidade ~ almoco + restaurante + categoria_padrao + categoria_vegano + dia_semana + periodo_semestre + feriado_dummy`

---

## Resultados

### Árbol de Decisión Regresor

| Métrica | Valor |
|---|---|
| R² | **0.9875** |
| MAE | 46,16 comidas |
| RMSE | 122,58 comidas |

### Clustering K-Prototipos (6 segmentos)

| Cluster | Perfil |
|---|---|
| 1 | Demanda alta — cenas en RU, miércoles de mediados de semestre |
| 2 | Demanda baja — cenas en RA, viernes de mediados de semestre |
| 3 | Demanda media — almuerzos en RS, martes de inicio de semestre |
| 4 | Demanda baja — almuerzos en RS, viernes feriados |
| 5 | Demanda pico — almuerzos en RU, lunes de inicio de semestre |
| 6 | Demanda baja — cenas en RS, lunes de fin de semestre |

### Pronóstico de Muestra (Octubre 2025 — Lunes)

| Comedor | Servicio | Comidas Pronosticadas |
|---|---|---|
| RU | Almuerzo | ~3.190 |
| RA | Almuerzo | ~1.321 |
| RS | Almuerzo | ~1.497 |
| RU | Cena | ~2.883 |

---

## Hallazgos Clave

- El **día de la semana** es el factor más determinante — los lunes y miércoles concentran la mayor demanda; las cenas del viernes caen significativamente.
- El **período académico** genera cambios estructurales en la demanda — el inicio de semestre muestra el pico más alto; los feriados reducen la demanda entre un 30–40%.
- El **comedor RU** maneja el 50–60% del volumen diario total en todos los períodos.
- La **demanda de almuerzos** es consistentemente 25–40% mayor que la de cenas en todos los comedores.
- El Árbol de Decisión captura interacciones no lineales entre todos estos factores, explicando el 98,75% de la varianza.

---

## Cómo Ejecutar

**Requisitos:** Python 3.10+, R 4.x

```bash
# 1. Clonar el repositorio
git clone <repo-url>
cd Demand-Forecasting-ML

# 2. Instalar dependencias Python
pip install -r requirements.txt

# 3. Ejecutar notebooks en orden
jupyter notebook notebooks/
# Ejecutar: 01 → 02 → 03 → 04

# 4. Para los análisis en R, abrir los archivos de r_analysis/ en RStudio
# Paquetes R requeridos: tidyverse, readxl, ggplot2, clustMixType
```

---

## Contexto Académico

Este proyecto fue desarrollado como trabajo final aplicando técnicas de machine learning supervisado y no supervisado sobre un dataset operacional real de una universidad pública brasileña. El objetivo fue tanto la precisión predictiva como la interpretabilidad — haciendo que los modelos sean accionables para los gestores de comedores que planifican las cantidades diarias de comidas.

**Referencia:** [Desperdicio Alimentario y Machine Learning — Paper de Investigación](reports/desperdicio_alimentar_e_machine_learning.pdf)
