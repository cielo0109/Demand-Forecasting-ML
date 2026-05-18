# Demand Forecasting for University Dining Services

A machine learning pipeline that forecasts daily meal demand across multiple university restaurants, enabling smarter resource allocation and reducing food waste.

---

## Problem Statement

University dining halls face a recurring operational challenge: preparing the right number of meals each day without over- or under-producing. Overproduction leads to food waste; underproduction affects student satisfaction. This project builds an end-to-end data pipeline that ingests historical meal records, cleans and transforms the data, trains predictive models, and generates short-term demand forecasts.

**Context:** Data sourced from UNICAMP (University of Campinas, Brazil) via public SIC requests — covering 3 dining halls, 2 daily meal services, and multiple menu categories.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                │
│                                                                     │
│  Raw Data           ETL Layer             Modeling Layer            │
│  ─────────          ─────────             ───────────────           │
│                                                                     │
│  SIC Excel   ──►  Menu          ──►  Decision Tree                  │
│  (meal logs)      Categorization      Regressor (R²=0.987)          │
│                   (Python)                                          │
│                        │          ──►  Negative Binomial GLM        │
│  Raw Excel   ──►  Data Cleaning        (count data model)           │
│  (raw ops)        (Python + R)                                      │
│                        │          ──►  K-Prototypes Clustering      │
│                        ▼               (demand segmentation)        │
│               Processed CSV                                         │
│               (2,052 rows)       ──►  Demand Forecasts              │
│                                       (30-day horizon)              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Demand-Forecasting-ML/
│
├── notebooks/                         # Jupyter notebooks (ordered pipeline)
│   ├── 01_menu_categorization.ipynb   # ETL: categorize dishes into protein groups
│   ├── 02_data_cleaning.ipynb         # ETL: clean, filter, encode features (Python)
│   ├── 03_exploratory_analysis.ipynb  # EDA: distributions, time series, boxplots (R)
│   └── 04_demand_forecasting.ipynb    # Models: Decision Tree + Negative Binomial GLM
│
├── r_analysis/                        # R Markdown analyses
│   ├── 01_initial_analysis.Rmd        # Initial exploratory data analysis
│   └── 02_clustering_kprototypes.Rmd  # Unsupervised clustering (K-Prototypes)
│
├── data/
│   ├── raw/                           # Original source files (unmodified)
│   │   ├── Dados_Semprocesar.xlsx
│   │   └── Solicitacoes_SIC_Volumetria_Refeicoes.xlsx
│   └── processed/                     # Output of the ETL pipeline
│       ├── dados-seg.csv              # Main training dataset (2,052 rows × 11 cols)
│       ├── dados-previsao.csv         # Prediction input (30 rows)
│       └── banco_limpo.xlsx           # Cleaned reference dataset
│
├── reports/                           # Project documentation and academic reports
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

## Tech Stack

| Layer | Technology |
|---|---|
| Data ingestion | `pandas`, `openpyxl`, `gdown` |
| Data transformation | `pandas`, `numpy`, R (`tidyverse`, `readxl`) |
| Supervised modeling | `scikit-learn` (DecisionTreeRegressor) |
| Statistical modeling | `statsmodels` (Negative Binomial GLM) |
| Unsupervised modeling | R (`clustMixType` — K-Prototypes) |
| Visualization | `matplotlib`, `seaborn`, R (`ggplot2`) |
| Environment | Jupyter Notebooks, R Markdown |

---

## Dataset

The processed dataset covers **February 2024 – August 2025** and contains one row per meal service per restaurant per menu category.

| Column | Description |
|---|---|
| `date` | Date of service |
| `tipo_refeicao` | Meal type (Almoço / Jantar) |
| `restaurante_utilizado` | Dining hall (RU, RA, RS) |
| `categoria_padrao` | Standard menu protein (beef, chicken, fish, pork, eggs, feijoada) |
| `categoria_vegano` | Vegan menu protein (legumes, PTS, whole grains, vegetables) |
| `dia_semana` | Day of week |
| `mes` | Month |
| `almoco` | Binary flag — 1 = lunch |
| `periodo_semestre` | Academic period (start, middle, end, holiday) |
| `feriado_dummy` | Holiday flag — 1 = public holiday |
| `quantidade` | **Target variable** — number of meals served |

---

## Notebooks Walkthrough

### `01_menu_categorization.ipynb` — ETL: Menu Parsing
Parses raw dish names from the SIC records and maps them to standardized protein categories using keyword dictionaries. This creates the `categoria_padrao` and `categoria_vegano` features used throughout the pipeline.

### `02_data_cleaning.ipynb` — ETL: Data Preparation
- Parses and filters dates (from 2024-02-28 onward)
- Drops irrelevant columns (salads, beverages, side dishes)
- Creates binary-encoded features (`almoco`, `feriado_dummy`)
- Exports the cleaned training dataset to `data/processed/dados-seg.csv`

### `03_exploratory_analysis.ipynb` — EDA
- Time series plots of demand by restaurant and meal type
- Boxplots comparing demand across days of week, academic periods, and categories
- Distribution analysis to detect seasonality and outliers

### `04_demand_forecasting.ipynb` — Modeling
Main modeling notebook implementing two complementary approaches:

**Decision Tree Regressor**
- One-hot encodes categorical features (41 total features after encoding)
- 80/20 train/test split
- `max_depth=20`

**Negative Binomial GLM**
- Appropriate for count data with overdispersion
- Formula: `quantidade ~ almoco + restaurante + categoria_padrao + categoria_vegano + dia_semana + periodo_semestre + feriado_dummy`

---

## Results

### Decision Tree Regressor

| Metric | Value |
|---|---|
| R² Score | **0.9875** |
| MAE | 46.16 meals |
| RMSE | 122.58 meals |

### K-Prototypes Clustering (6 segments)

| Cluster | Profile |
|---|---|
| 1 | High demand — RU dinners, mid-semester Wednesdays |
| 2 | Low demand — RA dinners, mid-semester Fridays |
| 3 | Medium demand — RS lunches, early-semester Tuesdays |
| 4 | Low demand — RS lunches, holiday Fridays |
| 5 | Peak demand — RU lunches, early-semester Mondays |
| 6 | Low demand — RS dinners, late-semester Mondays |

### Sample Forecast (October 2025 — Monday)

| Restaurant | Meal | Predicted Meals |
|---|---|---|
| RU | Lunch | ~3,190 |
| RA | Lunch | ~1,321 |
| RS | Lunch | ~1,497 |
| RU | Dinner | ~2,883 |

---

## Key Findings

- **Day of week** is the strongest demand driver — Mondays and Wednesdays peak; Friday dinners drop significantly.
- **Academic period** causes structural demand shifts — semester start shows peak demand, holiday periods reduce it by ~30–40%.
- **Restaurant RU** handles ~50–60% of total daily volume across all time periods.
- **Lunch demand** is consistently 25–40% higher than dinner demand at all restaurants.
- The Decision Tree captures non-linear interactions between all these factors effectively, explaining 98.75% of variance.

---

## How to Run

**Requirements:** Python 3.10+, R 4.x

```bash
# 1. Clone the repository
git clone <repo-url>
cd Demand-Forecasting-ML

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run notebooks in order
jupyter notebook notebooks/
# Execute: 01 → 02 → 03 → 04

# 4. For R analyses, open r_analysis/ files in RStudio
# Required R packages: tidyverse, readxl, ggplot2, clustMixType
```

---

## Academic Context

This project was developed as a final project applying supervised and unsupervised machine learning techniques to a real operational dataset from a Brazilian public university. The goal was both predictive accuracy and interpretability — making the models actionable for dining hall managers who plan daily meal quantities.

**Reference:** [Food Waste and Machine Learning — Research Paper](reports/desperdicio_alimentar_e_machine_learning.pdf)
