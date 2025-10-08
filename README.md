<!-- ===================== BANNER ===================== -->
<p align="center">
  <img src="https://raw.githubusercontent.com/NanoHtz/Assets/main/data/banner.svg" alt="Philosophers banner">
</p>

<!-- ===================== BADGES (ajusta/borra los que no apliquen) ===================== -->

</p>

<h1 align="center">Data Analysis with Python 2024–2025 — University of Helsinki</h1>
<p align="center"><i>Notebooks y utilidades que siguen el currículo oficial: NumPy, Pandas, visualización, limpieza de datos y ML básico.</i></p>

---

## Índice
- [Resumen](#resumen)
- [Objetivos y Temario](#objetivos-y-temario)
- [Capturas / Demo](#capturas--demo)
- [Stack / Dependencias](#stack--dependencias)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Instalación y uso](#instalación-y-uso)
- [Comandos útiles / Makefile](#comandos-útiles--makefile)
- [Datos: dónde colocarlos](#datos-dónde-colocarlos)
- [Calidad de código](#calidad-de-código)
- [Tests](#tests)
- [Rendimiento / Escalabilidad](#rendimiento--escalabilidad)
- [Roadmap](#roadmap)
- [Problemas conocidos](#problemas-conocidos)
- [FAQ](#faq)
- [Contribuir](#contribuir)
- [Estilo de código](#estilo-de-código)
- [Versionado y changelog](#versionado-y-changelog)
- [Licencia](#licencia)
- [Autor](#autor)
- [Agradecimientos](#agradecimientos)

---

## Resumen
- **Qué**: Colección de *notebooks* y scripts que siguen el curso **Data Analysis with Python (2024–2025)** de la Univ. de Helsinki.
- **Para qué**: Practicar un flujo completo de análisis de datos (ingestión → limpieza → EDA → visualización → modelos intro de *ML*).
---

## Objetivos y Temario
- **Objetivos**:
  1. Dominar operaciones fundamentales con **NumPy** y **Pandas**.
  2. Realizar **EDA** sistemática (tipos, *missing values*, outliers, transformaciones).
  3. Crear **visualizaciones** claras con **Matplotlib**.
  4. Aplicar **estadística básica** y modelos introductorios con **scikit-learn**.
  5. Documentar procesos y resultados de forma reproducible en *notebooks*.

- **Temario (notebooks principales)**:
  1. `01_numpy_basics.ipynb` — arrays, broadcasting, vectorización.
  2. `02_pandas_dataframes.ipynb` — series/dataframes, joins, groupby, pivot.
  3. `03_data_cleaning.ipynb` — *missing*, *duplicates*, *dtypes*, *string ops*.
  4. `04_visualization_matplotlib.ipynb` — figuras, ejes, layouts, anotaciones.
  5. `05_statistics_scipy.ipynb` — descriptiva, tests básicos, intervalos.
  6. `06_ml_intro_sklearn.ipynb` — *train/test split*, métricas, pipelines.

---

## Capturas / Demo
<p align="center">
  <!-- Sube tus imágenes a assets/ -->
  <!-- <img src="assets/eda_overview.png" width="85%" alt="EDA Overview"> -->
  <!-- <img src="assets/plots_demo.gif" width="85%" alt="Plots demo"> -->
  <i>Incluye aquí una captura de un EDA o una galería de gráficos.</i>
</p>

---

## Stack / Dependencias
- **Lenguaje**: Python 3.10+
- **Entorno**: `venv` (o `conda`)
- **Paquetes base**:
  - Núcleo: `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`
  - Notebooks: `jupyter`, `ipykernel`
  - Calidad (opcional): `black`, `flake8`, `isort`, `pytest`
- **Fichero de requisitos**: `requirements.txt`

> Si usas `conda`, crea un `environment.yml` equivalente.

---

## Estructura del repositorio
```text
Data-Analysis-with-Python-2024-2025/
├─ notebooks/
│  ├─ 01_numpy_basics.ipynb
│  ├─ 02_pandas_dataframes.ipynb
│  ├─ 03_data_cleaning.ipynb
│  ├─ 04_visualization_matplotlib.ipynb
│  ├─ 05_statistics_scipy.ipynb
│  ├─ 06_ml_intro_sklearn.ipynb
│  └─ templates/
│     ├─ eda_template.ipynb          # plantilla para arrancar un EDA
│     └─ notebook_starter.ipynb      # esqueleto con celdas comunes
├─ data/
│  ├─ raw/                           # datasets originales (no tocar)
│  └─ processed/                     # datasets transformados/limpios
├─ src/
│  ├─ io_utils.py                    # lectura/escritura CSV/JSON/Parquet
│  ├─ eda.py                         # funciones auxiliares EDA
│  ├─ viz.py                         # helpers de visualización
│  └─ ml_utils.py                    # split, métricas, pipelines básicos
├─ tests/
│  └─ test_io_utils.py               # ejemplo con pytest
├─ assets/                           # banner, capturas
├─ requirements.txt
├─ Makefile                          # atajos (fmt, lint, test, clean...)
└─ README.md

