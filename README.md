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
- [Explicaciones](#explicaciones)

---

## Resumen
- **Qué**: *noteboos* y scripts que siguen el curso **Data Analysis with Python (2024–2025)** de la Univ. de Helsinki.
- **Para qué**: Practicar un flujo completo de análisis de datos (ingestión → limpieza → EDA → visualización → modelos intro de *ML*).
---

## Objetivos y Temario
- **Objetivos**:
  1. Dominar operaciones fundamentales con **NumPy** y **Pandas**.
  2. Realizar **EDA** sistemática (tipos, *missing values*, outliers, transformaciones).
  3. Crear **visualizaciones** claras con **Matplotlib**.
  4. Aplicar **estadística básica** y modelos introductorios con **scikit-learn**.
  5. Documentar procesos y resultados de forma reproducible en *notebooks*.

## Explicaciones

### Tema 1 — Python (Parte 1)
- **Tipos y operadores**: `int/float/bool/str/None`; aritméticos, comparación y lógicos. *Truthiness* de `0`, `''`, `[]`, `{}`, `None`.
- **Cadenas**: *slicing* `s[a:b:c]`, métodos clave (`split/join/strip/replace/lower`), *f-strings*.
- **Estructuras**: listas (mutables), tuplas (inmutables), sets (sin duplicados), diccionarios (clave→valor). *Comprehensions* y *unpacking*.
- **Control de flujo y funciones**: `if/for/while`, `range`, funciones con argumentos por defecto y *docstrings*.

### Tema 2 — Python (Parte 2)
- **Modularidad**: `__init__.py`, organización en `src/`, patrón `if __name__ == "__main__":`.
- **Funciones avanzadas**: `*args/**kwargs`, *type hints*, `lambda`, `map/filter/sorted`, `functools.reduce`.
- **Iterables**: `zip`, `enumerate`, *generators* y *star expressions*.
- **Excepciones**: `try/except/else/finally`; captura específica y re-lanzado cuando proceda.
- **E/S**: `with open(...)` para seguridad; CSV/JSON con stdlib si no se usa pandas.

### Tema 2 — NumPy (Parte 1)
- **`ndarray` y `dtype`**: creación (`array/arange/linspace/zeros/ones/random`) y control de precisión/memoria.
- **Forma e indexación**: `shape/ndim/size`; *slicing*, máscaras booleanas, listas de índices.
- **Broadcasting**: alinear formas; añadir ejes con `None/np.newaxis`.
- **Vectorización**: ufuncs elementales (`+ - * /`, `exp`, `sqrt`) y reducciones (`sum/mean/...`) con `axis`.
- **Condicionales**: `np.where` para selección/asignación sin bucles.

### Tema 3 — Procesamiento de imágenes
- **Representación**: gris `H×W` (0–255 `uint8` o 0–1 `float`); color `H×W×3` (RGB). Normaliza a `[0,1]` para operar, vuelve a `uint8` al guardar.
- **Operaciones**: negativo, contraste/clipping, umbralado y máscaras booleanas; condicionales vectorizadas.
- **Convolución**: suavizado (media/gauss) y bordes (Sobel/Prewitt). Parámetros: kernel, *padding*, modo.
- **Geométricas**: *flip*, rotaciones 90°, *crop* y *resize* (cuidar aliasing).
- **Histograma/contraste**: lectura de distribución tonal; ecualización (idea general).
- **Buenas prácticas**: documentar espacio de color y rangos; mostrar **antes/después** y validar (p.ej., energía de bordes).

### Tema 3 — Matplotlib (visión rápida)
- **OO API**: `Figure/Axes`; títulos/etiquetas/leyendas; `tight_layout`/`constrained_layout`.
- **Gráficos comunes**: línea, dispersión, barras, hist, boxplot, imagen/heatmap (+ *colorbar* cuando el color codifica magnitud).
- **Composición**: `subplots`, ejes compartidos, *insets*, `twinx`.
- **Anotar y exportar**: `annotate`, `zorder`; exporta con `dpi` y `bbox_inches="tight"`.
- **Estilo**: colormaps perceptuales, leyendas que no tapen datos, *ticks* legibles.

### Tema 3 — NumPy (Parte 2)
- **Indexación avanzada**: listas de índices y máscaras booleanas para leer/escribir sin bucles.
- **Broadcasting y ejes**: `np.newaxis`/`None`; operaciones por filas/columnas con `axis`.
- **Agregaciones**: `sum/mean/std/min/max` (control de ejes y `keepdims`).
- **Ordenación/únicos**: `sort/argsort`, `unique(..., return_counts=True)`.
- **Composición**: `hstack/vstack/stack`, `split/array_split`; diferencia *view* vs copia.
- **Álgebra y aleatoriedad**: producto `@`, `lstsq` (mínimos cuadrados), `default_rng` reproducible.
- **Rendimiento**: evita bucles Python en *hot paths*; cuida `dtype` para no forzar *casts* ni gastar memoria.

### Tema 3 — Pandas (Parte 1)
- **Estructuras**: `Series` y `DataFrame`. Inspección rápida: `head/info/describe` y `dtypes`.
- **E/S**: CSV/JSON/Excel/Parquet (preferible Parquet). Define `dtype` y `parse_dates` al leer.
- **Selección**: `loc` (etiqueta), `iloc` (posición), filtrado booleano, `select_dtypes`.
- **Columnas**: vectorización, `assign`, *method chaining*; `to_numeric`, `astype('category')`.
- **Orden/deduplicación**: `sort_values`, `drop_duplicates`.
- **Nulos**: `isna/notna`; `fillna` (constante/mediana/moda) o `dropna` según contexto; decide **por columna**.

### Tema 4 — Pandas (Parte 2)
- **GroupBy/agg**: `groupby().agg(...)`, `transform` para resultados alineados; evita `apply` si hay alternativa.
- **Pivot/reshape**: `pivot_table(margins=True)`, `melt/pivot`, `stack/unstack`, `MultiIndex`.
- **Merge/concat**: `merge(..., how=...)`, `concat(axis=0/1)`, *join* por índice, `merge_asof`. Valida con `validate=` y revisa nulos post-merge.
- **Fechas/tiempo**: `to_datetime`, accesor `dt`; `set_index(sort)`; `resample/rolling/ewm`, `shift/diff`, `reindex+ffill`.
- **Calidad**: *casting* controlado, categorías, strings (`.str.*`), outliers (IQR/Z-score), *asserts* y reglas con `eval/query`.

### Tema 5 — ML básico (scikit-learn)
- **Flujo**: `train_test_split` → *Pipeline* (`ColumnTransformer` + `Imputer/Scaler/OneHot`) → `fit/predict`.
- **Regresión**: lineal y regularizada (Ridge/Lasso/ElasticNet); árboles/*ensembles*; k-NN (escalado).
- **Clasificación**: logística, SVM (sensible a escala), árboles/RandomForest, k-NN.
- **Métricas**: regresión (MAE/MSE/RMSE/R²); clasificación (accuracy, precision/recall/F1, ROC-AUC/PR-AUC).
- **Validación**: K-Fold/Stratified, `GridSearchCV/RandomizedSearchCV`; curvas de aprendizaje/validación; evitar *leakage* (todo dentro del *pipeline*).
- **Operativa**: semillas (`random_state=42`), *joblib* para persistencia, versionado de artefactos/datos.

### Tema 6 — Dimensionalidad, clustering y anomalías
- **Reducción**: PCA (escalado previo; `n_components` por varianza explicada), SVD/IncrementalPCA; t-SNE/UMAP solo para visual.
- **Clustering**: k-means (k por codo/silhouette), DBSCAN (eps/min_samples; detecta *outliers*), jerárquico (dendrogramas).
- **Selección vs extracción**: filtros (VarianceThreshold/SelectKBest), RFE, importancias; extracción (PCA/ICA/SVD).
- **Pipelines**: `Scaler → PCA → (KMeans/Modelo)`; todo **dentro de CV**.
- **Anomalías**: IsolationForest/LOF/OneClassSVM; política tras detección (etiquetar/excluir/tratar aparte).

## Proyecto final

**Qué**: análisis end-to-end con Python (ingesta → limpieza → EDA → modelado → evaluación → reporte) usando *notebook* y un pipeline reproducible.

### Objetivo
- **Regresión**: predecir una variable continua con un modelo lineal (y variante mejorada).
- **Clasificación**: predecir presencia de enfermedad coronaria (CHD) con regresión logística.

### Datos
- **Esquema**: tipado explícito (num/cat/fechas) y diccionario breve por columna.
- **Calidad**: manejo de nulos/duplicados, normalización de *strings*, detección de outliers.

### Metodología
1. **Ingesta**: `pandas.read_*` con `dtype` y `parse_dates`.
2. **Limpieza**: reglas por columna; documentar cambios.
3. **EDA**: descriptivos, correlaciones, distribuciones; 3–5 figuras clave.
4. **Features**: *scaling* numérico, *one-hot* categórico; opcional: interacciones/polinomios.
5. **Modelos**
   - **Regresión**: Lineal (`LinearRegression`) → baseline; considerar Ridge/Lasso si hay colinealidad.
   - **Clasificación**: Logística (umbral 0.5 por defecto; evaluar ajuste si hay desbalance).
6. **Validación**: `train_test_split` (y **K-Fold/StratifiedKFold** para media±std).
7. **Evaluación**
   - Regresión: **MAE**, **RMSE**, **R²**.
   - Clasificación: **Accuracy**, **Precision/Recall/F1**, **ROC-AUC** (+ matriz de confusión).
8. **Reporte**: tablas de métricas, gráficas exportadas y conclusiones.


