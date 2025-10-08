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

- **Tipos y operaciones básicas**
  - Tipos primitivos: `int`, `float`, `bool`, `str`, `None`.
  - Operadores: aritméticos (`+ - * / // % **`), comparación (`== != < <= > >=`), lógicos (`and or not`).
  - *Truthiness*: `0`, `0.0`, `''`, `[]`, `{}`, `set()` y `None` son *falsy*.
  - *Casting* y conversión: `int('3')`, `float('2.5')`, `str(42)`, `bool(x)`.

- **Cadenas (strings) y módulos**
  - *Slicing*: `s[a:b:c]`, `s[::-1]` invierte.
  - Métodos útiles: `split`, `join`, `strip`, `replace`, `lower/upper`, `count`, `startswith/endswith`.
  - *f-strings* para formateo: `f"{nombre}: {valor:.2f}"`.
  - Imports básicos

- **Secuencias e iterables: Listas, Tuplas, Conjuntos, Diccionarios…**
  - **Listas**: mutables; `append`, `extend`, `insert`, `remove`, `pop`, `sort`/`sorted(key=..., reverse=...)`.
  - **Tuplas**: inmutables; *unpacking* rápido.
  - **Conjuntos (set)**: sin duplicados; operaciones de conjuntos.
  - **Diccionarios**: pares clave–valor; `.get`, `.items`, *dict comprehensions*.
  - **Comprehensions** (lista, set, dict) y generadores:

- **Control de flujo y funciones**
  - `if/elif/else`, `for` sobre iterables, `while` cuando importa la condición.
  - `range(start, stop, step)` para bucles numéricos.
  - **Funciones**: argumentos por defecto, *docstrings*, retorno claro.

### Tema 2 — Python (Parte 2)

- **Funciones y modularidad avanzada**
  - Parámetros por defecto, *args, **kwargs*, *docstrings* y *type hints*.
  - Importación y empaquetado: `__init__.py`, módulos reutilizables en `src/`.
  - Patrón de entrada: `if __name__ == "__main__":` para separar uso de librería vs ejecución.

- **Expresiones lambda y funciones de orden superior**
  - `map`, `filter`, `sorted(key=...)` y `functools.reduce`.


- **Iterables, iteradores y *comprehensions* “pro”**
  - *List/set/dict comprehensions* anidadas, generadores perezosos.
  - `zip`, `enumerate`, *unpacking* y *star expressions*.


- **Manejo de errores (excepciones)**
  - `try/except/else/finally`, crear excepciones propias.
  - Mantener el **contrato** de la función: captura solo lo que entiendas y re-lanza si procede.

- **Entrada/Salida (ficheros)**
  - Lectura/ escritura con *context manager* para evitar fugas: `with open(...) as f:`.
  - CSV/JSON con módulos estándar cuando no hay pandas.
---

### Tema 2 — NumPy (Parte 1)

- **ndarray: creación y `dtype`**
  - `np.array`, `np.arange`, `np.linspace`, `np.zeros/ones/full`, `np.random`.
  - Control del tipo: `dtype=np.float32` vs `float64` (memoria/precisión).

- **Forma, *strides* e indexación**
  - `shape`, `ndim`, `size`, `itemsize`, `strides` (noción).
  - Indexado básico y avanzado: *slicing*, máscaras booleanas, listas de índices.


- **Broadcasting**
  - Reglas de alineación para operar entre formas distintas (añadir ejes con `None`/`np.newaxis`).


- **uFuncs y vectorización**
  - Operaciones element-wise (`+ - * /`, `np.exp`, `np.sqrt`, etc.) y reducción (`sum`, `mean`, `min/max`) con `axis`.


- **Selección y asignación condicional (`np.where`)**



## Capítulo 3

### Tema 3.1 — Procesamiento de imágenes (Image processing)

- **Representación de imágenes como arrays**
  - Escala de grises: matriz `H×W` con intensidades (0–255 `uint8` o 0–1 `float`).
  - Color: `H×W×3` (RGB). Suele trabajarse en `float32` normalizado 0–1 para operaciones numéricas estables.
  - Conversión de tipo/escala: normalizar a `[0,1]` para procesar y volver a `uint8` al guardar.

- **Operaciones puntuales y por máscara**
  - Inversión/negativo, estiramiento de contraste, recortes/clipping.
  - Umbralado (binario) y máscaras booleanas para seleccionar/combinar regiones.
  - Condicionales vectorizadas (`A si cond, B si no`) sin bucles explícitos.

- **Filtros y convolución 2D**
  - Suavizado (promedio/gauss) para reducir ruido; realce/detección de bordes (Sobel/Prewitt).
  - Parámetros clave: tamaño del kernel, manejo de bordes (*padding*), modo de convolución.

- **Transformaciones geométricas básicas**
  - *Flip* horizontal/vertical, rotaciones por múltiplos de 90°, recorte y *resize* (atención a *aliasing*).

- **Histograma y contraste**
  - Cálculo de histogramas e interpretación de la distribución tonal.
  - Ecualización (idea): redistribuir intensidades para mejorar contraste global.

- **Buenas prácticas y verificación**
  - Documentar espacio de color esperado (gris vs RGB) y rango de valores.
  - Visualizar **antes/después** y validar con métricas simples (p. ej., RMS de bordes).

---

### Tema 3.2 — Matplotlib

- **API orientada a objetos (Figure/Axes)**
  - Crear figura y ejes; añadir series y configurar títulos, etiquetas y leyendas.
  - Uso de `tight_layout`/`constrained_layout` para evitar solapes.

- **Tipos de gráfico frecuentes**
  - Linea, dispersión, barras (vertical/horizontal), histograma, boxplot, imagen/heatmap.
  - Añadir *colorbar* cuando el color codifica magnitud continua (p. ej., `imshow`).

- **Composición y diseño**
  - *Subplots* en cuadrícula; ejes compartidos; insets y ejes secundarios (`twinx`).
  - Estilo y legibilidad: tamaños de fuente coherentes, cuadrículas discretas, *ticks* legibles.

- **Anotaciones y exportación**
  - Etiquetar puntos de interés con `annotate` (flechas/cajas).
  - Exportar con resolución adecuada y recorte de márgenes (`dpi`, `bbox_inches="tight"`).

- **Buenas prácticas**
  - Colormaps perceptualmente uniformes y *colorblind-friendly* cuando sea posible.
  - Leyendas que no tapen datos; ordenar capas (z-order) si hay solapes.

---

### Tema 3.3 — NumPy (Parte 2)

- **Indexación avanzada y máscaras**
  - Selección por listas/arrays de índices y filtrado booleano.
  - Extracción y modificación de subconjuntos sin bucles explícitos.

- **Broadcasting y ejes nuevos**
  - Reglas para operar entre arrays de formas distintas añadiendo ejes (`None/np.newaxis`).
  - Construcción eficiente de mallas y operaciones por filas/columnas con `axis`.

- **Agregaciones y transformaciones por eje**
  - Estadísticos (`sum/mean/std/min/max`) con control de ejes.
  - Estandarización por columnas/filas y uso de `keepdims` para mantener formas.

- **Ordenación, ranking e índices**
  - `sort`/`argsort` para obtener valores e índices ordenados.
  - `unique` (con `return_counts`) para deduplicar y contar.

- **Composición y particionado**
  - Concatenación apilada (`hstack/vstack/stack`) y división (`split/array_split`).
  - Diferencia entre *views* (slices comparten memoria) y copias explícitas.

- **Álgebra lineal y aleatoriedad moderna**
  - Producto matricial (`@`), mínimos cuadrados (`lstsq`) y nociones de descomposiciones.
  - Generador `default_rng` para muestreo reproducible.

- **Rendimiento**
  - Evitar bucles Python en el *hot path*: preferir vectorización.
  - Control de `dtype` para evitar *casts* y exceso de memoria.

---

### Tema 3.4 — Pandas (Parte 1)

- **Estructuras básicas**
  - **Series** (unidimensional con índice) y **DataFrame** (tabular con columnas heterogéneas).
  - Inspección rápida: `head/info/describe` y verificación de `dtypes`.

- **Entrada/Salida de datos**
  - Lectura desde CSV/JSON/Excel/Parquet; escritura recomendada en Parquet por rendimiento.
  - Especificación de `dtype` y `parse_dates` en lectura para tipar correctamente.

- **Selección y filtrado**
  - Acceso por etiqueta (`loc`) y posición (`iloc`); filtrado booleano.
  - Selección de columnas por tipo con `select_dtypes`.

- **Creación y transformación de columnas**
  - Operaciones vectorizadas, `assign` y *method chaining*.
  - Conversión de tipos: `to_numeric`, categorización (`astype('category')`).

- **Ordenación y deduplicación**
  - `sort_values` (con control de `ascending` y `na_position`) y `drop_duplicates`.

- **Valores faltantes**
  - `isna`/`notna` para diagnóstico; `fillna` (constante/mediana/moda) y `dropna` según contexto.
  - Evitar sesgos: decidir explícitamente la estrategia por columna.

## Capítulo 4

### Tema 4.1 — Pandas (Parte 2): **GroupBy, pivotes y *reshape***
- **Agrupar y agregar**
  - `df.groupby(keys)` seguido de agregaciones (`agg`, `mean`, `sum`, `nunique`, etc.).
  - Agregaciones múltiples por columna: `agg({'col1': ['mean','std'], 'col2': 'sum'})`.
  - **`as_index`**: controla si las claves de agrupación pasan a índice.
- **Transformaciones por grupo**
  - `transform` (devuelve misma forma que el grupo): z-score por grupo, *fillna* con media del grupo.
  - `apply` cuando necesitas lógica más libre (ojo a rendimiento).
- **Tablas dinámicas y reestructuración**
  - `pivot_table(values, index, columns, aggfunc='mean')` con `margins=True` para totales.
  - *Reshape*: `melt` (ancho→largo), `pivot` (largo→ancho), `stack/unstack` con MultiIndex.
- **MultiIndex**
  - Crear/ordenar niveles: `set_index([...])`, `sort_index(level=...)`, `swaplevel`.
  - Selección jerárquica: `loc[('A','x')]`, *slicing* por niveles (`IndexSlice`).
- **Tips**
  - Nombra columnas tras agregaciones múltiples: `df.columns = ['_'.join(map(str,c)) for c in df.columns]`.
  - Evita `apply` si es posible con `agg`/`transform` vectorizados.

---

### Tema 4.2 — **Combinación de datasets** (*merge/join/concat*)
- **Merge/Join (SQL-like)**
  - `pd.merge(left, right, on='key', how='inner|left|right|outer')`.
  - Claves con distinto nombre: `left_on='A', right_on='B'`.
  - *Many-to-one* vs *many-to-many*: comprueba duplicados en claves antes de unir.
- **Concatenación**
  - Apilar filas: `pd.concat([df1, df2], axis=0, ignore_index=True)`.
  - Juntar columnas: `axis=1` (asegura índices alineados o *reset_index* antes).
- **Uniones especiales**
  - *Join por índice*: `df1.join(df2, how='left')` (requiere índices significativos).
  - *As-of merge* (series temporales ordenadas): `pd.merge_asof` con tolerancias.
- **Validación**
  - `validate='one_to_one'|'one_to_many'...` en `merge` para detectar duplicidades inesperadas.
  - Tras el *merge*, cuenta nulos en columnas clave para hallar claves huérfanas.

---

### Tema 4.3 — **Fechas y series temporales**
- **Tipos de fecha/hora**
  - Parseo: `pd.to_datetime`, parámetros `format`, `dayfirst`, `utc`.
  - Accesor `dt`: `dt.year`, `dt.month`, `dt.day_name()`, `dt.tz_convert`.
- **Índices temporales**
  - `set_index('fecha')` y `sort_index()` para habilitar *resample* eficiente.
- **Resample y ventanas**
  - `resample('D'|'W'|'M').agg(...)` para cambiar frecuencia.
  - Ventanas móviles: `rolling(window).mean()`, *expanding*, *ewm* (media exponencial).
- **Operaciones útiles**
  - Diferencias y *lags*: `diff`, `pct_change`, `shift(periods)`.
  - Reindex a calendario completo y *forward-fill* de huecos: `reindex(...).ffill()`.

---

### Tema 4.4 — **Limpieza y validación de datos**
- **Tipos y *casting***
  - `astype` controlado; numéricos seguros con `pd.to_numeric(errors='coerce')`.
  - Categorías para cardinalidad alta: `astype('category')` (memoria y velocidad).
- **Valores faltantes**
  - Diagnóstico: `isna().mean()` por columna (porcentaje de nulos).
  - Imputación: `fillna` (constante/media/mediana por grupo con `groupby().transform`).
  - Decide explícitamente por columna (evita mezclas silenciosas).
- **Strings y normalización**
  - Accesor `str`: `strip`, `lower`, `replace`, *regex*, división en columnas.
  - Estandariza *encodings* y espacios en blanco antes de agrupar/unir.
- **Outliers (detección básica)**
  - IQR: *fences* Q1−1.5·IQR / Q3+1.5·IQR; Z-score simple.
  - Etiqueta y reporta más que eliminar sin justificación.
- **Validación y *asserts***
  - `df.eval` + `df.query` para reglas; `assert df['id'].is_unique`.
  - Comprobaciones de rango, dominios categóricos y sin nulos en claves.

---

### Tema 4.5 — **Estadística descriptiva y muestreo**
- **Descriptivos**
  - `describe(include='all')`, `quantile([.1,.5,.9])`, `value_counts(normalize=True)`.
  - Correlación/covarianza: `corr`, `cov` (ojo a nulos y escalas).
- **Muestreo reproducible**
  - `df.sample(n|frac, random_state=42)`. Estratificado: `df.groupby('estrato').apply(lambda g: g.sample(frac=...))`.
- **Estandarización y *scaling***
  - *Z-score* manual por columnas, o `sklearn.preprocessing.StandardScaler` si procede.
- **Bootstrap (idea)**
  - Re-muestrear con reemplazo para estimar incertidumbre de medias/medianas.

---

### Tema 4.6 — **Visualización aplicada (Matplotlib avanzado)**
- **Composición**
  - `subplots` con *shared axes*, *insets* y `twinx` para magnitudes distintas.
  - *Layouts*: `constrained_layout=True` o `tight_layout()`.
- **Gráficos útiles para EDA**
  - Histogramas con *bins* adecuados; *ECDF* (curva acumulada simple); *heatmaps* con `imshow`/`pcolormesh`.
  - Mapas de *missingness* sencillos: `imshow(df.isna(), aspect='auto')` (+ *colorbar*).
- **Anotación y exportación**
  - `annotate` para outliers/puntos clave; controla `zorder` para capas.
  - Exporta con `dpi` alto y `bbox_inches='tight'`; fija límites/etiquetas legibles.

---

### Tema 4.7 — **Buenas prácticas del capítulo**
- Define un **contrato de datos** (tipos, dominios, claves) y valida antes de analizar.
- Evita *`apply` fila a fila* en el *hot path*; prefiere `vectorización/agg/transform`.
- Tras un `merge`, **audita nulos y duplicados**; guarda *checks* como celdas/funciones reutilizables.
- Documenta decisiones de limpieza e imputación (por columna) y conserva un **diccionario de datos**.
- Versiona datos intermedios (parquet) y fija `random_state` para reproducibilidad.

## Capítulo 5

### Tema 5.1 — **Introducción a *scikit-learn* y flujo de trabajo de ML**
- **Patrón básico**: `fit(X_train, y_train) → predict(X_test)`; estimadores y *transformers*.
- **Partición de datos**: `train_test_split(test_size=..., stratify=y, random_state=42)`.
- **Preprocesado**:
  - Escalado: `StandardScaler` / `MinMaxScaler` (imprescindible para modelos basados en distancia/regularización).
  - Codificación categórica: `OneHotEncoder(handle_unknown='ignore')`.
  - Imputación: `SimpleImputer(strategy='mean'|'median'|'most_frequent')`.
- **Pipelines**:
  - `Pipeline([('prep', preprocessor), ('model', estimator)])` para evitar *data leakage*.
  - `ColumnTransformer` para aplicar transformaciones por tipo de columna.

---

### Tema 5.2 — **Modelos supervisados (regresión)**
- **Lineal**: `LinearRegression` (MSE/R²). Supuestos: linealidad, homocedasticidad (diagnosticar con residuales).
- **Regularizados**:
  - `Ridge` (L2) y `Lasso` (L1, induce *sparsity*); `ElasticNet` (mix L1/L2).
  - Selección de α: `RidgeCV`/`LassoCV` o *grid search*.
- **k-NN Regressor**: sensible a escala; elegir `n_neighbors` por validación.
- **Árboles y *ensembles***:
  - `DecisionTreeRegressor` (propenso a *overfitting* sin poda).
  - `RandomForestRegressor` (robusto, buen *baseline*), `GradientBoostingRegressor`/`XGB` (si se permite).

**Métricas de regresión**: `mean_squared_error`, `mean_absolute_error`, `r2_score`.
- Reporta siempre un *baseline* (p.ej., media) para contextualizar.

---

### Tema 5.3 — **Modelos supervisados (clasificación)**
- **Regresión logística**: lineal en *log-odds*; regularización por defecto (`C` inverso de la fuerza).
- **k-NN Classifier**: requiere escalado; `weights='distance'` como alternativa.
- **Árboles/RandomForest**: manejan *features* no escaladas; interpretables vía *feature importance*.
- **SVM (lineal/RBF)**: potente con buenos *features*; sensible a escala y a `C`, `gamma`.

**Métricas de clasificación**:
- Exactitud (`accuracy`) ≠ suficiente con clases desbalanceadas.
- `precision`, `recall`, `f1`, *confusion matrix*; `roc_auc_score` (binaria), *PR AUC* si hay desbalance.
- Curvas ROC/PR: `plot_roc_curve` (o `from sklearn.metrics import RocCurveDisplay`).

---

### Tema 5.4 — **Validación, *cross-validation* e hiperparámetros**
- **K-Fold / StratifiedKFold**: estratificar en clasificación; fija `random_state`.
- **Búsquedas**:
  - `GridSearchCV` (exhaustiva) y `RandomizedSearchCV` (más eficiente).
  - Evalúa sobre *pipeline completo* para evitar *leakage*.
- **Learning/Validation Curves**:
  - *Under/overfitting*: inspecciona `train_score` vs `val_score`.
  - Ajusta complejidad del modelo (profundidad del árbol, `C`, `n_neighbors`, etc.).
- **Selección de modelo**: balancea rendimiento, interpretabilidad y coste computacional.

---

### Tema 5.5 — **Ingeniería de *features* y preparación de datos**
- **Numéricas**: *log-transform*, *binning* (con cuidado), interacciones/polinomios: `PolynomialFeatures`.
- **Categóricas**: *one-hot*; considera cardinalidad alta (hashing trick si aplica).
- **Escalado por grupo**: `groupby().transform` en pandas (si ML por grupo no es viable).
- **Leakage**: TODA transformación aprendida debe estar dentro del `Pipeline`.
- **Selección de *features***:
  - Filtros (`SelectKBest`), *wrappers* (RFE), *embeddings* del modelo (importancias).

---

### Tema 5.6 — **Evaluación robusta y *model diagnostics***
- **Divisiones repetidas**: *repeated CV* para estimar varianza de la métrica.
- **Intervalos de confianza**: *bootstrap* de métricas (si procede).
- **Calibración de probabilidades**: `CalibratedClassifierCV` (Platt/Isotonic).
- **Curvas de aprendizaje**: detecta si necesitas más datos o más capacidad.
- **Error analysis**: inspecciona falsos positivos/negativos con ejemplos concretos.

---

### Tema 5.7 — **Persistencia, reproducibilidad y despliegue ligero**
- **Semillas**: fija `numpy.random.default_rng(42)` / `random_state=42`.
- **Persistencia**: `joblib.dump(model, 'model.joblib')` / `load`.
- **Versionado**: guarda *artefactos* (modelo, *scaler*, *encoder*) y su *hash* de datos.
- **Inferencia**: función `predict(df_raw)` que ejecute el *pipeline* end-to-end.
- **Trazabilidad**: registra *params*, métricas y fecha (CSV/MLflow sencillo).

---

## Capítulo 6

### Tema 6.1 — **Reducción de dimensionalidad**
- **Motivación**
  - Quitar redundancia/ruido, acelerar modelos, facilitar visualización y mitigar la maldición de la dimensionalidad.
- **PCA (Análisis de Componentes Principales)**
  - Requiere **escalado** previo (`StandardScaler`) para que todas las variables pesen igual.
  - Elige nº de componentes por **varianza explicada acumulada** (p. ej., `n_components=0.95`).
  - Interpretación: *loadings* (aportación de cada variable a cada componente).
- **SVD y PCA incremental**
  - `TruncatedSVD` para datos *sparse*; `IncrementalPCA` para lotes y datasets grandes.
- **Proyecciones no lineales (visión)**
  - t-SNE/UMAP para visualización 2D/3D; no usar como *preprocessing* general de modelos.
- **Buenas prácticas**
  - Mantén un pipeline `Scaler → PCA → Modelo`; guarda los **componentes** para trazar *biplots* y analizar.

---

### Tema 6.2 — **Clustering (no supervisado)**
- **k-means**
  - Hiperparámetros: `n_clusters`, init `k-means++`, `n_init`.
  - Elección de *k*: codo (inercia) y **silhouette** (mejor ↑).
  - Sensible a escala y a formas no esféricas; inicialización afecta estabilidad.
- **DBSCAN**
  - Densidad: `eps` (radio) y `min_samples`; detecta *outliers* (label −1).
  - Ventaja: descubre formas arbitrarias; no requiere *k*.
- **Clustering jerárquico**
  - `AgglomerativeClustering` (vinculación completa/promedio/ward).
  - Dendrograma para inspección de niveles (scipy).
- **Evaluación**
  - Internas: **silhouette**, Davies-Bouldin, Calinski-Harabasz.
  - Externas (si hay etiquetas): ARI/NMI/FMI.

---

### Tema 6.3 — **Selección vs. extracción de *features***
- **Selección (mantener originales)**
  - Filtro: `VarianceThreshold`, `SelectKBest` (p-valores/score univariante).
  - *Wrappers*: RFE/RFECV con un estimador.
  - Embebida: importancias en árboles/*ensembles* o coeficientes regularizados.
- **Extracción (crear nuevas)**
  - PCA/ICA/TruncatedSVD; útiles cuando hay multicolinealidad o ruido.
- **Consejo**
  - En pipelines: `ColumnTransformer` para tratar numéricas/categóricas y encadenar selección/extracción.

---

### Tema 6.4 — **Pipelines no supervisados y mixtos**
- **Encadenar pasos**
  - Ejemplo: `StandardScaler → PCA → KMeans` (para explorar estructura y luego visualizar).
  - Supervisado con reducción previa: `Scaler → PCA → LogisticRegression`.
- **Validación adecuada**
  - Ajusta PCA/selección **dentro** de CV para evitar *data leakage*.
  - Reporta varianza explicada y estabilidad de clusters (varias semillas).

---

### Tema 6.5 — **Visualización de alta dimensión**
- **Matriz de dispersión y proyecciones**
  - *Scatter matrix* (pares de variables) para ≤10 columnas.
  - Proyección PCA 2D/3D con colores por cluster/etiqueta; añadir **explicada%** a ejes.
- **t-SNE/UMAP (exploración)**
  - t-SNE: sensible a `perplexity` y *learning rate*; no preserva distancias globales.
  - Usar solo como apoyo visual, no para métricas/entrenamiento final.

---

### Tema 6.6 — **Detección de anomalías (intro)**
- **Modelos comunes**
  - `IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`.
- **Flujo**
  - Escalado → modelo → marcar *outliers* → revisar manualmente → decidir política (etiquetar, excluir o tratar por separado).
- **Métricas**
  - Si hay *ground truth*: *precision/recall* sobre clase minoritaria; si no, inspección dirigida con *scores* y percentiles.

---


