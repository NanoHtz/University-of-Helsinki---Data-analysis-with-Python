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

