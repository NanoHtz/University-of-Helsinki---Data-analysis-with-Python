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
  - Ejemplo:
    ```python
    x = 5
    y = 2
    div = x / y      # 2.5 (float)
    flo = x // y     # 2   (floor division)
    mod = x % y      # 1
    pow_ = x ** y    # 25
    ```

- **Cadenas (strings) y módulos**
  - *Slicing*: `s[a:b:c]`, `s[::-1]` invierte.
  - Métodos útiles: `split`, `join`, `strip`, `replace`, `lower/upper`, `count`, `startswith/endswith`.
  - *f-strings* para formateo: `f"{nombre}: {valor:.2f}"`.
  - Import básico:
    ```python
    import math as m
    from random import randint
    print(m.pi, randint(1, 10))
    ```

- **Secuencias e iterables: Listas, Tuplas, Conjuntos, Diccionarios…**
  - **Listas**: mutables; `append`, `extend`, `insert`, `remove`, `pop`, `sort`/`sorted(key=..., reverse=...)`.
    ```python
    nums = [3, 1, 2]
    nums.sort()                 # [1, 2, 3]
    sorted_nums = sorted(nums, key=lambda x: -x)  # [3, 2, 1]
    ```
  - **Tuplas**: inmutables; *unpacking* rápido.
    ```python
    a, b = (1, 2)
    ```
  - **Conjuntos (set)**: sin duplicados; operaciones de conjuntos.
    ```python
    A, B = {1,2,3}, {3,4}
    A | B, A & B, A - B  # unión, intersección, diferencia
    ```
  - **Diccionarios**: pares clave–valor; `.get`, `.items`, *dict comprehensions*.
    ```python
    d = {"a": 1, "b": 2}
    d.get("c", 0)        # 0 por defecto
    inv = {v: k for k, v in d.items()}
    ```
  - **Comprehensions** (lista, set, dict) y generadores:
    ```python
    squares = [i*i for i in range(10) if i % 2 == 0]
    uniques = {c.lower() for c in "AaBb"}
    mapping = {w: len(w) for w in ["foo", "bar"]}
    gen = (i*i for i in range(10))  # perezoso
    ```

- **Control de flujo y funciones**
  - `if/elif/else`, `for` sobre iterables, `while` cuando importa la condición.
  - `range(start, stop, step)` para bucles numéricos.
  - **Funciones**: argumentos por defecto, *docstrings*, retorno claro.
    ```python
    def normalize(s: str) -> str:
        """Limpia espacios múltiples y pasa a minúsculas."""
        return " ".join(s.split()).lower()
    ```
  - Patrón para separar librería de ejecución:
    ```python
    def main():
        ...

    if __name__ == "__main__":
        main()
    ```

- **Buenas prácticas**
  - **No mutar** entradas sin necesidad (trabaja sobre copias o documenta el comportamiento).
  - Funciones **pequeñas** y **puras** cuando sea posible (mismo input → mismo output, sin efectos laterales).
  - Nombres explícitos; *docstrings* breves indicando contrato de la función.
  - Estructura los *notebooks*: objetivo → setup/imports → pasos → resultados → conclusiones.
  - Añade tests mínimos a utilidades reutilizables (p. ej., en `tests/` con `pytest`).

- **Patrones útiles (Cap. 1)**
  - **Ordenar por clave**:
    ```python
    records = [{"name":"a", "age":30}, {"name":"b", "age":20}]
    by_age = sorted(records, key=lambda r: r["age"])
    ```
  - **Contar frecuencias**:
    ```python
    from collections import Counter
    cnt = Counter(["a","b","a","c","b","a"])
    cnt.most_common(2)  # [('a', 3), ('b', 2)]
    ```
  - **Limpieza básica**:
    ```python
    def to_ints(xs):
        return [int(x.strip()) for x in xs if x.strip().isdigit()]
    ```
  - **Detectar consecutivos (runs)** — idea central usada en `detect_ranges`:
    ```python
    def detect_ranges(nums):
        if not nums:
            return []
        nums = sorted(nums)
        out = []
        start = prev = nums[0]
        for x in nums[1:]:
            if x != prev + 1:              # hay salto → cerramos run
                out.append((start, prev+1) if start != prev else start)
                start = x
            prev = x
        out.append((start, prev+1) if start != prev else start)
        return out

    # Ejemplo:
    # detect_ranges([2,5,4,8,12,6,7,10,13]) -> [(2, 3), (4, 8), 10, (12, 14)]
    ```
### Tema 2 — Python (Parte 2)

- **Funciones y modularidad avanzada**
  - Parámetros por defecto, *args, **kwargs*, *docstrings* y *type hints*.
  - Importación y empaquetado: `__init__.py`, módulos reutilizables en `src/`.
  - Patrón de entrada: `if __name__ == "__main__":` para separar uso de librería vs ejecución.

- **Expresiones lambda y funciones de orden superior**
  - `map`, `filter`, `sorted(key=...)` y `functools.reduce`.
  - Ejemplo:
    ```python
    words = ["AA", "bbb", "c"]
    by_len_desc = sorted(words, key=lambda w: (-len(w), w))
    ```

- **Iterables, iteradores y *comprehensions* “pro”**
  - *List/set/dict comprehensions* anidadas, generadores perezosos.
  - `zip`, `enumerate`, *unpacking* y *star expressions*.
    ```python
    pairs = [(i, j) for i in range(3) for j in range(i)]
    idx_vals = list(enumerate(["a","b","c"], start=1))
    ```

- **Manejo de errores (excepciones)**
  - `try/except/else/finally`, crear excepciones propias.
  - Mantener el **contrato** de la función: captura solo lo que entiendas y re-lanza si procede.
    ```python
    try:
        x = int(s)
    except ValueError as e:
        raise ValueError(f"Entrada inválida: {s}") from e
    ```

- **Entrada/Salida (ficheros)**
  - Lectura/ escritura con *context manager* para evitar fugas: `with open(...) as f:`.
  - CSV/JSON con módulos estándar cuando no hay pandas.
    ```python
    import json
    with open("data.json") as f:
        cfg = json.load(f)
    ```

- **Testing ligero y calidad**
  - Pruebas mínimas con `pytest`, *asserts* y casos borde.
  - Formateo/estilo: `black`, `isort`, `flake8`.

- **Patrones útiles**
  - **Orden estable con múltiples claves**:
    ```python
    data = [{"n":"Ana","age":30},{"n":"Ana","age":25},{"n":"Bob","age":25}]
    data = sorted(data, key=lambda r: (r["n"], r["age"]))
    ```
  - **Agrupar sin pandas**:
    ```python
    from itertools import groupby
    data = sorted(data, key=lambda r: r["n"])
    groups = {k: list(g) for k, g in groupby(data, key=lambda r: r["n"])}
    ```
  - **Generador perezoso**:
    ```python
    def chunked(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i+n]
    ```

---

### Tema 2 — NumPy (Parte 1)

- **ndarray: creación y `dtype`**
  - `np.array`, `np.arange`, `np.linspace`, `np.zeros/ones/full`, `np.random`.
  - Control del tipo: `dtype=np.float32` vs `float64` (memoria/precisión).
    ```python
    import numpy as np
    a = np.arange(12, dtype=np.int32).reshape(3,4)
    ```

- **Forma, *strides* e indexación**
  - `shape`, `ndim`, `size`, `itemsize`, `strides` (noción).
  - Indexado básico y avanzado: *slicing*, máscaras booleanas, listas de índices.
    ```python
    m = a[:, 1:3]          # slicing por columnas
    sel = a[a % 2 == 0]    # máscara booleana
    ```

- **Broadcasting**
  - Reglas de alineación para operar entre formas distintas (añadir ejes con `None`/`np.newaxis`).
    ```python
    x = np.arange(3)        # (3,)
    y = np.arange(4)[:,None]# (4,1)
    grid = x + y            # (4,3) por broadcasting
    ```

- **uFuncs y vectorización**
  - Operaciones element-wise (`+ - * /`, `np.exp`, `np.sqrt`, etc.) y reducción (`sum`, `mean`, `min/max`) con `axis`.
    ```python
    X = np.random.randn(1000, 50)
    col_means = X.mean(axis=0)
    row_norms = np.sqrt((X*X).sum(axis=1))
    ```

- **Selección y asignación condicional (`np.where`)**
  ```python
  z = np.where(X > 0, X, 0.0)  # ReLU simple


## Capítulo 3

### Tema 3.1 — Procesamiento de imágenes (Image processing)

**Objetivo:** tratar imágenes como **arrays NumPy** para filtrado, máscaras, transformaciones y mejora de contraste.

**Representación y conversiones**
- Escala de grises: `H x W` (`uint8` 0–255 o `float` 0–1).
- Color (RGB): `H x W x 3`. Trabaja en `float32` normalizado `[0,1]` durante el procesado.
- Conversión rápida entre `uint8` y `float[0,1]`:
    
    import numpy as np
    
    def to_float01(img_u8: np.ndarray) -> np.ndarray:
        return img_u8.astype(np.float32) / 255.0
    
    def to_u8(img_f: np.ndarray) -> np.ndarray:
        return (np.clip(img_f, 0, 1) * 255).astype(np.uint8)

**Operaciones básicas**
    
    # Negativo y estiramiento de contraste
    inv = 1.0 - img_f
    stretched = (img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-8)
    
    # Umbralado (binario)
    mask = (img_f > 0.5).astype(np.float32)  # 0/1

**Convolución / filtros** (suavizado, bordes)
    
    from scipy.signal import convolve2d
    import numpy as np
    
    blur = np.ones((5,5), dtype=np.float32) / 25.0
    img_blur = convolve2d(img_f, blur, mode="same", boundary="symm")
    
    # Gradiente tipo Sobel
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    sobel_y = sobel_x.T
    gx = convolve2d(img_f, sobel_x, mode="same", boundary="symm")
    gy = convolve2d(img_f, sobel_y, mode="same", boundary="symm")
    mag = np.sqrt(gx*gx + gy*gy)

**Transformaciones**
- Recortes, *flip* horizontal/vertical, rotaciones 90°.
- *Resize* (cuidado con aliasing).
- Efectos por condición: `np.where(cond, A, B)`.

**Histogramas y contraste**
    
    hist, bins = np.histogram(img_f, bins=256, range=(0,1))
    # Ecualización: usar la CDF para mapear intensidades al rango completo.

**Buenas prácticas**
- Procesa en **float [0,1]**; convierte a `uint8` solo al guardar.
- Documenta si esperas **grises** o **RGB** y si devuelves **copia** o **vista**.
- Controla bordes en convolución (`boundary="symm"`/`"fill"`).

**Checklist**
- [ ] Tipo/escala correctos (`uint8` ↔ `float`).
- [ ] Tratamiento de bordes en filtros.
- [ ] Máscaras coherentes (bool o 0/1).
- [ ] Visualización antes/después para validar.

---

### Tema 3.2 — Matplotlib

**Objetivo:** dominar la **API orientada a objetos** (Figure/Axes) para gráficos claros y reproducibles.

**Patrón base**
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, y, label="serie")
    ax.set(title="Título", xlabel="X", ylabel="Y")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

**Subplots y layouts**
    
    fig, axs = plt.subplots(2, 2, figsize=(8,6), sharex=True)
    axs[0,0].hist(data, bins=30)
    axs[0,1].scatter(x, y, s=10, alpha=0.8)
    axs[1,0].plot(t, s)
    im = axs[1,1].imshow(img, cmap="gray")
    fig.colorbar(im, ax=axs[1,1])
    plt.tight_layout()

**Gráficos frecuentes**
- `plot`, `scatter`, `bar`/`barh`, `hist`, `boxplot`, `imshow`, `stairs`.

**Anotaciones / ejes secundarios / insets**
    
    ax.annotate("pico", xy=(x0,y0), xytext=(x0+1,y0+1),
                arrowprops=dict(arrowstyle="->", lw=1))
    ax2 = ax.twinx()  # segunda escala Y

**Exportar**
    
    plt.savefig("fig.png", dpi=200, bbox_inches="tight")

**Estilo y legibilidad**
- Tamaños de fuente y *ticks* legibles; `tight_layout()` o `constrained_layout=True`.
- Leyendas sin tapar datos; considera `loc` y `bbox_to_anchor`.

**Checklist**
- [ ] Título / ejes / unidades.
- [ ] Leyenda sin tapar datos.
- [ ] Colormap legible (idealmente *colorblind-friendly*).
- [ ] Resolución suficiente al exportar.

---

### Tema 3.3 — NumPy (Parte 2)

**Objetivo:** profundizar en **indexación avanzada**, **broadcasting**, agregaciones por ejes, concatenación, *views* vs **copias** y álgebra lineal.

**Indexación avanzada & máscaras**
    
    import numpy as np
    
    X = np.arange(16).reshape(4,4)
    sel = X[[0, 2], :][:, [1, 3]]  # filas 0,2 y columnas 1,3
    mask = (X % 2 == 0)
    even = X[mask]

**Broadcasting & ejes nuevos**
    
    x = np.arange(3)                # (3,)
    y = np.arange(4)[:, None]       # (4,1)
    grid = x + y                    # (4,3) por broadcasting

**Agregaciones por ejes**
    
    col_mean = X.mean(axis=0, keepdims=True)
    col_std  = X.std(axis=0, keepdims=True, ddof=0)
    X_std = (X - col_mean) / np.where(col_std == 0, 1, col_std)

**Ordenación / índices / únicos**
    
    scores = np.array([.3, .9, .1, .7, .5])
    idx = np.argsort(scores)        # ascendente
    top3 = idx[-3:][::-1]           # top-3 descendente
    vals, counts = np.unique(labels, return_counts=True)

**Concatenar / *stack* / dividir**
    
    Z = np.hstack([A, B])           # columnas
    W = np.vstack([A, B])           # filas
    parts = np.array_split(x, 4)

**Aleatoriedad moderna**
    
    rng = np.random.default_rng(0)
    sample = rng.normal(size=(100, 3))

**Álgebra lineal**
    
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)  # mínimos cuadrados

**Rendimiento y memoria**
- Minimiza copias; usa *slices* (views) cuando sea seguro.
- Vectoriza; evita bucles Python en zonas calientes.
- Revisa `dtype` explícito; los *casts* implícitos cuestan.

**Checklist**
- [ ] `shape`/`axis` coherentes.
- [ ] Documentar si devuelves **vista** o **copia**.
- [ ] Semilla reproducible (`default_rng(seed)`).

---

### Tema 3.4 — Pandas (Parte 1)

**Objetivo:** trabajar con **Series** y **DataFrames**: E/S, inspección, selección, *dtypes*, nulos, nuevas columnas y ordenación.

**E/S de datos**
    
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv("data.csv")           # también read_json/parquet/excel
    df.to_parquet("data.parquet", index=False)

**Series y DataFrame**
    
    s = pd.Series([10, 20, 30], name="x")
    df = pd.DataFrame({"a":[1,2,3], "b":[4,5,6]})

**Inspección y *dtypes***
    
    df.info()
    df.describe(include="all")
    df["col"] = pd.to_numeric(df["col"], errors="coerce")
    df["cat"] = df["cat"].astype("category")

**Selección**
    
    df.loc[5:10, ["a","b"]]        # por etiqueta
    df.iloc[:3, :2]                # por posición
    df[df["a"] > 0]                # filtrado booleano

**Nuevas columnas y transformaciones**
    
    df["z"] = df["x"] / df["y"].replace(0, np.nan)
    df = df.assign(ratio=lambda d: d["x"] / d["y"].where(d["y"] != 0, np.nan))

**Ordenar y deduplicar**
    
    df.sort_values(["key1","key2"], ascending=[True, False], na_position="last")
    df.drop_duplicates(subset=["id"], keep="last")

**Valores faltantes**
    
    df.isna().mean()                            # % de nulos por columna
    df["col"].fillna(df["col"].median(), inplace=True)

**Buenas prácticas**
- Evita bucles; usa operaciones vectorizadas y métodos pandas.
- Define `dtype` al leer para mejor memoria/velocidad.
- Documenta supuestos de limpieza (qué se trató como nulo, codificaciones, etc.).

**Checklist**
- [ ] `info()` sin sorpresas: tipos correctos.
- [ ] Filtrado/ordenación deterministas (`ascending`, `na_position`).
- [ ] No pisar datos originales sin querer (usa copias si hace falta).


## Capítulo 4 — Pandas (Parte 2)

**Objetivo:** profundizar en **Pandas** para análisis “real”: `groupby/agg/transform`, `merge`/`join`, *reshaping* (`pivot`/`melt`), **MultiIndex**, **series temporales** (resample/rolling/expanding), cadenas y categóricos, manejo de nulos/outliers y rendimiento.

---

### 4.1 Agrupación y agregaciones

- **`groupby(...).agg(...)`**: estadísticas por grupo (una o varias funciones, y con nombres claros).
- **`transform(...)`**: devuelve una **serie del mismo tamaño** que el grupo; ideal para estandarizar por grupo, *shares* o *target encoding*.
- **`apply(...)`**: máxima flexibilidad (más lento; úsalo cuando `agg/transform` no sirven).

**Patrones:**
- Media/desviación por grupo con nombres de salida:
    
    df.groupby("city").agg(
        temp_mean=("temp","mean"),
        temp_std =("temp","std"),
        hum_max  =("hum","max")
    )
- Z-score por grupo (misma longitud que el DF):
    
    df["temp_z"] = df.groupby("city")["temp"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0)
    )
- Rango por grupo con `transform`:
    
    df["temp_range_by_city"] = df.groupby("city")["temp"].transform(lambda s: s.max() - s.min())

**Consejos:** prefiere `agg/transform` a `apply` por rendimiento; revisa alineación de índices tras agrupar.

---

### 4.2 Joins / Merge

- **Tipos:** `how="left"` (el más común), `inner`, `right`, `outer`.
- **Claves:** `on="col"` o `left_on`/`right_on`; con índices: `left_index=True`, `right_index=True`.
- **Especiales:** `merge_asof` (por tiempo cercano), `merge_ordered` (series ordenadas).

**Patrón:**
    
    users  = pd.DataFrame({"id":[1,2,3], "name":["Ana","Bob","Cris"]})
    orders = pd.DataFrame({"user_id":[1,1,2], "total":[100,50,75]})
    dfm = users.merge(orders, left_on="id", right_on="user_id", how="left").drop(columns=["user_id"])

**Consejos:** valida **duplicados** en claves antes del *merge* (pueden multiplicar filas); tras el join, trata nulos introducidos.

---

### 4.3 Reshaping: `pivot`, `pivot_table`, `melt`

- **`pivot`**: largo → ancho (requiere pares únicos *index–column*).
- **`pivot_table`**: como `pivot` pero con **agregación** (acepta duplicados, `aggfunc` y `margins=True`).
- **`melt`**: ancho → largo (formato *tidy*).

**Patrón:**
    
    sales = pd.DataFrame({
        "city":["A","A","B","B"],
        "year":[2023,2024,2023,2024],
        "revenue":[10,12,8,11]
    })
    wide = sales.pivot(index="city", columns="year", values="revenue")
    pt   = sales.pivot_table(index="city", columns="year", values="revenue", aggfunc="sum")
    long = wide.reset_index().melt(id_vars="city", var_name="year", value_name="revenue")

**Consejos:** rellena nulos tras `pivot` (`.fillna(...)`) y ordena índices/columnas.

---

### 4.4 MultiIndex

- Índices jerárquicos en filas/columnas; útiles para datos por **múltiples dimensiones** (ej. `city`, `year`).
- Operaciones: `stack`/`unstack`, `swaplevel`, `sort_index`, `xs` (selección por nivel).

**Aplanar columnas MultiIndex:**
    
    pt_flat = pt.copy()
    pt_flat.columns = [f"revenue_{c}" for c in pt.columns]

**Consejos:** nombra niveles (`index.name`, `columns.name`) para claridad; aplana si entorpece el trabajo.

---

### 4.5 Series temporales

- **Fechas:** `pd.to_datetime`, *accessor* `.dt`, *timezones* (`dt.tz_localize/convert`).
- **Resampling:** cambiar frecuencia (diario→mensual): `resample('M').agg(...)`.
- **Ventanas:** `rolling` (móvil), `expanding` (acumulado), `ewm` (exponencial).

**Patrones:**
    
    ts   = pd.Series([1,2,0,3,4,6,2], index=pd.date_range("2025-01-01", periods=7, freq="D"))
    m    = ts.resample("2D").sum()
    roll = ts.rolling(window=3).mean()
    ema  = ts.ewm(span=3, adjust=False).mean()

**Consejos:** define **zona horaria** si mezclas fuentes; para emparejar por tiempo, usa `merge_asof`.

---

### 4.6 Texto y categóricos

- **Texto (`.str`)**: `lower/upper`, `strip`, `contains`, `extract` (regex), `split`, `replace`.
- **Categóricos:** ahorran memoria y permiten **orden** (impacta en `sort_values`, `groupby`).

**Patrones:**
    
    s_clean = df["name"].str.strip().str.lower()
    df["level"] = pd.Categorical(df["level"], categories=["low","medium","high"], ordered=True)

---

### 4.7 Nulos y outliers

- **Nulos:** `isna`, `fillna` (constante, mediana, *ffill/bfill*), `dropna`.
- **Outliers (básico):** cuantiles/IQR; *winsorizing*.

**Patrón IQR:**
    
    q1, q3 = df["temp"].quantile([0.25, 0.75])
    iqr    = q3 - q1
    lo,hi  = q1 - 1.5*iqr, q3 + 1.5*iqr
    df["temp_capped"] = df["temp"].clip(lo, hi)

---

### 4.8 Ordenación, ranking y duplicados

- **Ordenar:** `sort_values(["city","temp"], ascending=[True, False], na_position="last")`
- **Ranking:** `df["rank"] = df["temp"].rank(method="dense", ascending=False)`
- **Duplicados:** `drop_duplicates(subset=["id"], keep="last")`

---

### 4.9 Rendimiento y memoria

- **Evita bucles**; usa métodos vectorizados de pandas/NumPy.
- **Tipos:** especifica `dtype` al leer (`read_csv(..., dtype=...)`); usa *categorical* en variables discretas.
- **I/O:** **Parquet** para rapidez y eficiencia.
- **Copias:** sé explícito (`.copy()`) para evitar `SettingWithCopyWarning`.
- **Eval:** `pd.eval`/`DataFrame.eval` pueden acelerar expresiones.

**Patrón lectura robusta:**
    
    dtypes = {"city":"category", "hum":"float32", "temp":"float32"}
    df = pd.read_csv("data.csv", dtype=dtypes, parse_dates=["date"])

---

### 4.10 Snippets útiles

- **Top-N por grupo:**
    
    top3 = (df.sort_values(["city","temp"], ascending=[True, False])
              .groupby("city")
              .head(3))
- **Porcentaje dentro del grupo (share):**
    
    df["hum_share"] = df["hum"] / df.groupby("city")["hum"].transform("sum")
- **Target encoding simple (media por grupo):**
    
    enc = df.groupby("city")["temp"].mean().rename("temp_enc")
    df  = df.join(enc, on="city")
- **Ida y vuelta con `melt` y `pivot_table`:**
    
    long = df.melt(id_vars=["city"], value_vars=["temp","hum"],
                   var_name="metric", value_name="value")
    wide = (long.pivot_table(index="city", columns="metric", values="value", aggfunc="mean")
                 .reset_index())
- **Merge temporal (asof):**
    
    ticks  = pd.DataFrame({"t": pd.to_datetime(["2025-01-01","2025-01-03"]), "x":[1,2]})
    prices = pd.DataFrame({"t": pd.to_datetime(["2025-01-01","2025-01-02","2025-01-04"]), "p":[10,11,12]})
    asof   = pd.merge_asof(ticks.sort_values("t"), prices.sort_values("t"), on="t", direction="backward")

---
