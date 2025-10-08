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
  
### Tema 3.1 — Procesamiento de imágenes (Image processing)

- **Objetivo**: tratar imágenes como **arreglos NumPy** (matrices de píxeles) para filtrado, máscaras, transformaciones geométricas y mejoras.
- **Representación**:
  - Escala de grises: `H x W` (valores 0–255 o 0–1).
  - Color (RGB): `H x W x 3`.
  - Conversión de tipos: `img.astype(np.float32) / 255.0` para operaciones numéricas estables.
- **Operaciones básicas**:
  ```python
  import numpy as np
  # Negativo / contraste lineal
  inv = 1.0 - img
  stretched = np.clip((img - img.min()) / (img.max() - img.min()), 0, 1)
  # Umbral binario
  mask = (img > 0.5).astype(np.float32)
