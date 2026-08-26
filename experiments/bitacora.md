# Bitácora de experimentos

Registro narrativo de las decisiones detrás de cada versión del sistema de
recomendación: qué se probó, por qué, y qué resultado dio. Complementa a
`experiments/log.csv`, que tiene los números crudos (NDCG local vs Kaggle
por corrida). Acá se agrega una sección nueva por versión, sin borrar las
anteriores, para que quede como historial de decisiones.

---

## v0 — Baseline: popularidad

### Objetivo / hipótesis

Antes de meterse con filtrado colaborativo o modelos más sofisticados,
establecer un piso: ¿qué NDCG@20 se consigue recomendando simplemente "lo
más popular" a todo el mundo, sin personalización? Sirve como referencia
para medir si los próximos modelos realmente aportan algo, y para validar
que el pipeline de evaluación (split, métrica, submission) está bien
armado antes de invertir en modelos más complejos.

### Lógica del enfoque

**1. Split leave-one-out por usuario**, no aleatorio global. Si se
mezclaran todas las interacciones y se partiera al azar, muchos usuarios
quedarían sin ninguna fila en train (imposible evaluarlos) o sin ninguna
en val (no aportan a la métrica), y el split dejaría de reflejar el
escenario real de "predecir el próximo libro de cada usuario". Por eso se
agrupa por `id_lector` y se separa un % de las interacciones *de cada
usuario* para validación:

```python
# src/recsys/data.py
for _, grupo in interacciones.groupby("id_lector", sort=False):
    idx = grupo.index.to_numpy().copy()
    rng.shuffle(idx)
    n_val = min(int(len(idx) * frac_val), len(idx) - 1)
    val_idx.extend(idx[:n_val])
    train_idx.extend(idx[n_val:])
```

El `min(..., len(idx) - 1)` es clave: garantiza que ningún usuario quede
con 0 filas en train (si tiene 1 sola interacción, esa va entera a train
y no aporta a val). Confirmado en la corrida: de 1769 usuarios con una
sola interacción, ninguno terminó en val.

**2. `libros_leidos_por_usuario`** arma un `dict {id_lector: set(id_libro)}`
a partir de las interacciones. Se usa para dos cosas distintas según el
contexto: filtrar del ranking los libros ya leídos (i) durante la
evaluación local, calculado sobre *train* únicamente, y (ii) al armar la
submission real, calculado sobre *todo* el dataset — no tendría sentido
recomendarle a alguien un libro que ya leyó.

**3. Score bayesiano de popularidad** (`fit_popularity`), en vez de
ordenar directo por rating promedio:

```python
# src/recsys/models/popularity.py
C = stats["n"].mean()
m = interacciones["rating"].mean()

stats["score"] = (stats["n"] / (stats["n"] + C)) * stats["avg_rating"] + (
    C / (C + stats["n"])
) * m
```

Un promedio simple de rating favorece libros con pocas interacciones que
tuvieron la suerte de que a esos 2 o 3 lectores les encantara — ruido, no
señal. El score bayesiano "encoge" el promedio de cada libro hacia la
media global (`m`) en proporción inversa a cuánta evidencia tiene (`n`
frente a `C`, el volumen promedio de interacciones por libro): un libro
con pocas reseñas pesa más hacia el promedio global, uno con muchas pesa
más hacia su propio promedio. `C` y `m` se calculan solo con los datos que
recibe la función, para poder reusarla igual con train (validación local)
o con el dataset completo (submission real) sin filtrar información de
más.

**4. `ndcg_at_k` / `evaluar_ndcg`**: descuento logarítmico estándar
(`1/log2(pos+2)`, posición 0-indexed) normalizado por el IDCG del ranking
ideal, promediado por usuario. `evaluar_ndcg` arma, para cada usuario de
validación, su ranking recomendado filtrando los libros que ya leyó en
train, y lo compara contra lo que efectivamente leyó en validación.

### Código destacado

- `src/recsys/data.py` — `split_train_val`, `libros_leidos_por_usuario`
- `src/recsys/evaluation.py` — `ndcg_at_k`, `evaluar_ndcg`
- `src/recsys/models/popularity.py` — `fit_popularity`
- `src/recsys/submit.py` — arma el csv final con el mismo ranking global
  para todos los usuarios, filtrando lo ya leído

### Resultado

Split con `frac_val=0.2`, `seed=42`: 373,174 filas en train / 88,234 en
val (sobre 461,408 interacciones totales).

| | NDCG@20 |
|---|---|
| Local (val, train-only) | **0.009960** |
| Kaggle | 0.009540 |
| Diferencia | +0.000420 (+4.4%) |

Fila correspondiente en `experiments/log.csv`. La validación local quedó
muy cerca del score real de Kaggle — buena señal de que el split y la
métrica están bien armados y no hay data leakage evidente, antes de subir
la apuesta con modelos personalizados.

### Próximos pasos / ideas descartadas

- **Se descartó** ordenar por rating promedio simple (sin score
  bayesiano): sube demasiado libros con 1-2 interacciones y rating alto
  por azar, ruido puro con este volumen de datos.
- **Próximo candidato natural:** filtrado colaborativo (ALS vía
  `implicit`, ya está en las dependencias del proyecto) — el score de
  popularidad no personaliza nada, así que cualquier señal de usuario-item
  debería superarlo.
- **Otra idea a explorar:** un modelo content-based usando `genero`,
  `autor` y `editorial` de `libros`, para lectores con pocas interacciones
  donde collaborative filtering va a tener poca señal (cold start).
- **Para más adelante:** combinar señales con un ranker (LightGBM, ya
  está en las dependencias) una vez que haya más de un modelo base para
  generar features.

---

## v1 — Popularidad segmentada (género → franja de nacimiento → global)

### Objetivo / hipótesis

v0 recomienda exactamente lo mismo a todo el mundo — cero personalización.
La hipótesis es simple: si sabemos qué género literario lee un usuario,
recomendarle lo más popular *dentro de ese género* debería ganarle a
recomendarle lo más popular a secas. Antes de armar el modelo se corrió
un EDA completo (ver `experiments/eda.md` y `scripts/eda.py`) para
confirmar que la señal de género tiene cobertura suficiente como para
apoyarse en ella.

### Lógica del enfoque

**Aclaración importante que salió del EDA:** `lectores.genero` (género del
lector: Hombre/Mujer) y `libros.genero` (género literario: Narrativa,
Ensayo, etc.) son campos distintos que comparten nombre de columna. Este
modelo usa exclusivamente el género *literario preferido*, inferido de la
historia de lecturas del usuario — nunca su género como persona.

**1. Género preferido por usuario** (`genero_preferido_por_usuario`): el
género literario más leído en su historial de train. El EDA mostró que el
100% de los usuarios con actividad tiene al menos un libro de género
conocido, así que esta señal cubre a todo el mundo con interacciones.

**2. Normalización de género** (`_normalizar_genero`): el EDA encontró que
`libros.genero` tiene 66 valores únicos pero solo 55 tras normalizar
capitalización/espacios (ej. "Histórica y aventuras" / "HIstórica Y
Aventuras" / "hist**ó**rica y aventuras" son el mismo género). Sin esto,
la preferencia de un usuario se fragmentaría entre variantes del mismo
género y el ranking por género perdería densidad de datos.

**3. Popularidad por género** (`fit_popularity_por_genero`): reusa
`fit_popularity` de v0 tal cual, pero aplicada a cada subgrupo de género
por separado — cada género calcula su propio `C` y `m`, así que un género
con pocas interacciones no queda distorsionado por el promedio de otro
género con mucho volumen.

**4. Franja de nacimiento como segundo fallback**
(`franja_nacimiento_por_usuario`, `fit_popularity_por_franja_nacimiento`):
mismo patrón que género pero agrupando por década de nacimiento en vez de
edad real — `nacimiento` no tiene una fecha de referencia confiable para
calcular edad (ver el EDA). Cubre ~70% de los usuarios (30.4% no tiene
`nacimiento` informado, según el EDA).

**5. Cadena de fallback con backfill** (`recomendar_por_usuario`): para
cada usuario arma su lista de candidatos probando, en orden, género →
franja de nacimiento → popularidad global, saltando libros ya leídos y
sin repetir un libro que ya haya entrado por una fuente anterior. Si el
género no alcanza para completar los k=20 (por ejemplo el usuario ya leyó
casi todo lo popular de ese género), se completa con la fuente siguiente
en vez de devolver una lista corta:

```python
# src/recsys/models/popularity_segmentada.py
for fuente in fuentes:  # [genero, franja, global] — según lo que se conozca
    if len(candidatos) >= k:
        break
    for libro in fuente:
        if len(candidatos) >= k:
            break
        if libro in vistos:
            continue
        candidatos.append(libro)
        vistos.add(libro)
```

**6. Evaluación personalizada** (`evaluar_ndcg_personalizado` en
`evaluation.py`): a diferencia de v0, acá cada usuario tiene su propio
ranking en vez de uno global compartido, así que se agregó una variante
de `evaluar_ndcg` que recibe directamente el dict `{id_lector: ranking}`
ya armado y filtrado.

### Código destacado

- `src/recsys/models/popularity_segmentada.py` — todo el modelo v1
- `src/recsys/evaluation.py` — `evaluar_ndcg_personalizado`

### Resultado

Mismo split que v0 (`frac_val=0.2`, `seed=42`). De los 6,770 usuarios en
val: 100% con género preferido conocido, 71.6% con franja de nacimiento
conocida (consistente con el 69.6% medido en el EDA sobre todos los
lectores).

| | NDCG@20 |
|---|---|
| v0 — popularidad global | 0.009960 |
| v1 — popularidad segmentada | **0.022563** |
| Diferencia | +0.012603 (**+126.5%**) |

Más del doble de NDCG@20 solo con una señal de género bastante simple —
confirma que había mucho margen para personalización, como sugería la
cola larga vista en el EDA. Fila correspondiente en `experiments/log.csv`.

`submit.py` se extendió para soportar modelos con ranking por usuario
(antes asumía un único ranking global): `MODELOS` ahora mapea cada
nombre a una función `(usuarios, k) -> {id_lector: [id_libro, ...]}` en
vez de a un `fit_*` que devuelve un ranking único.

Subido a Kaggle:

| | NDCG@20 |
|---|---|
| v0 — popularidad global (Kaggle) | 0.00954 (logueado) |
| v1 — popularidad segmentada (local) | 0.022563 |
| v1 — popularidad segmentada (Kaggle) | **0.01558** |

La mejora se sostiene en el leaderboard (v1 > v0), aunque con un
descuento más marcado que en v0: local vs Kaggle en v0 difería apenas
+4.4%, acá v1 local sobreestima bastante más (+44.8% relativo). Puede ser
señal de que el fallback por género/franja se ajusta más al detalle de
la muestra de validación local que lo que generaliza al set de Kaggle —
a seguir de cerca en las próximas versiones.

**Inconsistencia a revisar:** Kaggle reportó este resultado como mejora
sobre un "previous best" de 0.01024, que no coincide con el 0.00954
logueado para v0 en `experiments/log.csv`. Puede deberse a
public/private leaderboard, algún submit no logueado, o un typo al
transcribir el score de v0 — pendiente de confirmar cuál es el numero
correcto antes de tomarlo como referencia para las próximas comparaciones.

### Próximos pasos / ideas descartadas

- **Pendiente inmediato:** aclarar la inconsistencia del "previous best"
  de Kaggle (0.01024 vs 0.00954) antes de seguir comparando contra él.
- **Se descartó** calcular edad real a partir de `nacimiento`: no hay
  fecha de referencia confiable en los datos, así que se usó década de
  nacimiento como proxy (ver EDA).
- **Próximo candidato:** en vez de un solo género "top", ponderar por los
  2-3 géneros más leídos del usuario (mezclando sus rankings) en lugar de
  quedarse solo con el primero — podría ayudar a los usuarios con
  intereses mixtos.
- **Otra idea:** franja de nacimiento + género combinados (popularidad
  dentro de la intersección género × década) para usuarios con suficiente
  historial, en vez de tratarlos como fallback secuencial excluyente.
