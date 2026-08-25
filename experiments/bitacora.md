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
