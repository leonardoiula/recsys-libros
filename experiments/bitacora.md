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
| Local (val, train-only, con filtro de ya leído) | **0.009960** |
| Kaggle (código actual: score bayesiano + filtro) | 0.010240 |
| Diferencia | +0.000280 (+2.7%) |

Fila `popularity` en `experiments/log.csv`. La validación local quedó muy
cerca del score real de Kaggle — buena señal de que el split y la métrica
están bien armados y no hay data leakage evidente, antes de subir la
apuesta con modelos personalizados.

**Nota sobre una comparación anterior incorrecta:** en un primer momento
comparamos el local (0.009960, que ya filtra libros ya leídos porque
`evaluar_ndcg` lo hace) contra un score de Kaggle de 0.00954 — pero ese
0.00954 correspondía a una submission *sin* ese filtro, generada antes de
este repo (no hay código equivalente acá). Filtrar los libros ya leídos
demostró tener valor propio en el leaderboard, independiente de la
segmentación por género: **+7.3%** solo por agregar ese filtro
(0.00954 → 0.01024, fila `popularity_sin_filtro` en `log.csv` como
referencia histórica). La comparación correcta para v0 es contra 0.01024,
que es lo que efectivamente genera el código actual.

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
| v0 — popularidad, sin filtro de ya leído (histórico, sin código) | 0.00954 |
| v0 — popularidad + filtro de ya leído (código actual) | 0.01024 |
| v1 — popularidad segmentada (local) | 0.022563 |
| v1 — popularidad segmentada (Kaggle) | **0.01558** |

La comparación correcta es v1 vs v0-con-filtro (0.01024), que es lo que
realmente genera el código base sobre el que se construyó v1: **+52.1%**
en Kaggle (0.01024 → 0.01558) — coincide con el "previous best" que
reportó Kaggle al confirmar la mejora. La segmentación por género/franja
aporta una mejora bien real, adicional a la que ya daba el filtro de ya
leído por sí solo.

Localmente la mejora de v1 sobre v0 fue de +126.5% (0.009960 →
0.022563) — bastante más optimista que el +52.1% real en Kaggle. La
brecha local-vs-Kaggle también creció respecto a v0 (que difería solo
+2.7%): acá el local sobreestima en +44.8%. Es esperable que agregar
segmentación abra algo de brecha (el fallback por género/franja se ajusta
más a los usuarios y libros específicos de la muestra de validación local
que a la población completa de Kaggle), pero vale la pena vigilar esa
brecha en las próximas versiones para que no sea señal de overfitting al
split local.

### Próximos pasos / ideas descartadas

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

---

## v2 — Filtrado colaborativo (ALS)

### Objetivo / hipótesis

v0 y v1 son heurísticas de popularidad (global o segmentada por
género/franja) — ninguna de las dos usa una señal real usuario-item.
Ya lo señalaba la sección de "próximos pasos" de v0: filtrado
colaborativo vía ALS (`implicit`, ya en las dependencias) debería
superarlas, porque puede capturar afinidades específicas
usuario-libro que ninguna heurística de segmentación captura.

### Lógica del enfoque

**Rating explícito como confianza implícita.** `implicit.als.AlternatingLeastSquares`
está pensado para feedback implícito (una matriz de "confianza", no de
rating). Acá el dato real es un rating explícito de 1 a 10, siempre
positivo (EDA). En vez de binarizar "leyó o no leyó" y tirar la señal de
intensidad, se arma la matriz sparse usuario×libro usando el rating
directamente como peso/confianza de esa celda — patrón estándar para
aplicar ALS implícito sobre datos que en el fondo son explícitos pero
muy sparse (99.91% sparsity, EDA).

**`construir_matriz_usuario_libro`** arma la matriz y dos mapeos de
índice (`fila_por_usuario`, `libros_por_columna`) para poder traducir
filas/columnas del modelo de vuelta a `id_lector`/`id_libro` reales.

**`fit_als`** entrena `AlternatingLeastSquares` sobre esa matriz.
Hiperparámetros elegidos con un sweep local chico (no committeado, mismo
patrón que v0/v1) sobre el split `frac_val=0.2`/`seed=42`:

| factors | regularization | iterations | NDCG@20 local |
|---|---|---|---|
| 32  | 0.05 | 15 | 0.226281 |
| 64  | 0.10 | 20 | 0.245332 |
| 64  | 0.01 | 20 | 0.245427 |
| 128 | 0.10 | 20 | **0.260068** |
| 128 | 0.01 | 20 | 0.259924 |
| 192 | 0.10 | 20 | 0.264488 |
| 128 | 0.10 | 30 | 0.260638 |

Rendimientos decrecientes claros después de `factors=128`: subir a 192
factors o 30 iteraciones apenas mueve la aguja a cambio de más tiempo de
entrenamiento. Se eligió `factors=128, regularization=0.1, iterations=20`
como default de `fit_als` — buen balance entre NDCG y costo.

**`recomendar_por_usuario`** arma el top-k por usuario con
`modelo.recommend(..., filter_already_liked_items=True)`, con dos
salvedades encontradas al validar el código contra la librería instalada
(`implicit==0.7.3`) directamente, sin pasar por el pipeline completo:

1. Cuando no alcanzan k libros no vistos para un usuario (no pasa con
   este dataset — cada usuario activo tiene miles de libros sin leer),
   `implicit.recommend` **no** rellena los huecos con índice `-1` como se
   asumió en un primer borrador, sino repitiendo ids de libros ya vistos
   con un score inválido (sentinela). Por eso la protección real contra
   ese caso es el filtro explícito contra `libros_leidos` (que de todos
   modos ya se hacía por consistencia con el resto del proyecto), no un
   chequeo de índice — se sacó ese chequeo del código por ser un supuesto
   incorrecto sobre la librería.
2. Usuarios sin ninguna fila en la matriz (sin datos de entrenamiento)
   caen enteramente a `ranking_global` (popularidad, ya filtrada) — no
   pasó ningún caso así en el split local (todo usuario con actividad en
   train tiene fila), pero cubre el caso general.

### Código destacado

- `src/recsys/models/als.py` — todo el modelo v2
- `src/recsys/submit.py` — `_recomendaciones_als`, registrado como
  `"als"` en `MODELOS`
- `tests/test_als.py` — construcción de la matriz + lógica de
  `recomendar_por_usuario` contra un modelo ALS *stub* (no se testea con
  un `AlternatingLeastSquares` real: lo que hay que verificar es la
  lógica propia del proyecto, no el comportamiento interno de `implicit`)

### Resultado

Mismo split que v0/v1 (`frac_val=0.2`, `seed=42`): 373,174 filas en
train / 88,234 en val, 6,770 usuarios en val (los 6,770 tuvieron fila en
la matriz de ALS — ninguno cayó al fallback global).

| | NDCG@20 |
|---|---|
| v0 — popularidad global | 0.009960 |
| v1 — popularidad segmentada | 0.022563 |
| v2 — ALS | **0.260068** |

Salto grande — más de 10x sobre v1. Se investigó específicamente para
descartar un bug de leakage antes de loguearlo (matriz fit solo con
train, `libros_leidos` calculado solo con train, filas de val mapeadas
por `id_lector` sin depender de orden). El salto se explica principalmente
porque el split de este proyecto separa un **20% de las interacciones de
cada usuario** para validación (no literalmente "una" interacción, pese a
como se lo suele llamar) — para un lector con muchas interacciones eso
son decenas de libros relevantes en val, y ahí un modelo con señal
usuario-item real (ALS) tiene muchísimo más para acertar que una
heurística de segmentación. Desagregando por actividad en train:

| interacciones en train | usuarios en val | NDCG@20 medio |
|---|---|---|
| 3–5 | 799 | 0.1448 |
| 6–10 | 1,261 | 0.1908 |
| 11–20 | 1,275 | 0.2237 |
| 21+ | 3,435 | 0.2968 |

Los usuarios con más historial (que también tienen los sets de
validación más grandes) empujan el promedio hacia arriba, pero incluso
los usuarios con poco historial (3-5 interacciones en train) ya superan
holgadamente el 0.0226 de v1.

Subido a Kaggle:

| | NDCG@20 |
|---|---|
| v0 — popularidad + filtro de ya leído (Kaggle) | 0.01024 |
| v1 — popularidad segmentada (Kaggle) | 0.01558 |
| v2 — ALS (Kaggle) | **0.03864** |

ALS sigue siendo una mejora real en Kaggle: **+148.0%** sobre v1
(0.01558 → 0.03864) y **+277.3%** sobre v0. La señal usuario-item real sí
aporta, incluso descontando la sobreestimación del local.

Pero la sobreestimación del local fue mucho más grande de lo que se
esperaba al escribir la sección anterior de esta bitácora:

| | NDCG@20 |
|---|---|
| v2 — ALS (local) | 0.260068 |
| v2 — ALS (Kaggle) | 0.03864 |
| Diferencia | -0.221428 (local **6.7x** el valor de Kaggle, **+573%**) |

La brecha local-vs-Kaggle venía creciendo con cada versión más
personalizada (v0: +2.7%, v1: +44.8%), como era esperable, pero el salto
a +573% en v2 es demasiado grande para explicarlo solo por "más
personalización = más brecha". La hipótesis más probable, coherente con
el análisis de arriba (el NDCG local sube fuerte con la actividad del
usuario en train): el split local separa **20% de las interacciones de
cada usuario** para validación, así que un usuario con mucho historial
termina con decenas de libros "relevantes" simultáneos en val — una
tarea mucho más fácil de acertar para un modelo con señal usuario-item
real que la tarea que efectivamente evalúa Kaggle (que probablemente se
parece más a predecir un puñado de próximas lecturas por usuario, no una
fracción proporcional a todo su historial). v0/v1 no dependían tanto de
señal usuario-item específica, así que esa diferencia de tarea los
afectaba mucho menos.

**Esto pone en duda la metodología de validación local para modelos
personalizados** (no el modelo en sí, que sí mejora en Kaggle) — ver
próximos pasos.

### Próximos pasos / ideas descartadas

- **Se descartó** binarizar el rating (leyó/no leyó) para usar el modelo
  implícito "puro": se prefirió usar el rating como confianza para no
  tirar la señal de intensidad, dado que el EDA mostró que el rating
  tiene variación real (no todo 10).
- **Prioridad alta para la próxima versión:** revisar el split de
  validación local. `frac_val=0.2` por usuario infla artificialmente el
  NDCG de usuarios con mucho historial (más "relevantes" simultáneos en
  val de lo que un ranking de k=20 puede reflejar realistamente) y por
  eso dejó de ser un buen proxy de Kaggle apenas el modelo empezó a usar
  señal usuario-item real. Vale la pena probar un split más parecido a
  "leave-one-out" literal (un solo libro por usuario en val, no un %) o
  un tope fijo de libros en val por usuario, para que la tarea local se
  parezca más a la de Kaggle antes de confiar en comparaciones de NDCG
  local entre versiones.
- **Próximo candidato:** combinar ALS con las señales de v1 (género,
  franja) en un ranker (LightGBM, ya está en las dependencias) en vez de
  quedarse con un solo modelo base — usar el score de ALS y el de
  popularidad segmentada como features.

---

## Corrección de metodología: split temporal + `n_val` fijo, `C` configurable

### Objetivo / hipótesis

Surgió de una ronda de feedback sobre la documentación (ver
`experiments/decisiones.md`), con dos hallazgos concretos:

1. `split_train_val` ignoraba `fecha` y retenía a val una fracción
   (`frac_val=0.2`) de las interacciones de cada usuario elegida al azar.
   Se probó: sobre el mismo modelo (ALS), pasar de split aleatorio a
   split *temporal* (últimas interacciones a val) bajó el NDCG@20 local
   de 0.260068 a 0.122789 sin cambiar nada más — confirma leakage
   temporal real (el modelo veía información del futuro del usuario).
   Y el `frac_val` proporcional dejaba a los usuarios con mucho historial
   con muchos más libros "relevantes" simultáneos en val que a los
   livianos, inflando su NDCG de forma dispareja.
2. `fit_popularity` usa `C = stats["n"].mean()` sin haberlo validado
   nunca contra NDCG, pese a que la distribución de interacciones por
   libro es fuertemente right-skewed (mediana 2, media ~9.6).

### Lógica del enfoque

**1. Split temporal con `n_val` fijo** (`data.py::split_train_val`,
firma nueva: `n_val=1` en vez de `frac_val=0.2`). Por usuario, se
ordenan sus interacciones por `fecha` (parseada con
`pd.to_datetime(..., format="%d-%m-%Y", errors="coerce")`) y se retienen
a val las últimas `n_val` — no una fracción proporcional a su actividad.
Con `n_val=1` es un leave-one-out literal: "predecir la próxima
lectura", que es mucho más parecido al escenario real que "adivinar un
20% aleatorio de todo lo que leyó". Fechas no parseables (~0.0002%) se
tratan como las más antiguas del usuario, nunca se mandan a val "a
ciegas". Se mantiene el invariante de nunca vaciar el train de un
usuario (`min(n_val, len(idx) - 1)`).

**2. `C` configurable en `fit_popularity`**, pero el sweep dio un
resultado que **contradice la sospecha inicial**. Se probaron tres
candidatos sobre v0 con el split corregido:

| C | valor | NDCG@20 (v0) |
|---|---|---|
| media (actual) | 9.52 | **0.006620** |
| media geométrica | 2.99 | 0.000675 |
| mediana | 2.00 | 0.000221 |

La media gana por lejos, casi 10x contra las alternativas "menos
agresivas". La explicación: con leave-one-out estricto (1 solo libro
relevante por usuario en validación), lo que más importa es que el
top-20 esté dominado por libros de atractivo *ampliamente* comprobado.
Un `C` chico deja subir demasiado a libros de nicho con 2-3 interacciones
y un rating alto por azar, diluyendo el top-20 con "flukes" en vez de
apuestas seguras — el mismo problema, en el fondo, que motivó usar un
score bayesiano en primer lugar en vez de un promedio simple (v0). **No
se cambia el default**, pero ahora es explícito y configurable, y quedó
validado con datos en vez de asumido.

**3. Re-sweep de hiperparámetros de ALS** bajo el split corregido:
`factors=128, regularization=0.1, iterations=20` (los mismos de antes)
siguen siendo un buen punto — probar 192 factors o más iteraciones dio
mejoras marginales (<5%) a cambio de más tiempo de entrenamiento. Sin
cambios.

**4. Se probó (y se descartó) un ruteo híbrido para v2**: la hipótesis
era que el embedding de ALS es poco confiable para usuarios con poca
actividad (mal condicionado, alta varianza), y que convendría, para esos
casos, caer a la cadena de popularidad por género de v1 en vez de
confiar en ALS. Se implementó `als.py::recomendar_hibrido` (rutea por un
`umbral` de interacciones en train) y se midió el NDCG@20 por bucket de
actividad, género-solo vs ALS-solo, con el split corregido:

| interacciones en train | usuarios | NDCG género | NDCG ALS |
|---|---|---|---|
| 1 | 972 | 0.0162 | **0.0863** |
| 2 | 710 | 0.0208 | **0.1179** |
| 3–4 | 894 | 0.0201 | **0.1389** |
| 5–9 | 1,276 | 0.0172 | **0.1291** |
| 10–14 | 747 | 0.0121 | **0.1458** |
| 15–19 | 500 | 0.0143 | **0.1483** |
| 20–29 | 657 | 0.0160 | **0.1092** |
| 30–49 | 858 | 0.0064 | **0.1015** |
| 50–99 | 1,060 | 0.0106 | **0.0718** |
| 100+ | 1,230 | 0.0068 | **0.0236** |

**ALS le gana a género en absolutamente todos los buckets**, incluso con
una sola interacción de historial. La hipótesis original no se sostiene
bajo una evaluación sin leakage: el factorizado colaborativo aprovecha
la estructura de *todos* los usuarios del sistema incluso para alguien
con actividad mínima, y le gana por lejos a un heurístico de 55
categorías de género. `submit.py` sigue usando ALS puro. La función
`recomendar_hibrido` queda implementada y testeada (`tests/test_als.py`)
por si el escenario cambia en el futuro, pero **no está en producción**:
usarla con cualquier `umbral > 0` empeora el resultado con estos datos.

### Código destacado

- `src/recsys/data.py::split_train_val` — reescrita, firma nueva
- `src/recsys/models/popularity.py::fit_popularity` — parámetro `C`
- `src/recsys/models/als.py::recomendar_hibrido` — implementada, no usada
- `tests/test_data.py`, `tests/test_popularity.py` (nuevos), extensión de
  `tests/test_als.py`

### Resultado

Split corregido (`n_val=1`, `seed=42`): 452,504 filas en train / 8,904 en
val (un usuario por fila, ya que `n_val=1`).

| | NDCG@20 local (split viejo) | NDCG@20 local (split corregido) | Kaggle real |
|---|---|---|---|
| v0 — popularidad | 0.009960 | 0.006620 | 0.01024 |
| v1 — popularidad segmentada | 0.022563 | 0.013719 | 0.01558 |
| v2 — ALS | 0.260068 | **0.101473** | 0.03864 |

El split corregido bajó el NDCG local de las tres versiones (esperable:
ya no hay leakage temporal ni inflación proporcional), y en particular
achicó muchísimo la sobreestimación de v2 sobre Kaggle: de **+573%**
(split viejo, 0.260068 vs 0.03864) a **+162.6%** (split corregido,
0.101473 vs 0.03864). Mejora real, pero todavía sobreestima bastante.

**Pista extra sobre lo que queda de brecha:** los usuarios que Kaggle
efectivamente califica (`ejemplo.csv`) tienen una actividad muchísimo
mayor que la población general (mediana 74 interacciones en train contra
9 en la población general activa — ver `experiments/decisiones.md`). Si
se reordena el NDCG local ponderando por la distribución de actividad
real de `ejemplo.csv` en vez de promediar parejo sobre todos los
usuarios de validación, el estimado baja de 0.101473 a **0.064008**
(+65.6% sobre Kaggle en vez de +162.6%). No es una corrección que se
aplicó al código (es un análisis post-hoc, no una feature del split),
pero confirma que buena parte de la brecha restante es composición de
población, no un problema nuevo del modelo. Vale la pena, en una futura
iteración, considerar si local debería muestrear/ponderar la validación
para parecerse más a la población real de `ejemplo.csv`.

### Próximos pasos / ideas descartadas

- **Se descartó** el ruteo híbrido género/ALS por umbral de actividad
  (ver arriba) — los datos no lo respaldan con este dataset.
- **Se descartó** usar `C=mediana` o `C=media geométrica` en
  `fit_popularity` — empeoran el NDCG real, pese a la intuición inicial
  de que la media actual "sobre-shrinkeaba" el catálogo.
- **Pendiente:** el análisis de reponderación por actividad sugiere que
  local podría acercarse más a Kaggle si se muestrea/pondera la
  validación para reflejar la composición real de usuarios que califica
  la competencia, en vez de la población general activa. No implementado
  todavía.
- **Pendiente de confirmar en Kaggle:** este paquete no generó una nueva
  submission subida — los números de Kaggle en la tabla de arriba siguen
  siendo los de la corrida anterior de v0/v1/v2 (el modelo v2 no cambió,
  solo la metodología de evaluación local).

---

## Mejoras a ALS: confianza ponderada, búsqueda sistemática, BPR

### Objetivo / hipótesis

Con el split corregido, el NDCG@20 local de ALS (0.101473) seguía bajo
en términos absolutos. Se discutieron tres palancas para mejorar el
*modelo en sí* (no la evaluación): (1) la fórmula de confianza de ALS
usaba el rating crudo como peso, en vez de la fórmula estándar de
feedback implícito ponderado; (2) el sweep de hiperparámetros venía
siendo manual (grid chico, un hiperparámetro por vez); (3) nunca se
probó una alternativa a ALS como BPR, que optimiza directamente el orden
relativo entre ítems en vez de reconstruir la matriz de confianza.

También surgió la pregunta de si medir top-200 en vez de top-20 tenía
sentido: la entrega a Kaggle es exactamente k=20 por usuario (formato
fijo de `ejemplo.csv`, no negociable), pero un top-200 más amplio sirve
como diagnóstico -- separa un problema de *cobertura* (el libro correcto
ni entra entre los candidatos) de uno de *ranking* (entra, pero mal
ordenado dentro del top-20 real).

### Lógica del enfoque

**1. Confianza ponderada** (`als.py::construir_matriz_usuario_libro`,
parámetro `alpha`): `confianza = 1 + alpha * rating` (Hu/Koren/Volinsky),
en vez de usar el rating crudo como peso. `alpha` controla cuánta más
confianza da cada punto de rating, de forma independiente a la escala
del rating -- con el rating crudo, esa relación estaba fija en 1:1 sin
posibilidad de ajustarla.

**2. `recall_at_k` / `evaluar_recall_personalizado`**
(`evaluation.py`): mismo patrón que `ndcg_at_k`/`evaluar_ndcg_personalizado`,
pero mide si algún relevante aparece en el top-k, sin importar el orden.
Con `n_val=1` (un solo relevante por usuario) equivale a "¿el libro
correcto apareció en el top-k, sí o no?". Se usa con `k=200` como
diagnóstico, nunca para la entrega real.

**3. `bpr.py`** (nuevo módulo): `BayesianPersonalizedRanking` de
`implicit`, sobre una matriz *binaria* (interactuó/no, no ponderada por
rating -- BPR es un ranking pairwise, no una reconstrucción). Se
confirmó por inspección directa de la librería que expone la misma API
`.recommend(...)` que `AlternatingLeastSquares`, así que
`als.py::recomendar_por_usuario` se reusa tal cual, sin duplicar esa
lógica.

**4. `scripts/tune_als.py`** (nuevo, committeado): dos estudios de
`optuna` (30 trials cada uno, TPE sampler) maximizando NDCG@20
personalizado sobre el split corregido:
- ALS: `factors` (32-256, log), `regularization` (0.001-1.0, log),
  `alpha` (0.01-10, log). `iterations=20` fijo (rendimiento decreciente
  ya confirmado en el sweep manual anterior).
- BPR: `factors` (32-256, log), `regularization` (1e-5-0.1, log),
  `learning_rate` (0.001-0.1, log), `iterations` (50-300) -- SGD
  converge distinto a ALS, no se fijó de antemano.

`optuna` se agregó como dependencia de desarrollo (`uv add --dev optuna`,
no hace falta para correr `submit.py`).

### Código destacado

- `src/recsys/models/als.py` — parámetro `alpha`, nuevos defaults
- `src/recsys/models/bpr.py` — módulo nuevo
- `src/recsys/evaluation.py` — `recall_at_k`, `evaluar_recall_personalizado`
- `scripts/tune_als.py` — búsqueda con optuna

### Resultado

Mismo split que la sección anterior (`n_val=1`, `seed=42`).

| candidato | NDCG@20 | Recall@200 | hiperparámetros |
|---|---|---|---|
| ALS actual (sin tunear) | 0.100883 | 0.3976 | factors=128, regularization=0.1, alpha=1.0 |
| **ALS tuneado (optuna)** | **0.112530** | 0.3817 | factors=256, regularization=0.128, alpha=4.718 |
| BPR tuneado (optuna) | 0.084991 | 0.3182 | factors=196, regularization=0.0061, learning_rate=0.034, iterations=235 |

**ALS le gana a BPR** en este dataset, con margen claro tanto en NDCG@20
como en Recall@200 -- la hipótesis de que BPR se ajustaría mejor a una
métrica de ranking no se confirmó acá. **Tunear `alpha` junto con
`factors`/`regularization` sí valió la pena**: +11.5% de NDCG@20 sobre
el config anterior, con `alpha≈4.7` (mucha más separación entre ratings
altos y bajos que el `alpha=1.0` por defecto de la fórmula) y más
`factors` (256 vs 128). Hay un trade-off real: Recall@200 bajó de 0.398
a 0.382 -- el modelo quedó más preciso en el top-20 pero un poco menos
amplio en cobertura general.

**Nota técnica:** tanto `factors` (256, el techo del rango buscado) como
`alpha` (varios de los mejores trials entre 3-9) terminaron cerca o en
el borde superior del espacio de búsqueda -- señal de que una segunda
pasada con rangos más amplios (`factors` hasta ~512, `alpha` hasta
~20-30) podría encontrar algo todavía mejor. No se hizo en este paquete.

Se decidió con el usuario **wirear el ALS tuneado a `submit.py`**
(nuevos defaults de `fit_als`) en vez de dejarlo solo como diagnóstico o
ampliar la búsqueda primero.

### Próximos pasos / ideas descartadas

- **Se descartó** BPR como modelo de producción -- perdió contra ALS en
  ambas métricas con este dataset.
- **Pendiente:** una segunda búsqueda de optuna con rangos más amplios
  de `factors`/`alpha`, dado que los ganadores actuales quedaron cerca
  del borde superior del espacio explorado.
- **Pendiente de confirmar en Kaggle:** se regeneró
  `outputs/submissions/als.csv` con los hiperparámetros nuevos, pero no
  se subió a Kaggle en este paquete.
- **Sigue en pie** la idea de dos etapas (ALS + género + popularidad
  como features de un ranker LightGBM) -- no se tocó en este paquete.

---

## Regresión en Kaggle: el ALS tuneado con optuna empeoró el score real

### Lo que pasó

Se subió el `als.csv` generado con los hiperparámetros elegidos por
optuna (`factors=256, regularization=0.128, alpha=4.718`). Resultado:

| | NDCG@20 local | NDCG@20 Kaggle |
|---|---|---|
| ALS anterior (rating crudo, factors=128, regularization=0.1) | 0.100883 | 0.03864 |
| ALS tuneado (confianza `1+alpha*rating`, factors=256, regularization=0.128, alpha=4.718) | 0.112530 (**+11.5%**) | **0.03341 (-13.5%)** |

**El modelo que mejoró el NDCG local terminó peor en Kaggle.** No es un
matiz -- es la señal más fuerte hasta ahora de que el NDCG local, tal
como se calcula hoy (aun con el split corregido), no es un proxy
confiable para decisiones de *optimización fina* de hiperparámetros,
aunque sí lo fue para decisiones más gruesas (ALS >> popularidad
segmentada >> popularidad, confirmado en las tres versiones vía Kaggle
real).

### Hipótesis de por qué pasó

1. **Sobreajuste al split de validación local por exceso de búsqueda.**
   30 trials de optuna, todos evaluados contra el mismo split fijo
   (`n_val=1`, `seed=42`, 8,904 usuarios). Cuantos más configs se prueban
   contra el mismo conjunto de validación, más probable es terminar
   eligiendo el que mejor se ajusta al *ruido específico* de ese split
   en particular, no una mejora real y generalizable -- el clásico
   problema de "hyperparameter hacking" cuando se hacen muchas
   iteraciones contra un único held-out set.
2. **El propio trade-off que ya se había detectado y no se tomó en
   serio:** el config tuneado bajó el Recall@200 de 0.398 a 0.382 al
   mismo tiempo que subía el NDCG@20. Eso ya era una señal de que el
   modelo se estaba volviendo *más afilado* en su propio top-20 (más
   basado en diferenciar fuerte entre ratings altos y bajos, con
   `alpha≈4.7`) a costa de cobertura general -- si la tarea real de
   Kaggle premia más la cobertura de lo que el NDCG@20 local logra
   capturar, ese trade-off explica directamente la caída.
3. **La fórmula de confianza en sí (`1+alpha*rating`) puede no
   transferir igual que el rating crudo**, independientemente del
   tuneo de `factors`/`regularization` -- no se aisló este efecto (no
   se subió a Kaggle una versión intermedia con la fórmula nueva pero
   los hiperparámetros viejos), así que no se puede saber cuánto de la
   caída es la fórmula y cuánto es el sobreajuste del sweep.

### Decisión

Se le presentó el hallazgo al usuario con tres opciones (revertir al
config anterior, probar una config intermedia para aislar si la causa es
la fórmula de confianza o los hiperparámetros, o dejarlo como está). Se
decidió **dejar el config tuneado (optuna) en `submit.py` por ahora**,
pese a tener peor score real en Kaggle que el anterior, para seguir
explorando otras líneas (ej. el ranker de dos etapas) antes de fijar el
default final de ALS. Queda explícito acá que el estado actual de
producción **no es el mejor confirmado en Kaggle** -- si se sube una
nueva submission antes de resolver esto, tenerlo presente.

### Próximos pasos

- **Elevar la prioridad** del ítem ya anotado sobre corregir la
  metodología de evaluación local -- esta regresión es evidencia directa
  (no solo teórica) de que optimizar agresivamente contra el NDCG local
  actual puede empeorar el resultado real.
- **Para el futuro:** si se vuelve a usar optuna, considerar un esquema
  de validación menos propenso a sobreajuste (ej. cross-validation con
  varios splits/seeds en vez de uno solo, o un conjunto de validación
  separado del usado para elegir el mejor trial) antes de confiar en un
  solo número de un solo split para elegir hiperparámetros finales.
- **Presupuesto de submissions:** Kaggle probablemente limita cuántas
  entregas por día se pueden subir -- vale la pena ser deliberado sobre
  qué configs se confirman en Kaggle en vez de subir cada iteración
  local, dado lo que se acaba de confirmar sobre la confiabilidad del
  proxy local.

---

## v3 — Ranker de dos etapas (ALS + género + popularidad → LightGBM)

### Objetivo / hipótesis

Es la idea "próximo candidato" anotada desde la sección v2: en vez de un
solo modelo base, generar candidatos con varias señales (ALS,
popularidad por género, popularidad global) y dejar que un modelo
supervisado (LightGBM) aprenda a combinarlas y reordenarlas mejor de lo
que cualquiera hace sola.

**Restricción explícita del usuario para este diseño:** no repetir el
error de optimizar contra un solo split local. Por eso este modelo se
evalúa desde el arranque con **validación cruzada sobre 3 seeds**, no
un split fijo -- es el "próximo paso" que había quedado anotado tras el
episodio de la regresión de ALS en Kaggle.

### Lógica del enfoque

**Split de tres niveles** para evitar leakage de un modelo apilado (un
ranker que usa el score de otro modelo como *feature* necesita que ese
score salga de datos que el ranker no vio como etiqueta):

```python
train_candidatos, test_final = split_train_val(interacciones, n_val=1, seed=S)
train_candidatos, train_ranker = split_train_val(train_candidatos, n_val=1, seed=S+1000)
```

- `train_candidatos`: fit de ALS/popularidad/género (las señales).
- `train_ranker`: etiquetas conocidas (el próximo libro real de cada
  usuario) para entrenar el `LGBMRanker`, con candidatos/features
  generados por los modelos fit solo en `train_candidatos`.
- `test_final`: hold-out final, aislado de todo lo anterior.

Se repite para 3 seeds y se reporta media ± desvío (`evaluation.py::evaluar_multisplit`,
nueva utilidad genérica pensada para reusarse con cualquier modelo futuro).

**`ranker.py::generar_candidatos_con_features`**: por usuario, une los
candidatos de las tres fuentes (ALS vía `modelo.recommend(N=150)` con
sus scores reales, popularidad global top-150, popularidad por género
top-150 si se conoce el género preferido), excluyendo ya leídos. Cada
candidato queda con `score_*`/`rank_*` por fuente que lo propuso (rank =
posición real dentro de esa fuente, no posición entre los candidatos
finalmente elegidos) y `en_*` (1/0, qué fuentes lo propusieron); un
candidato que no vino de una fuente queda con score 0.0 y rank 150
(sentinel "justo afuera de la ventana"). Reusa `fit_popularity`,
`fit_popularity_por_genero` y `genero_preferido_por_usuario` tal cual --
no duplica esa lógica.

**`ranker.py::armar_dataset_entrenamiento`**: arma `(X, y, group)` para
`lightgbm.LGBMRanker.fit`. Si el libro-etiqueta de un usuario no está
entre sus candidatos generados, se lo inyecta igual con features
"ausente" -- si no, ese usuario no aporta ningún positivo al
entrenamiento (práctica estándar en learning-to-rank).

**`ranker.py::fit_ranker`**: `LGBMRanker(objective="lambdarank", ...)`
con hiperparámetros **conservadores** (`num_leaves=31, learning_rate=0.05,
n_estimators=200`), sin sweep agresivo tipo optuna en esta primera
versión -- para no repetir el mismo error con un modelo nuevo.

### Código destacado

- `src/recsys/models/ranker.py` — todo el modelo v3
- `src/recsys/evaluation.py::evaluar_multisplit` — validación cruzada genérica
- `scripts/evaluate_ranker.py` — evaluación de 3 seeds, ALS solo vs ranker
- `src/recsys/submit.py::_recomendaciones_ranker` — wiring a producción

### Resultado

Validación cruzada sobre 3 seeds (42, 7, 123), NDCG@20 en `test_final`:

| seed | ALS solo | Ranker (dos etapas) |
|---|---|---|
| 42 | 0.098168 | 0.102442 |
| 7 | 0.103139 | 0.105693 |
| 123 | 0.100474 | 0.105327 |
| **media ± desvío** | **0.100594 ± 0.002488** | **0.104487 ± 0.001780** |

**El ranker le ganó a ALS solo en los 3 seeds**, no solo en promedio
(+3.9%) -- y con menor desvío entre seeds (0.00178 vs 0.00249), más
estable. Es una señal bastante más confiable que la mejora frágil que
tuvo el sweep de ALS con optuna (que perdió justo por no sostenerse
fuera del split en el que se midió).

**Caveat:** esta comparación usa el ALS actual (`factors=256, alpha=4.718`)
como una de las señales del ranker -- el mismo config que ya sabemos que
no generalizó igual en Kaggle que en local. El ranker suma valor por
*encima* de ese ALS, pero hereda parte de esa incertidumbre sobre la
señal de base.

**Simplificación de producción**: para la submission real no hay un
"futuro" que reservar como `test_final` -- se entrena un solo split de
dos niveles (`train_candidatos`/`train_ranker`, `seed=42`) y los modelos
de etapa 1 se reusan tal cual (sin refittear con todos los datos) tanto
para entrenar el ranker como para generar los candidatos finales, para
evitar el desajuste de aplicar el ranker sobre scores de un modelo
distinto al que vio durante su entrenamiento. El costo: los candidatos
finales no aprovechan la interacción más reciente de cada usuario como
señal (sí se usa para filtrar libros ya leídos).

Wireado a `submit.py` como modelo **nuevo** `"ranker"` -- no reemplaza
`"als"`, conviven (mismo criterio que `popularity`/`popularity_segmentada`/`als`
hoy). Submission generada
(`outputs/submissions/ranker_20260829-150300_ranker-dos-etapas.csv`),
pendiente de confirmar en Kaggle.

### Próximos pasos / ideas descartadas

- **Pendiente de confirmar en Kaggle** -- es la única forma real de
  saber si la mejora de +3.9% (consistente en 3 seeds locales) se
  sostiene, dado el precedente de la regresión de ALS.
- **Pendiente:** resolver primero la incertidumbre del ALS de base
  (revisar si conviene volver a `factors=128/alpha=1.0` para la señal
  del ranker también) antes de invertir en tunear el ranker en sí.
- **No se hizo (a propósito) un sweep de hiperparámetros del
  `LGBMRanker`** en esta primera versión -- si se hace en el futuro,
  usar la misma validación cruzada multi-seed, no un solo split.
- **Nota técnica:** `scripts/evaluate_ranker.py` imprime resultados con
  el símbolo "±", que rompe el filtro `grep -v` usado para limpiar el
  log de progreso de `implicit`/LightGBM (lo trata como binario y
  descarta la salida) -- si se re-corre el script, mejor sin ese grep,
  o revisar el archivo de output completo en vez de la salida filtrada.

### Resultado real en Kaggle

| Modelo | Kaggle |
|---|---|
| v2 ALS tuneado (optuna, base actual del ranker) | 0.03341 |
| **v3 ranker (sobre ese mismo ALS tuneado)** | **0.03578 (+7.1%)** |
| v2 ALS original (factors=128, regularization=0.1, rating crudo) | **0.03864 -- sigue siendo el mejor histórico** |

**Buena noticia, y es la que veníamos buscando confirmar:** a diferencia
del episodio de ALS+optuna, acá la mejora medida con validación cruzada
local (+3.9% promedio, ganó en los 3 seeds) **sí se sostuvo en Kaggle
real** (+7.1% sobre su propia base). Es la primera evidencia directa de
que evaluar con varios seeds en vez de un split fijo efectivamente
protege contra el sobreajuste al proxy local que causó la regresión de
ALS.

**Pero el ranker sigue por debajo del mejor score histórico del
proyecto** (0.03864, el ALS *original* antes de la fórmula
`1+alpha*rating`) -- un -7.4% respecto a ese máximo. La explicación más
directa: el ranker mejora *sobre su propia base* (el ALS tuneado con
optuna, que ya sabíamos deteriorado), pero esa base sigue siendo peor
que el ALS original. El ranker no está fallando -- está sumando valor
real sobre una señal de entrada que no es la mejor disponible.

### Próximo paso natural

Reconstruir el ranker usando el ALS *original* (`factors=128,
regularization=0.1`, la config con 0.03864 confirmado) como la señal de
ALS, en vez del ALS tuneado con optuna, y confirmar en Kaggle si supera
el 0.03864. Requiere decidir primero cómo tratar la fórmula de confianza
(el ALS original usaba rating crudo, sin el `+1` de
`1+alpha*rating` introducido después -- no hay un `alpha` que reproduzca
exactamente "rating crudo" con la parametrización actual, hay que
resolver eso antes de re-generar la señal).

---

## Revert de ALS: vuelve el rating crudo como default

### Lo que se hizo

`als.py::construir_matriz_usuario_libro` ahora soporta `alpha=None`
(confianza = rating crudo, exactamente la fórmula original). `fit_als`
cambia su default a `factors=128, regularization=0.1, alpha=None` --
la única config con score *confirmado en Kaggle* (0.03864, el mejor
histórico). La config tuneada con optuna (`factors=256,
regularization=0.128, alpha=4.718`) queda documentada en el código como
probada y descartada (dio 0.03341 en Kaggle real), no se borra el
conocimiento de por qué se intentó.

Como `submit.py` ya llamaba `fit_als(...)` sin pasar hiperparámetros
explícitos, tanto `"als"` como `"ranker"` heredan el nuevo default
automáticamente -- no hizo falta tocar `submit.py`.

### Hallazgo importante al re-validar con la misma metodología

Se re-corrió `scripts/evaluate_ranker.py` (validación cruzada, 3 seeds)
con el ALS revertido, para comparar apples-to-apples contra la corrida
anterior (con el ALS tuneado):

| | ALS solo | Ranker |
|---|---|---|
| Base tuneada (`factors=256, alpha=4.718`) | 0.100594 ± 0.002488 | 0.104487 ± 0.001780 |
| Base original (`factors=128, reg=0.1`, rating crudo) | 0.094406 ± 0.001359 | 0.097641 ± 0.001152 |

**La validación cruzada sigue mostrando el número local más alto para
la base tuneada** (0.1006 vs 0.0944), pese a que en Kaggle real es al
revés (0.03341 vs 0.03864). Esto es un hallazgo relevante en sí mismo:
la validación cruzada multi-seed protege contra sobreajustar a *un*
split específico (por eso el ranker generalizó bien), pero **no corrige
un sesgo sistemático que comparten todos los splits locales** frente a
la tarea real de Kaggle -- son dos problemas distintos. El segundo sigue
sin resolverse (relacionado con la composición de población de usuarios
que ya se había detectado sin resolver del todo, ver la sección de
"Corrección de metodología" más arriba).

**Lo que sí parece un patrón confiable**: el *uplift relativo* del
ranker sobre su propia base de ALS es consistente entre las dos
configuraciones -- +3.9% local (base tuneada, confirmado +7.1% en
Kaggle) y +3.4% local (base original). Sobre esa base se construye una
hipótesis razonada (no una certeza): si el ranker vuelve a amplificar su
uplift local al pasar a Kaggle real (como pasó la primera vez), la
versión sobre el ALS original podría rondar 0.040-0.041, superando el
0.03864 actual.

### Decisión

Se decidió con el usuario generar y subir esta nueva submission
(`ranker_20260829-152403_ranker-sobre-als-original.csv`) para confirmar
o refutar la hipótesis con datos reales, en vez de esperar a resolver
primero el sesgo sistemático del proxy local (que es un problema más
grande y no bloquea esta prueba puntual).

### Próximos pasos

- **Pendiente, más de fondo:** investigar el sesgo sistemático que hace
  que el proxy local favorezca configs que empeoran en Kaggle real,
  incluso con validación cruzada -- la validación cruzada no es
  suficiente por sí sola, hace falta revisar si el split local sigue sin
  representar bien la población real de `ejemplo.csv` (ver la
  reponderación por actividad que quedó como pendiente en la sección de
  "Corrección de metodología").

---

## Ranker: el uplift no es estable entre bases de ALS

### Resultado real en Kaggle

| Modelo | Kaggle |
|---|---|
| **ALS original solo** | **0.03864 -- sigue siendo el mejor histórico** |
| Ranker sobre ALS original | 0.03815 (**-1.3%** respecto a su propia base) |
| Ranker sobre ALS tuneado | 0.03578 (+7.1% respecto a su base) |
| ALS tuneado solo | 0.03341 |

### La hipótesis no se confirmó

Se había extrapolado, a partir de un solo dato point (el ranker sobre
ALS tuneado mejorando +7.1%), que el uplift relativo del ranker sería
más o menos estable independiente de la calidad de la base de ALS --
la señal local (+3.9% sobre base tuneada, +3.4% sobre base original)
parecía apoyar esa idea. **No se sostuvo**: sobre la base fuerte
(ALS original), el ranker empeora levemente (-1.3%) en vez de mejorar.

**Lectura más ajustada a los datos**: el ranker parece ayudar cuando la
señal de ALS de base es débil (compensa con género/popularidad lo que
ALS no está capturando bien) pero no aporta -- e incluso puede diluir
levemente -- cuando ALS ya es una señal fuerte por sí sola. Con
hiperparámetros de LightGBM conservadores y solo 11 features, el modelo
puede no estar encontrando una combinación que supere a una señal de
ALS ya casi óptima.

**Consecuencia práctica**: `als` (sola, config original) sigue siendo la
mejor entrega confirmada del proyecto. El ranker no se descarta -- ya
demostró que puede sumar valor real en al menos un escenario -- pero no
es hoy una mejora incondicional sobre ALS.

### Próximos pasos

- **No conviene seguir invirtiendo en tunear LightGBM o agregar features
  al ranker todavía** -- sin resolver primero el sesgo sistemático del
  proxy local (arriba), cualquier mejora que se vea localmente corre el
  mismo riesgo de no sostenerse en Kaggle, como ya pasó dos veces.
  Prioridad: investigar ese sesgo antes de invertir más en el ranker.
- Si se retoma el ranker, valdría la pena entender primero *por qué*
  ayuda menos cuando ALS ya es fuerte -- por ejemplo, mirando caso por
  caso qué usuarios el ranker reordena peor que ALS solo, en vez de
  seguir agregando features/hiperparámetros a ciegas.

---

## Investigando el "sesgo sistemático": no es sesgo, es tamaño de muestra

### Dos hipótesis probadas y descartadas

**1. ¿Es solo aleatoriedad del entrenamiento de ALS?** Se fiteó ALS 5
veces con el mismo split fijo (`seed=42`) y distintas semillas internas
del modelo. Desvío entre esos 5 fits: **0.000559**. Comparado contra el
desvío entre splits distintos (0.001359, de la validación cruzada
anterior), la aleatoriedad propia del entrenamiento explica una parte
menor -- la mayoría de la variación entre splits viene de *qué usuarios*
caen en cada partición, no de la optimización de ALS en sí. Un ensamble
de semillas de modelo ayudaría un poco, pero no es la explicación
principal.

**2. ¿Es que el proxy local no representa la población de Kaggle?** Se
recalculó el NDCG por bucket de actividad y se reponderó por la
distribución real de `ejemplo.csv` (usuarios pesados dominan, ver
sección anterior). Resultado: **el config tuneado le sigue ganando al
original en casi todos los buckets, incluido el de 100+ interacciones
que es el 47% de la población de Kaggle** -- reponderar no cambia el
signo de la comparación (0.0686 tuneado vs 0.0640 original). Esta
hipótesis tampoco explica la discrepancia.

### La explicación real: las diferencias en Kaggle están dentro del ruido

Se estimó el error estándar esperado de un NDCG@20 agregado sobre los
**832 usuarios reales** de `ejemplo.csv` (usando la distribución de
NDCG por usuario que sí podemos medir localmente, ponderada por la
mezcla de actividad real de Kaggle):

| | valor |
|---|---|
| Error estándar estimado (n=832, mezcla real de actividad) | **0.00651** |
| Intervalo de confianza ~95% de una sola medición | **± 0.01275** |

Comparado contra las diferencias reales que motivaron decisiones en esta
sesión:

| Comparación | Diferencia observada | ¿Supera 1 error estándar? |
|---|---|---|
| ALS original vs tuneado | 0.00523 | No (0.65 SE) |
| Ranker vs ALS (base original) | -0.00049 | No (0.08 SE) |
| Ranker vs ALS (base tuneada) | 0.00237 | No (0.36 SE) |

**Ninguna de las tres diferencias que usamos para tomar decisiones esta
sesión llega siquiera a 1 error estándar**, y mucho menos a los ~2 SE
que se suelen pedir para hablar de una diferencia "real". Con un test
set de 832 usuarios y una métrica donde el 81% de los usuarios tiene
NDCG=0 (todo o nada, muy dispersa), Kaggle **no tiene la resolución
necesaria para distinguir configs que difieren tan poco** -- no es que
el proxy local esté sesgado, es que la vara de comparación (una sola
submission de Kaggle) es demasiado ruidosa para el tamaño de efecto que
estábamos tratando de medir.

### Implicancia importante (reformula varias conclusiones de esta sesión)

Esto no invalida el trabajo de esta sesión, pero sí cambia cómo hay que
leerlo:
- El revert de ALS a `factors=128/regularization=0.1` **puede haber sido
  una decisión tomada sobre ruido**, no sobre una diferencia real. No se
  deshace (sigue siendo razonable dado que también hay una explicación
  causal plausible -- sobreajuste de optuna a un split), pero la
  confianza en que es *definitivamente* mejor que la tuneada debería
  bajar.
- Lo mismo aplica al resultado del ranker: la conclusión "no aporta
  sobre una base fuerte" está basada en una diferencia (-0.00049) que es
  ruido puro, prácticamente cero. **No hay evidencia real de que el
  ranker empeore sobre ALS original** -- tampoco de que mejore. El dato
  es simplemente inconcluyente.
- **De acá en adelante, una sola submission de Kaggle no alcanza para
  decidir entre configs que difieren menos de ~0.01-0.013 en NDCG@20.**
  Diferencias de esa magnitud (que es la escala de casi todo lo que
  venimos probando después de v2) necesitan otra fuente de evidencia:
  confiar más en la validación cruzada local (que tiene un error
  estándar mucho menor por usar ~10x más usuarios) para decisiones finas,
  y reservar submissions reales de Kaggle para confirmar cambios de
  magnitud grande (como v0→v1→v2, donde las diferencias eran 3-10x más
  grandes que el ruido) en vez de ajustes incrementales.

### Próximos pasos

- **Recalibrar el criterio de decisión**: no tratar una sola submission
  de Kaggle como verdad definitiva para diferencias chicas. Si hace
  falta desempatar entre dos configs cercanas, considerar promediar
  varias submissions (si el límite diario de Kaggle lo permite) antes de
  concluir cuál es mejor.
- **No hay una acción de código que "arregle" esto** -- es una
  limitación del tamaño del test set de Kaggle, no de la metodología
  local. Lo que sí cambia es cuánto peso ponerle a cada submission
  futura.

---

## Se retoma el ranker: features nuevas + tuneo de LightGBM

### Objetivo

Con el criterio recalibrado (confiar en la validación cruzada local para
decisiones finas, reservar Kaggle para confirmar mejoras grandes), se
retomaron las dos líneas que habían quedado pendientes: features nuevas
para el ranker y tuneo de hiperparámetros de `LGBMRanker` -- esta vez con
CV desde el arranque, nunca contra un solo split.

### Features nuevas (`ranker.py::calcular_features_auxiliares`)

Con buena cobertura entre libros con interacción (autor conocido 99.99%,
año de edición 100%, verificado antes de implementar):
- `en_autor_leido` / `n_libros_autor_leidos`: ¿el usuario ya leyó a este
  autor?, cuántas veces.
- `anio_edicion_dif`: diferencia entre el año de edición del candidato y
  el promedio de lo que lee el usuario (clásicos vs. novedades).
- `n_generos_distintos_usuario`: diversidad de género del historial
  (generalista vs. especializado).
- `dias_desde_ultima_interaccion_usuario`: recencia general del usuario,
  relativa a la fecha más reciente de `train_candidatos`.

Todas calculadas sobre `train_candidatos` únicamente (mismo principio
que las señales base) para no filtrar información al ranker. Se dejaron
afuera, a propósito, editorial (señal esperada más débil) y co-lectura
ítem-ítem (cara de calcular bien, se documenta como idea futura).

### Refactor: `ranker.py::evaluar_pipeline`

La lógica de split de tres niveles + fit + evaluación que antes vivía
duplicada en `scripts/evaluate_ranker.py` se movió a una función en
`ranker.py` (`evaluar_pipeline`), reusada tanto por ese script como por
`scripts/tune_ranker.py` (nuevo) -- evita mantener dos copias del mismo
pipeline.

### Resultado: features nuevas (grande y consistente)

| | NDCG@20 (CV, 3 seeds) |
|---|---|
| ALS solo | 0.094406 ± 0.001359 |
| Ranker, features viejas (11) | 0.097641 ± 0.001152 |
| **Ranker, +5 features nuevas (16)** | **0.106815 ± 0.001498** |

**+9.4% sobre el ranker sin estas features, positivo en los 3 seeds
individualmente** (0.10518 / 0.10713 / 0.10813) -- la mejora más sólida
de toda la sesión con el ranker, muy por encima del "piso de ruido" que
identificamos en la sección anterior. Autor/año/diversidad/recencia
aportan señal real que ALS + popularidad + género no estaban capturando.

### Resultado: tuneo de LightGBM (marginal, no se adopta)

`scripts/tune_ranker.py`: optuna sobre `num_leaves`, `learning_rate`,
`n_estimators`, `min_child_samples`, `reg_alpha`, `reg_lambda` -- cada
trial evaluado con 2 seeds (nunca 1), mejor config confirmado con los 3
seeds completos al final:

| | NDCG@20 (CV, 3 seeds) |
|---|---|
| Conservador (`num_leaves=31, learning_rate=0.05, n_estimators=200`) | 0.106815 ± 0.001498 |
| Tuneado (`num_leaves=23, learning_rate=0.017, n_estimators=391, ...`) | 0.108127 ± 0.001499 |

+1.2%, pero la mejora absoluta (0.0013) es **menor que el desvío entre
seeds (0.0015)** -- aplicando el mismo criterio que aprendimos con
Kaggle (no confiar en diferencias menores que la variabilidad natural),
esto es indistinguible de ruido con solo 3 seeds. **Se decide no
adoptarlo** -- quedan los hiperparámetros conservadores en producción,
más simples y con básicamente el mismo resultado.

### Decisión

Se wirean las features nuevas (ya estaban en `submit.py` desde que se
implementaron) y se genera una submission
(`ranker_20260829-170942_ranker-features-autor-anio-genero-recencia.csv`)
para confirmar en Kaggle -- a diferencia de las mejoras marginales de
antes, esta es grande (+13.1% sobre ALS) y consistente en los 3 seeds,
justo el tipo de cambio que el criterio recalibrado dice que vale la
pena confirmar con una submission real.

### Resultado real en Kaggle: nuevo récord del proyecto

| Modelo | Kaggle |
|---|---|
| v0 — popularidad | 0.01024 |
| v1 — popularidad segmentada | 0.01558 |
| ALS tuneado (optuna, descartado) | 0.03341 |
| Ranker sobre ALS tuneado | 0.03578 |
| Ranker sobre ALS original (features viejas) | 0.03815 |
| ALS original solo (récord anterior) | 0.03864 |
| **Ranker + features nuevas (autor/año/género/recencia)** | **0.04457 — nuevo récord** |

**+15.3% sobre el récord anterior, +16.8% sobre el ranker sin estas
features.** Es la primera vez en toda esta ronda de experimentos que una
mejora se confirma limpiamente de punta a punta: grande y consistente en
validación cruzada local (multi-seed, positiva en cada seed individual)
*y* grande en Kaggle real -- sin la sorpresa negativa que tuvimos con
ALS+optuna y con el primer intento del ranker. Confirma la apuesta
metodológica de esta sección: **priorizar features de dominio (más
información real para el modelo) por sobre tunear hiperparámetros**
cuando hay que elegir dónde invertir el esfuerzo.

**`"ranker"` pasa a ser el modelo de referencia del proyecto**, por
encima de `"als"`.

### Próximos pasos

- Ideas de features que quedaron afuera por ahora y podrían seguir
  empujando en la misma dirección: co-lectura ítem-ítem (cara de
  calcular bien), editorial, metadata de `resumen`/texto.
- Con `"ranker"` como nueva referencia, reconsiderar si vale la pena
  retomar el tuneo de LightGBM ahora que la base de features cambió (se
  descartó sobre las features viejas, podría comportarse distinto sobre
  las nuevas) -- siempre con la misma validación cruzada multi-seed.
- Sigue pendiente, de fondo, investigar si el split local puede
  acercarse más a la tarea real de Kaggle (más allá de la validación
  cruzada) -- aunque esta vez la brecha local-Kaggle fue mucho más
  razonable que en corridas anteriores.

---

## Co-lectura, editorial, resumen y género macro: confirmado pese a no superar el desvío local

### Objetivo

Se retomaron las tres ideas de features que habían quedado pendientes
(co-lectura ítem-ítem, editorial, metadata de `resumen`) y, a pedido del
usuario, se sumó una cuarta línea propia: una observación sobre
`libros.genero` (muchas categorías únicas, la gran mayoría
subrepresentadas) que derivó en co-diseñar con el usuario una
macro-taxonomía de género y dos features numéricas nuevas a partir de
ella.

### Co-lectura, editorial y resumen (`ranker.py`)

- **`score_coleido`**: la razón por la que esta feature se había dejado
  afuera ("cara de calcular bien") no aplica calculándola como un solo
  matmul disperso: `cooc = X.T @ X` (matriz ítem×ítem, reusando la
  matriz binaria de `fit_als`) se construye en ~3s para 48k libros/461k
  interacciones. Por usuario, el score contra sus candidatos se obtiene
  acotado al batch de usuarios consultados (nunca un cruce contra los
  ~10.7k usuarios completos) -- ver `_calcular_cooccurrencia`.
- **`en_editorial_leida`/`n_libros_editorial_leidos`**: calco exacto del
  patrón ya usado para autor.
- **`sim_resumen_historial`**: `TfidfVectorizer` (scikit-learn, ya en
  `pyproject.toml`, sin nueva dependencia) sobre `resumen` de todo el
  catálogo (metadata estática del libro, no depende del split -- igual
  que autor/año de edición) + perfil de texto por usuario calculado
  **solo** con `train_candidatos` (eso sí es interacción-derivado). La
  trampa evitada: nunca se calculó un producto usuario×catálogo
  completo (eso sí explota en memoria) -- la similitud se computa solo
  para los pares (usuario, candidato) que ya existen en `candidatos_df`.
  Ver `_calcular_perfil_texto`.

Evaluado con la misma validación cruzada de 3 seeds de siempre
(`scripts/evaluate_ranker.py`): **mixto**. Empeora en seed=42 (-0.00024)
respecto al baseline de 16 features, y la mejora promedio (+1.15%) es
menor que el desvío entre seeds -- no se confirma (ver `log.csv`,
2026-08-30). Un ablation sacando editorial (la señal más débil según
`feature_importances_`) mejora un poco el panorama (ya no empeora en
ningún seed) pero tampoco alcanza a superar el desvío por sí solo.

### Género macro: co-diseño con el usuario

El usuario notó que `libros.genero` tenía muchas categorías
subrepresentadas y pidió co-construir una limpieza + una categoría
macro + features numéricas a partir de eso, en vez de que se implementara
unilateralmente. Con los datos reales:

- `_normalizar_genero` (`popularity_segmentada.py`) hacía solo
  `strip().lower()`, dejando 54 categorías -- pero 2 pares eran el mismo
  género real duplicado por una tilde inconsistente en el dato crudo
  (`"clásicos de la literatura"` 754 vs `"clasicos de la literatura"` 2;
  `"biografías, memorias"` 2087 vs `"biografiás, memorias"` 2). Ignorar
  acentos (NFKD) las une sin mapa de alias a mano -- **52 categorías
  reales**.
- Con esas 52, se acordó con el usuario una taxonomía de **10
  macro-géneros de dominio** (`MACRO_GENERO_POR_GENERO`), confirmada por
  el usuario con la distribución real de libros por familia (de 14.746
  en "narrativa y clásicos" a 1.799 repartidos en un catch-all
  "práctico y misceláneo" de 27 categorías minúsculas). Ver
  `experiments/decisiones.md` para la tabla completa.
- Dos features nuevas confirmadas por el usuario (las dos juntas, no una
  sola): `popularidad_genero_macro_candidato` (score bayesiano tipo
  `fit_popularity`, pero pooleado a las 10 familias en vez de las 52
  categorías granulares -- muchas de esas categorías tienen muy pocas
  interacciones para un score confiable) y `frecuencia_genero_macro_usuario`
  (proporción 0-1 del historial del usuario en el macro-género del
  candidato -- señal graduada, más rica que la diversidad
  binaria/conteo que ya existía).

### Resultado: positivo en los 3 seeds, pero no supera el desvío

| Variante | CV 3 seeds (media ± desvío) | vs. baseline (16 features) |
|---|---|---|
| Baseline (16 features, autor/año/género/recencia) | 0.106815 ± 0.001498 | -- |
| +co-lectura/editorial/resumen (20) | 0.108039 ± 0.003079 | mixto, empeora en seed=42 |
| +co-lectura/resumen sin editorial (18) | 0.107802 ± 0.002443 | +0.92%, no supera el desvío |
| **+género macro también (22, con editorial)** | **0.108740 ± 0.002955** | **positivo en los 3 seeds** |

La versión de 22 features fue la primera de toda esta ronda con mejora
**positiva en los 3 seeds individualmente** (+0.00038 / +0.00214 /
+0.00326 -- creciente, no errático como el ruido de rondas anteriores).
Pero aplicando la regla mecánica de siempre (mejora promedio > desvío
entre seeds): +1.80% (0.00193) sigue siendo *menor* que el desvío
observado (0.00296) -- la regla estricta diría "no confirmado".

### Decisión: confirmar con Kaggle en vez de seguir analizando el número local

Caso límite genuino: la regla estricta no lo confirma, pero el patrón
(siempre positivo, no errático) es distinto a los episodios anteriores
de puro ruido (ALS+optuna, tuneo de LightGBM). Es exactamente el tipo de
situación que la sección "Recalibrar el criterio de decisión" (más
arriba) ya anticipó: cuando el local no alcanza a desempatar solo, usar
una submission real. Se decidió con el usuario confirmar con Kaggle en
vez de seguir puliendo el análisis local.

**Resultado real: 0.04658 -- nuevo récord del proyecto, +4.5% sobre el
récord anterior** (ranker con autor/año/género/recencia + ALS original,
0.04457). Primera vez que un resultado que *no* superaba la regla
estricta local se confirma igual en Kaggle -- refuerza la idea de que la
regla (mejora > desvío) es un buen filtro para *no* gastar submissions
en ruido evidente, pero no es infalible en los casos límite: ahí,
confirmar con una submission real sigue siendo el criterio final, como
ya se había concluido antes.

`"ranker"` (ahora con 22 features: ALS + popularidad + género +
autor/año/diversidad/recencia + co-lectura + editorial + resumen +
popularidad/frecuencia de macro-género) sigue siendo el modelo de
referencia del proyecto, con récord actualizado.

### Próximos pasos

- Con editorial dentro del set confirmado pese a ser la señal más débil
  en `feature_importances_`, sigue sin estar claro si sacarla mejoraría
  el resultado aún más -- no se probó esa combinación específica (22
  features sin editorial) contra Kaggle.
- Retomar el tuneo de LightGBM (descartado dos veces sobre bases de
  features distintas) ahora que la base cambió de nuevo -- siempre con
  la misma validación cruzada multi-seed.
- La brecha entre "regla estricta" y "confirmado en Kaggle" de esta
  ronda sugiere que, con solo 3 seeds, la regla puede ser demasiado
  conservadora para mejoras reales pero chicas -- posible línea futura:
  evaluar con más seeds antes de descartar un caso límite como este.

---

## Tamaño de catálogo de editorial: segundo caso límite confirmado en Kaggle

### Objetivo

El usuario preguntó, después de confirmar la ronda de género macro, si
tendría sentido hacer algo similar con `editorial` ("imagino que también
hay editoriales grandes y muchas chicas con pocos libros"). Antes de
implementar nada, se investigaron los datos reales.

### Por qué editorial es un caso distinto de género

- **2.762 editoriales distintas** entre libros con interacción (vs. 52
  categorías de género) -- una cola muchísimo más larga: 91% tiene menos
  de 20 libros, **51% tiene exactamente 1 libro**. Se necesitan 152
  editoriales para cubrir el 80% del catálogo.
- A diferencia de género, **no hay una agrupación temática natural**
  entre editoriales -- "Anagrama" y "Alfaguara" no comparten ningún
  "dominio" más que ser editoriales grandes. Forzar una macro-taxonomía
  categórica (6-10 "familias de editoriales") habría sido artificial, a
  diferencia de la de género, que sí tenía una semántica de dominio
  clara detrás.
- Conclusión co-diseñada con el usuario: en vez de una taxonomía
  categórica, una **feature numérica simple de tamaño de catálogo**
  (`n_libros_editorial_catalogo` -- cuántos libros tiene esa editorial en
  *todo* `libros`, no solo los leídos) captura la intuición real
  ("hay grandes y muchas chicas") sin inventar cortes de bucket
  arbitrarios. Se descartaron explícitamente dos alternativas más
  complejas (score bayesiano pooleado por editorial, buckets
  grande/mediana/chica) por no tener una justificación tan directa como
  con género.

### Resultado: mismo patrón límite que la ronda anterior

| | CV 3 seeds (media ± desvío) | vs. la versión anterior (22 features) |
|---|---|---|
| 22 features (confirmado en Kaggle: 0.04658) | 0.108740 ± 0.002955 | -- |
| **+ `n_libros_editorial_catalogo` (23)** | **0.109735 ± 0.003719** | **positivo en los 3 seeds** |

Otra vez positivo en los 3 seeds (+0.00012 / +0.00127 / +0.00159), y otra
vez la mejora promedio (+0.92%) no supera el desvío entre seeds (que
además creció más que la mejora esta vez) -- la regla estricta volvería
a decir "no confirmado". `feature_importances_` ubica
`n_libros_editorial_catalogo` en la mitad de la tabla, consistentemente
por delante de `en_editorial_leida`/`n_libros_editorial_leidos` en los 3
seeds -- confirma que el tamaño global de la editorial es una señal más
útil que si el usuario ya leyó de ahí.

### Decisión: confirmar con Kaggle de nuevo

Se decidió con el usuario repetir el mismo criterio que la ronda
anterior (confirmar con una submission real en vez de seguir analizando
el número local), justo porque la vez pasada un caso límite idéntico
resultó ser una mejora real.

**Resultado real: 0.04831 -- nuevo récord del proyecto, +3.7% sobre el
récord anterior** (0.04658). Es la **segunda vez seguida** que un
resultado que no superaba la regla estricta local (positivo en los 3
seeds, pero mejora menor que el desvío) se confirma como mejora real en
Kaggle -- refuerza la idea de la sección anterior: con solo 3 seeds, la
regla puede ser demasiado conservadora para mejoras reales pero chicas,
y "positivo en los 3 seeds sin ser errático" parece ser, en la práctica,
una señal más confiable que la comparación mecánica contra el desvío.

`"ranker"` (ahora 23 features) sigue siendo el modelo de referencia del
proyecto, con récord actualizado.

### Próximos pasos

- Con dos casos límite seguidos confirmados como mejoras reales, vale la
  pena reconsiderar el criterio de decisión en sí: quizás "positivo en
  los 3 seeds individualmente" (sin importar si supera el desvío) sea un
  filtro suficiente para justificar una submission, reservando la regla
  del desvío para decisiones que *no* involucran gastar una submission
  (como elegir hiperparámetros).
- Sigue sin probarse la combinación de 22 features sin editorial contra
  Kaggle (ver "Próximos pasos" de la sección anterior).

---

## ¿Sacar editorial? Ablation resuelve la pregunta pendiente: no conviene

Se probó la pregunta que había quedado pendiente: las 23 features
actuales, pero sin `en_editorial_leida`/`n_libros_editorial_leidos` (la
señal de editorial basada en historial, la más débil según
`feature_importances_` en todas las rondas anteriores) -- manteniendo
`n_libros_editorial_catalogo` (tamaño, recién confirmada).

| | CV 3 seeds (media ± desvío) |
|---|---|
| 23 features (confirmado en Kaggle: 0.04831) | 0.109735 ± 0.003719 |
| 21 features (sin `en_editorial_leida`/`n_libros_editorial_leidos`) | 0.109287 ± 0.003276 |

A diferencia de los dos casos límite anteriores (siempre positivos en
los 3 seeds), esta vez el resultado es **negativo en 2 de 3 seeds**
(seed=7: -0.00081, seed=123: -0.00067, seed=42: +0.00013) -- media
-0.41% peor. Conclusión: aunque `en_editorial_leida`/
`n_libros_editorial_leidos` son las features individualmente más débiles
del set, **no son ruido puro** -- sacarlas empeora el conjunto. No se
gastó una submission en esto: a diferencia de los casos límite
anteriores (todos positivos en los 3 seeds, lo que justificó confirmar
con Kaggle), acá la señal local ya apunta claramente en contra. Se
mantienen las 23 features tal cual.

---

## Separar armado de candidatos de tuneo de LightGBM (antes de tunear)

### Objetivo

Antes de retomar el tuneo de `LGBMRanker` (pendiente en `decisiones.md`)
sobre la base actual de 23 features, se midió cuánto tardaría con el
diseño existente de `scripts/tune_ranker.py`, que llama a
`ranker.evaluar_pipeline` completo en cada trial de optuna.

### Medición (una corrida completa, seed=42, 23 features)

| Etapa | Tiempo | ¿Depende de `lgbm_params`? |
|---|---|---|
| ALS + popularidad + género + `calcular_features_auxiliares` (TF-IDF, co-lectura, macro-género) | ~18s | No |
| `generar_candidatos_con_features` (train_ranker, ~7.9k usuarios) | ~129s | No |
| `armar_dataset_entrenamiento` | ~22s | No |
| `fit_ranker` (LightGBM) | ~22s | **Sí -- lo único que cambia entre trials** |

(Se suma una segunda llamada a `generar_candidatos_con_features` para
`test_final`, de magnitud similar a la de `train_ranker`.)

De los ~350-450s que tarda una corrida completa, **solo ~22s dependen
de los hiperparámetros que se están tuneando** -- el resto (armar
candidatos y el dataset de entrenamiento) es idéntico entre trials para
el mismo seed. Con el diseño anterior (10 trials × 2 seeds + confirmación
final con 3 seeds), el tuneo completo hubiera tardado **~2.5 horas**.

### Decisión: separar `preparar_pipeline` de `evaluar_con_params`

`ranker.py::evaluar_pipeline` se partió en dos funciones (ver sus
docstrings para el detalle completo):

- **`preparar_pipeline(interacciones, libros, seed, n_por_fuente, k)`**:
  todo lo que no depende de `lgbm_params` -- split de tres niveles, fit
  de ALS/popularidad/género, `calcular_features_auxiliares`, las dos
  llamadas a `generar_candidatos_con_features`, `armar_dataset_entrenamiento`,
  y el NDCG de ALS solo. Devuelve un dict "contexto" reusable.
- **`evaluar_con_params(contexto, lgbm_params)`**: la parte barata
  (~22s) -- entrena `LGBMRanker` con una config puntual sobre el
  contexto ya armado y evalúa NDCG@k.
- `evaluar_pipeline` queda como atajo de conveniencia (`preparar_pipeline`
  + `evaluar_con_params`) para el caso de una sola config por seed --
  `scripts/evaluate_ranker.py` no necesitó ningún cambio.

`scripts/tune_ranker.py` ahora arma el contexto **una sola vez por
seed** (cacheado en `_CONTEXTOS`) y prueba todas las configuraciones de
optuna contra ese mismo contexto vía `evaluar_con_params`. Mismo alcance
de búsqueda (10 trials, 2 seeds por trial, confirmación final con 3
seeds), de **~2.5 horas a ~20-25 minutos**.

### Verificación de que no cambió el resultado

Se confirmó que el refactor es puramente de reorganización: corriendo
`scripts/evaluate_ranker.py` para seed=42 antes y después del cambio, el
NDCG@20 del ranker dio exactamente igual (`0.105679`) -- mismo pipeline,
solo separado en dos funciones en vez de una.

---

## Tuneo de LightGBM sobre 23 features: tercera vez que no se adopta

### Objetivo

Con el pipeline ya separado (sección anterior), se corrió
`scripts/tune_ranker.py` sobre la base actual de 23 features: 10 trials
de optuna, 2 seeds por trial, confirmación final con los 3 seeds
completos.

### Corrida real: dos jobs en background se cortaron solos

La búsqueda de optuna (10 trials × 2 seeds) terminó bien, en 1324.6s
(~22 min, tal como estimaba el refactor). Pero el proceso se cortó
("killed") durante la confirmación final con los 3 seeds -- y un
segundo intento de correr *solo* esa confirmación también se cortó,
esta vez casi de inmediato. En vez de seguir reintentando en background,
se corrió la confirmación **seed por seed, en primer plano** (tres
llamadas de ~370s cada una, bien por debajo del límite de una corrida
en primer plano) -- inestabilidad del entorno de esta sesión, no del
código.

### Resultado: mixto, no se adopta

Mejor config encontrado por optuna: `num_leaves=7, learning_rate=0.134,
n_estimators=280, min_child_samples=82, reg_alpha=0.0012,
reg_lambda=7.58`.

| | seed=42 | seed=7 | seed=123 | media ± desvío |
|---|---|---|---|---|
| Conservador (actual, confirmado en Kaggle: 0.04831) | 0.10568 | 0.11054 | 0.11299 | 0.109735 ± 0.003719 |
| Tuneado (optuna) | 0.10637 | 0.11113 | 0.11209 | 0.109864 ± 0.003066 |

Mixto: positivo en seed=42 (+0.00069) y seed=7 (+0.00059), negativo en
seed=123 (-0.00090). Mejora promedio +0.12% (0.00013) -- indistinguible
de ruido, ni siquiera cerca del patrón "positivo en los 3 seeds" que
justificó confirmar los dos casos límite anteriores (género macro,
tamaño de editorial). Es la **tercera vez** que el tuneo de LightGBM en
este proyecto da un resultado así de chico e inconsistente (las dos
anteriores, sobre bases de features distintas, tampoco se adoptaron --
ver sección "Se retoma el ranker" y `experiments/log.csv`). Se
mantienen los hiperparámetros conservadores (`num_leaves=31,
learning_rate=0.05, n_estimators=200`) en producción -- no se gasta una
submission en esto, ni hace falta: la señal local ya es claramente
insuficiente, sin ningún patrón consistente que sugiera lo contrario.

### Próximos pasos

- Con tres intentos de tuneo de LightGBM (sobre tres bases de features
  distintas) dando siempre mejoras menores al ruido, quizás no valga la
  pena seguir insistiendo con esto -- el patrón sugiere que los
  hiperparámetros conservadores ya están cerca de un óptimo razonable
  para este tipo de dataset/tarea, y que el margen real sigue estando en
  features nuevas, no en hiperparámetros.


## País del usuario (`vive_en`): explorada y descartada

### Objetivo

Retomar el primer punto de "próximos pasos pendientes" de `decisiones.md`:
sumar `vive_en` (ubicación del lector, nunca explorada) como feature del
ranker.

### Exploración de los datos, co-diseñada con el usuario

`lectores.vive_en` es texto libre ("Ciudad - País", a veces solo "País",
a veces vacío) con 1.585 valores distintos. Extrayendo el país (lo que
sigue al último " - "): **98 países reales, muy sesgado** (68% de los
lectores vive en España, 7.675/11.285), con ~9% de valores "desconocido"
explícitos (534 vacíos + 457 con el placeholder `"¿?"` que usa el
dataset). Un valor llamativo, `"Santiago - Cote d'Ivoire"` (75 casos),
es casi seguro un artefacto de geocodificación upstream (Santiago es top
de Chile, no de Costa de Marfil) -- no se intentó corregir, se dejó
como un país más de la cola larga.

Los libros no tienen país propio (no hay país de la editorial ni nada
así), así que no hay cruce directo lector↔libro por ubicación -- la
única feature con sentido es del mismo tipo que `popularity_segmentada`
v1 (fallback por género/franja): popularidad del libro entre lectores
del *mismo país que el usuario*.

Antes de implementar, se co-diseñó con el usuario (memoria del
proyecto: los tradeoffs de dominio se deciden en conjunto, no se
implementa una propuesta ya cerrada):
- **Granularidad**: se descartó agrupar en macro-regiones (España/
  Latinoamérica/otro) o saltar la feature directamente -- se eligió país
  tal cual (98 categorías), aceptando el sesgo hacia España.
- **Desconocidos**: se eligió tratarlos como categoría propia
  ("desconocido") en vez de caer a un fallback de popularidad global,
  a diferencia de cómo se manejan los `NaN` de franja de nacimiento
  (que simplemente se descartan).

### Implementación

`popularity_segmentada.py`: `pais_por_usuario` (parsea y normaliza
`vive_en` como género -- minúsculas, sin acentos -- confirmado que no
queda ninguna colisión real entre países al normalizar así) y
`fit_popularity_por_pais` (popularidad bayesiana segmentada por país,
mismo patrón que `fit_popularity_por_franja_nacimiento`). `ranker.py`:
nueva feature `popularidad_pais_candidato` -- a diferencia de macro-género/
editorial (propiedades del *libro*, un score por `id_libro`), acá el
segmento lo define el *usuario*, así que el lookup queda anidado por
país (`{pais: {id_libro: score}}`).

### Resultado: negativo, no se confirma

| | seed=42 | seed=7 | seed=123 | media ± desvío |
|---|---|---|---|---|
| 23 features (confirmado en Kaggle: 0.04831) | 0.10568 | 0.11054 | 0.11299 | 0.109735 ± 0.003719 |
| 24 features (+ `popularidad_pais_candidato`) | 0.10565 | 0.10942 | 0.11193 | 0.109002 ± 0.003161 |

Negativo en 2 de 3 seeds (seed=7: -0.00112, seed=123: -0.00106) y
prácticamente empatado en el tercero (seed=42: -0.00003) -- media
-0.67%. No cumple el criterio de "positivo en los 3 seeds
individualmente" que el usuario confirmó como suficiente para justificar
una submission (género macro, tamaño de editorial); es más parecido al
caso de la ablation de `en_editorial_leida` (negativo en 2/3 seeds), que
tampoco se subió a Kaggle. `feature_importances_` la ubica en la mitad
de la tabla en los 3 seeds -- no es la señal más floja del set, pero el
conjunto en su totalidad empeora con ella adentro.

**No se gastó una submission.** Se revirtió la feature del ranker
(`FEATURES` vuelve a las 23 confirmadas en Kaggle) para que
`submit.py --model ranker` no regresione respecto al récord actual. Se
mantienen `pais_por_usuario`/`fit_popularity_por_pais` en
`popularity_segmentada.py` (testeadas, sin usar en el ranker) -- decisión
explícita del usuario, igual que `franja_nacimiento_por_usuario` hoy: no
cuesta nada dejarlas por si vale la pena retomar la idea con otro
enfoque (otra granularidad, cruzarla con otra señal).


## Franja de nacimiento del lector: explorada y descartada (con un fix de datos que sí queda)

### Objetivo

Retomar el segundo candidato de "próximos pasos pendientes": franja de
nacimiento del lector (probada en v1 como fallback, nunca llevada al
ranker), con el mismo patrón de `popularidad_pais_candidato` de esta
sesión: popularidad bayesiana segmentada por franja *del usuario*.

### Hallazgo de calidad de datos: el sentinel 1910

Antes de implementar, mirando la distribución de `nacimiento`: ~30% de
los lectores tiene `nacimiento` inválido o faltante (vs. ~9% de
"desconocido" en país). Dentro de la franja "1910s" (438 lectores), 415
tienen el valor *exacto* 1910 -- muy distinto de los 1-9 casos de
1911-1917. Ese patrón no es gente real nacida en 1910 (implicaría más
de 110 años en un dataset de lectura activa): es casi seguro un default
de formulario. Confirmado con el usuario, se decidió tratar
`nacimiento == 1910` igual que un `nacimiento` inválido -- categoría
propia `"desconocido"`, no una década real.

Como `franja_nacimiento_por_usuario`/`fit_popularity_por_franja_nacimiento`
son compartidas con el fallback de v1 (`popularity_segmentada.recomendar_por_usuario`,
género → franja → global), se decidió con el usuario tocar esas
funciones compartidas (no una copia aislada para el ranker) -- v1 ahora
también trata "desconocido" como su propia categoría en el fallback de
franja, en vez de saltarla directo a popularidad global para ese ~30%
de usuarios.

### Implementación

Nuevas constantes `FRANJA_DESCONOCIDA`/`NACIMIENTO_SENTINEL` en
`popularity_segmentada.py`. `franja_nacimiento_por_usuario` ahora
devuelve una entrada para *todos* los lectores (antes descartaba
`nacimiento` inválido). `ranker.py`: nueva feature
`popularidad_franja_candidato`, calco exacto de `popularidad_pais_candidato`
(mismo patrón de lookup anidado `{franja: {id_libro: score}}`, porque el
segmento lo define el usuario, no el libro).

### Resultado: mixto, negativo en 2 de 3 seeds, no se confirma

| | seed=42 | seed=7 | seed=123 | media ± desvío |
|---|---|---|---|---|
| 23 features (confirmado en Kaggle: 0.04831) | 0.10568 | 0.11054 | 0.11299 | 0.109735 ± 0.003719 |
| 24 features (+ `popularidad_franja_candidato`) | 0.10638 | 0.10984 | 0.11168 | 0.109300 ± 0.002688 |

Positivo en seed=42 (+0.00070), negativo en seed=7 (-0.00070) y
seed=123 (-0.00132) -- media -0.40%. No cumple "positivo en los 3
seeds". Mismo desenlace que país (sección anterior), pese a que género
del lector/franja de nacimiento está mejor distribuido que país (menos
sesgado a una sola categoría dominante) -- la mejor distribución no
alcanzó para convertirlo en una señal útil para el ranker.

**No se gastó una submission.** Se revirtió la feature del ranker
(`FEATURES` vuelve a las 23 confirmadas), pero **se mantiene el fix del
sentinel 1910** -- es una mejora de calidad de dato independiente de si
esta feature puntual sirve, y beneficia también a v1. Las funciones
`franja_nacimiento_por_usuario`/`fit_popularity_por_franja_nacimiento`
quedan sin usar en el ranker, mismo criterio que `pais_por_usuario`/
`fit_popularity_por_pais`.


## Señales cruzadas lector↔libro: nuevo récord del proyecto (0.04855)

### Objetivo

Con país y franja de nacimiento descartados (secciones anteriores), el
usuario pidió probar sistemáticamente los 3 candidatos de señales
cruzadas lector↔libro que quedaron del brainstorm, en vez de seguir con
más variantes de "popularidad segmentada por X" (el patrón que ya
falló dos veces).

### Hallazgo previo: `lectores.genero` es el género DECLARADO del lector

Antes de implementar, se encontró que `lectores.genero` (Mujer 3925,
Hombre 3712, "-" 3648 -- ~35%/33%/32%, sin variantes de capitalización
que normalizar) es una columna con el mismo nombre que `libros.genero`
(género literario) pero un significado completamente distinto -- una
trampa de nombres real, documentada explícitamente en el código
(`popularity_segmentada.genero_lector_por_usuario`) para no confundirla
con el resto del módulo, que siempre habla de género *literario*.

### Las 3 features implementadas juntas

1. **`popularidad_genero_lector_candidato`**: popularidad bayesiana
   segmentada por género declarado del usuario (Mujer/Hombre/desconocido
   -- `"-"` mapeado a categoría propia). Mismo patrón exacto que
   `popularidad_pais_candidato`/`popularidad_franja_candidato`, pero con
   mejor balance de categorías que las dos anteriores.
2. **`frecuencia_genero_macro_por_genero_lector`**: a diferencia de
   `frecuencia_genero_macro_usuario` (historial *individual*), mide qué
   proporción de las interacciones de la *cohorte* que declaró el mismo
   género que el usuario cae en el macro-género del candidato -- señal
   de grupo, no de persona.
3. **`edad_lector_al_publicarse`** (`anio_edicion - nacimiento`): cruza
   una propiedad del lector con una del libro directamente -- distinta
   de `anio_edicion_dif` (que compara contra el promedio de lectura del
   propio usuario, no contra su nacimiento). Reusa el sentinel `1910`
   corregido en la sección anterior.

Implementadas en `calcular_features_auxiliares`/`generar_candidatos_con_features`
(mismo patrón de lookup anidado por segmento que país/franja para las
dos primeras); 26 features en total.

### Resultado combinado: el más cercano de la sesión, pero no limpio

| | seed=42 | seed=7 | seed=123 | media ± desvío |
|---|---|---|---|---|
| 23 features (confirmado en Kaggle: 0.04831) | 0.10568 | 0.11054 | 0.11299 | 0.109735 ± 0.003719 |
| 26 features (+3 cruzadas) | 0.10626 | 0.11132 | 0.11275 | 0.110112 ± 0.003411 |

Positivo en seed=42 (+0.00058) y seed=7 (+0.00078), apenas negativo en
seed=123 (-0.00024) -- una diferencia muy por debajo del desvío entre
seeds (~0.0034) y mucho más chica que los fallos reales de país/franja
(que rondaban -0.001 a -0.0013). `feature_importances_` ubica a
`popularidad_genero_lector_candidato` como la más fuerte de las 3
(consistentemente top-10 del set completo), y a las otras dos como las
más débiles del trío.

### Ablation: sacar la más floja no ayudó (al revés)

Para intentar que seed=123 cruzara a positivo limpio, se probó sacar
`edad_lector_al_publicarse` (25 features). Resultado contrario al
esperado: seed=123 empeoró (-0.00084 vs -0.00024 con las 26 completas),
pese a que seed=42/seed=7 mejoraron levemente. Media total también peor
(0.109897 vs 0.110112). Mismo patrón que ya se vio con
`en_editorial_leida`/`n_libros_editorial_leidos`: la feature más floja
individualmente no es ruido puro, aporta algo real en conjunto. Se
revirtió el ablation, quedaron las 26 features completas.

### Decisión y confirmación en Kaggle

Con el resultado más cercano de la sesión al criterio estricto (y
mucho más cercano que país/franja), se decidió con el usuario confirmar
con una submission real -- mismo criterio que género macro/tamaño de
editorial, casos límite anteriores que sí resultaron mejoras reales.
**CONFIRMADO EN KAGGLE: 0.04855** -- nuevo récord del proyecto, +0.5%
sobre el récord anterior (0.04831). Tercera vez que un caso límite (que
no pasa la regla estricta de "positivo en los 3 seeds") se confirma
igual como mejora real -- refuerza que la regla estricta es útil como
guía pero demasiado conservadora para descartar candidatos sin probar,
sobre todo cuando el "fallo" es de una magnitud mucho menor que el
ruido entre seeds. `"ranker"` (26 features) pasa a ser el modelo de
referencia del proyecto.


## 4ª fuente de candidatos: libros de autores ya leídos — la mejora más grande de la sesión

### Objetivo

Implementar la recomendación #1 del análisis de generador de candidatos
(`experiments/modelo_actual.md`, sección "Recomendación: ¿cambiar de
paradigma?"): agregar una 4ª fuente de candidatos (no solo feature)
para libros de autores que el usuario ya leyó. El análisis midió que
28.6% de los targets de validación son de un autor ya leído, señal que
antes solo existía como *feature* (`en_autor_leido`/
`n_libros_autor_leidos`), incapaz de proponer un candidato nuevo por sí
sola -- solo podía activarse si ALS o popularidad ya habían traído el
libro por otro motivo.

### Implementación

Para cada autor que el usuario ya leyó, hasta `n_por_autor=20` libros
sin leer de ese autor, rankeados por el score de popularidad **global**
(no se refittea un score bayesiano por autor -- la mayoría tiene muy
pocas interacciones para un shrinkage propio confiable). 3 features
nuevas: `score_autor_candidato`/`rank_autor_candidato`/
`en_autor_candidato` (26 → 29), deliberadamente con nombres distintos
de `en_autor_leido`/`n_libros_autor_leidos` (que miden historial, sin
importar qué fuente trajo el candidato).

### Problema real de memoria, encontrado en la verificación

La primera versión no topeaba el total de candidatos por usuario de
esta fuente (solo el top-20 *por autor*). Usuarios que leyeron cientos
de autores distintos llegaron a **5.304 candidatos** de esta fuente
sola (media 653/usuario, contra ~419 antes) -- una corrida completa de
`scripts/evaluate_ranker.py` se cortó (`killed`, sin traceback) después
de 2 de 3 seeds. Se agregó un tope total de `n_por_fuente` (150)
candidatos por usuario para esta fuente, igual que las otras 3,
**priorizando los autores que más leyó el usuario** (no el orden
arbitrario del dict) en vez de cortar a ciegas. Con el tope: recall
0.473→0.445 (baja un poco, sigue muy por encima del 0.394 original) y
candidatos por usuario mucho más razonables (media 486, máximo 581).

Nota aparte: durante la verificación, corridas en *background* de este
pipeline más pesado se cortaron dos veces sin error visible (posible
límite de memoria del entorno de esta sesión, no del código -- ver
también el hallazgo similar del análisis de paradigma, sección
"agentes"). Correrlas en *foreground*, seed por seed, funcionó sin
problemas -- mismo patrón que ya se había usado antes en el proyecto
cuando el background resultó inestable (ver sección "Tuneo de LightGBM
sobre 23 features").

### Resultado: la mejora más grande y sólida de la sesión

| | seed=42 | seed=7 | seed=123 | media ± desvío |
|---|---|---|---|---|
| 26 features (confirmado en Kaggle: 0.04855) | 0.10568 | 0.11054 | 0.11299 | 0.109735 ± 0.003719 |
| 29 features + 4ª fuente (autor, topeada) | 0.11481 | 0.11776 | 0.11991 | 0.117495 ± 0.002562 |

**Positivo en los 3 seeds**, con una diferencia (+0.00913/+0.00722/
+0.00692) **5-10x más grande** que cualquier resultado de esta sesión
(los "casos límite" rondaban 0.0001-0.0013). Recall del set de
candidatos (`scripts/recall_candidatos.py`): 0.3941→0.4450 (+12.9%).

### Hallazgo adicional: la mejora viene de los candidatos, no de las features

Test pareado (`scripts/comparar_features_pareado.py`, seed=42, mismo
contexto/pool de candidatos con vs. sin las 3 features de tracking de
esta fuente): diferencia +0.0004, **0.44 sigma, no significativa**. El
ranker ya lograba aprovechar casi toda la mejora usando solo las
features *existentes* (`en_autor_leido`, `n_libros_autor_leidos`,
`n_interacciones_libro`, etc.) para puntuar los candidatos nuevos -- las
3 features explícitas de esta fuente no aportan mucho por sí solas,
aunque tampoco restan. Se mantienen de todos modos (documentan
explícitamente qué fuente propuso el candidato, sin costo).

Este resultado **confirma limpiamente la tesis central del análisis de
paradigma**: el margen real estaba en el recall del generador de
candidatos, no en agregar más features al reranking -- exactamente lo
opuesto al patrón de retornos decrecientes que se venía viendo con
features puramente informativas (país, franja, señales cruzadas).

### Confirmación en Kaggle

Se generó la submission (`ranker_...4a-fuente-autor...csv`) y se
confirmó: **0.05140 en Kaggle -- nuevo récord del proyecto, +5.9% sobre
el récord anterior** (0.04855). Es la segunda mejora más grande de todo
el proyecto (detrás de +15.3% de autor/año/género/recencia), y la
**mejor validada estadísticamente de las dos**: acá hubo un diagnóstico
de recall aparte (no solo NDCG ruidoso), CV positivo en los 3 seeds por
un margen muy por encima del ruido, y un test pareado que aisló el
mecanismo real (candidatos, no features) antes de gastar la submission.
`"ranker"` (29 features, 4 fuentes de candidatos) pasa a ser el modelo
de referencia del proyecto.


## n_por_fuente=150→500: recall sube fuerte, NDCG no acompaña

### Objetivo

Retomar el next step #2 del análisis de generador de candidatos: subir
`n_por_fuente` de 150 a 500 (cambio de una línea) sobre la base ya
confirmada de 29 features/4 fuentes (0.05140 en Kaggle).

### Resultado: recall +30%, NDCG +0.7% -- no se justifica

Medido con `scripts/recall_candidatos.py` (seed=42, la herramienta
pensada justo para decidir esto sin correr el pipeline completo):

| | recall | candidatos/usuario (media) | NDCG@20 | eficiencia (NDCG/recall) |
|---|---|---|---|---|
| `n_por_fuente=150` | 0.4450 | 486 | 0.114810 | 0.258 |
| `n_por_fuente=500` | 0.5784 | 1496 (hasta 1864) | 0.115601 | 0.200 |

El recall subió muchísimo (+30%), pero la eficiencia de ranking bajó
de 0.258 a 0.20 -- con ~3x más candidatos por usuario, la tarea de
rankear se vuelve más difícil (más distractores), tal como anticipaba
el análisis original ("no se va a sostener igual con más candidatos").
Acá el efecto fue más fuerte de lo esperado: prácticamente cancela toda
la ganancia de recall, dejando un NDCG casi idéntico (+0.7% en un solo
seed, dentro del ruido). Además, el costo de cómputo casi se duplicó
(contexto: 880s vs ~450s; dataset de entrenamiento: ~12M filas vs ~3.9M).

**No se corrió el CV completo de 3 seeds** -- el propio método diseñado
esta sesión ("medir recall primero, correr el pipeline completo solo
si vale la pena") funcionó como se esperaba: evitó gastar ~20 minutos
extra en una idea que ya se veía marginal con un solo seed. Se
descarta el cambio; `n_por_fuente` se mantiene en 150 en producción
(`scripts/recall_candidatos.py` revertido a su default).

### Reflexión: no toda ganancia de recall se traduce en NDCG

Este resultado matiza la lectura simple de "más recall = mejor" -- el
recall es un techo, no una garantía. Agregar candidatos ayuda si son
candidatos que el ranker puede aprender a distinguir bien del resto
(como pasó con la fuente de autor: candidatos con una señal fuerte y
específica -- "ya leíste a este autor"). Simplemente ampliar la
ventana de las fuentes existentes agrega, en cambio, sobre todo
*distractores* de las mismas fuentes que ya venían aportando poco en
ese rango (libros populares/de género que el usuario no leyó por buenas
razones), sin agregar una señal cualitativamente nueva.


## 5ª fuente de candidatos: similitud de resumen (contenido, no popularidad)

### Objetivo

Retomar la charla sobre el sesgo hacia popularidad de las 4 fuentes
actuales (ALS, popularidad global, popularidad por género, autor ya
leído -- todas sesgadas hacia libros bien conectados en mayor o menor
medida, incluso ALS por cómo se degradan sus embeddings con pocas
interacciones en una matriz 99.91% dispersa). Antes de implementar, se
midió con el usuario cuán importante es esto: los libros objetivo que
las 4 fuentes fallan en capturar son ~11x menos populares (mediana 21
interacciones en todo el dataset) que los que sí capturan (mediana
231) -- 28.7% de los faltantes tiene ≤5 interacciones (vs 4.3% de los
capturados). Confirma que vale la pena atacarlo.

### Implementación

`sim_resumen_historial` (similitud coseno TF-IDF, ya existente como
*feature*) nunca proponía candidatos nuevos por sí sola -- solo
puntuaba candidatos que ya habían llegado de otra fuente. Se agregó
`_generar_candidatos_por_resumen`: para cada usuario, el top-`n_por_fuente`
de **todo el catálogo con resumen** (~48.320 libros) más similar a su
perfil de lectura. Procesado en **lotes** (`TAMANO_LOTE_RESUMEN=500`)
para no materializar un producto denso usuarios×libros de una sola vez
-- precaución aprendida de los dos problemas de memoria reales de esta
sesión (autor sin tope, `n_por_fuente=500`). 3 features nuevas
(`score_resumen_candidato`/`rank_resumen_candidato`/`en_resumen_candidato`,
29→32), con nombres deliberadamente distintos de `sim_resumen_historial`.

### Anomalía de rendimiento (resuelta: no era un bug)

Corriendo el CV, seed=7 tardó ~2.9 horas contra los ~10 minutos
esperados. Se sospechó inicialmente paginación de memoria (el cálculo
por lotes de la similitud TF-IDF es la parte más pesada del pipeline),
pero la cuenta no cerraba (194MB por lote, muy por debajo de la memoria
libre de la máquina). **La causa real: la PC entró en reposo** durante
esa corrida (confirmado por el usuario) -- nada que ver con el código.
Reintentado con la máquina despierta: seed=123 tardó 603s, normal.
Lección: no asumir causas de rendimiento sin confirmar la causa real
-- la hipótesis de memoria era plausible pero incorrecta.

### Resultado: positivo pero más modesto que la fuente de autor

| | recall | candidatos/usuario | NDCG@20 (seed=42) |
|---|---|---|---|
| 4 fuentes (sin resumen) | 0.4450 | 486 | 0.114810 |
| 5 fuentes (+resumen) | 0.4559 | 620.5 | 0.117503 |

CV 3 seeds: 0.120547±0.002674 vs 0.117495±0.002562 de las 29 features
-- **positivo en los 3 seeds** (+0.00269/+0.00476/+0.00171, media
+2.6%). Test pareado (mismo patrón que autor): las 3 features de
tracking no aportan por sí solas (0.67 sigma) -- la mejora viene de los
candidatos nuevos, no de las features, consistente con lo que ya se
había visto.

Es una mejora más chica que la de autor (+7.1%), pero el mecanismo es
distinto y complementario: en vez de ampliar la cobertura hacia libros
de un autor conocido, amplía hacia libros temáticamente afines sin
importar su popularidad -- exactamente la dirección que sugería el
análisis de rareza de los targets faltantes.

### Confirmación en Kaggle: mejora real, pero difícil de ver en una sola muestra

Se generó la submission y se confirmó: **0.05181 en Kaggle -- nuevo
récord del proyecto** (+0.8% sobre 0.05140, +0.00041 absoluto). El
salto absoluto es del mismo orden que el error estándar de una sola
submission (~0.0065, calculado antes en esta sesión con la muestra de
832 usuarios de `ejemplo.csv`) -- no alcanza, por sí solo, para
confirmar con certeza estadística que esta ronda específica mejoró
Kaggle. Pero a diferencia del episodio de señales cruzadas (donde la
evidencia *local* ya era límite, "casi positivo en los 3 seeds"), acá
la evidencia local es sólida (positivo en los 3 seeds por márgenes
0.0017-0.0048, claramente por encima del ruido local calibrado, +
mecanismo confirmado con el test pareado). La lectura más razonable:
la mejora real (~+2.6% local) probablemente sigue ahí, pero esta
muestra puntual de Kaggle (una sola submission, ruidosa) la subestima
-- no es evidencia de que la mejora local haya sido ruido, solo de que
el instrumento de confirmación (una submission) tiene menos resolución
que el instrumento de medición local (CV de 3 seeds + test pareado).

`"ranker"` (32 features, 5 fuentes de candidatos) pasa a ser el modelo
de referencia del proyecto.

## 6ª fuente de candidatos: co-lectura ítem-ítem (kNN) — nuevo récord, límite pero consistente

### Objetivo

Siguiente punto de la agenda dejada al cierre de la sesión anterior en
`decisiones.md`: `score_coleido` (matriz de co-ocurrencia ítem-ítem,
`cooc = X.T @ X`) hoy solo puntúa candidatos que ya trajo otra fuente
-- nunca propone candidatos nuevos por sí sola. Sigue siendo una señal
colaborativa (mismo sesgo hacia libros con suficientes interacciones
que ALS/popularidad, aunque más suave), pero puede traer libros que ALS
no trae: dos libros pueden co-leerse mucho sin que ALS los recomiende
al mismo usuario.

### Implementación

El batch `co_scores_por_usuario` (`X_batch @ cooc`) que ya arma
`score_coleido` contiene, para cada usuario, el score de co-lectura
contra **todo el catálogo indexado por ALS** -- no hizo falta ningún
matmul nuevo. Se agregó, en `generar_candidatos_con_features`, tomar el
top-`n_por_fuente` de ese mismo cálculo (`heapq.nlargest`, para no
ordenar el dict completo de un usuario con mucho historial) como
candidatos nuevos. 3 features de tracking (`score_coleido_candidato`/
`rank_coleido_candidato`/`en_coleido_candidato`, 32→35), mismo patrón y
mismos nombres que autor/resumen -- distintas de `score_coleido`, que
sigue puntuando cualquier candidato sin importar su fuente.

### Resultado: recall sube fuerte, pero con la misma advertencia de `n_por_fuente=500`

| | recall | candidatos/usuario | NDCG@20 (seed=42) | eficiencia (NDCG/recall) |
|---|---|---|---|---|
| 5 fuentes (sin kNN) | 0.4559 | 620.5 | 0.117503 | 0.258 |
| 6 fuentes (+kNN) | 0.5115 | 694.9 | 0.118625 | 0.232 |

Recall +12.2%, pero la eficiencia de ranking baja 10% (menos que el
-22% de `n_por_fuente=500`, que sí se descartó sin correr el CV
completo) -- señal de que parte del recall extra no trae información
distinguible para el ranker, aunque no tanto como en aquel caso. Al no
ser un "casi no se movió" tan limpio como `n_por_fuente=500`, se decidió
correr igual el CV completo en vez de descartar solo con este número.

### CV 3 seeds: reproducido el baseline en la MISMA sesión para comparar en igualdad de condiciones

A diferencia de rondas anteriores (donde el baseline salía de una
corrida de una sesión previa), acá se reprodujo el baseline de 5
fuentes con `git stash` **en esta misma sesión**, mismo entorno, antes
de comparar semilla por semilla -- para no depender de que los números
de `bitacora.md` fueran exactamente reproducibles entre sesiones (lo
son: seed=42 dio 0.117503, idéntico al ya logueado, confirmando que el
split es 100% determinístico entre sesiones; solo el entrenamiento de
LightGBM tiene un ruido de punto flotante mínimo, ~0.0001-0.0006, ver
`comparar_features_pareado.py`).

Nota operativa: las primeras dos corridas de este baseline terminaron
`killed` a mitad de camino (una a los ~600s, justo después de terminar
seed=42; otra casi al arrancar) -- mismo síntoma que el episodio de la
5ª fuente en que la PC entró en reposo durante una corrida larga. Se
reintentó con la PC despierta y terminó sin problemas.

| seed | 5 fuentes (baseline) | 6 fuentes (+kNN) | diferencia |
|---|---|---|---|
| 42 | 0.117503 | 0.118625 | +0.001122 (+0.95%) |
| 7 | 0.122518 | 0.124146 | +0.001628 (+1.33%) |
| 123 | 0.121619 | 0.123179 | +0.001560 (+1.28%) |
| **media** | **0.120547 ± 0.002674** | **0.121983 ± 0.002949** | **+0.001436 (+1.19%)** |

**Positivo en los 3 seeds**, con magnitudes parecidas entre sí
(0.0011-0.0016, sin un seed que domine ni ninguno negativo) -- cumple
el criterio de "positivo en los 3 seeds individualmente" que el
usuario confirmó esta sesión como suficiente para justificar una
submission, aunque la mejora promedio (+1.19%) sigue sin superar el
desvío entre seeds (0.0029). Es la mejora local más chica de todas las
que se confirmaron así en esta sesión (género macro +1.80%, editorial
+0.92%, señales cruzadas ~0%/casi positivo) -- pero a diferencia de
señales cruzadas (mixto, un seed casi negativo) o país/franja (negativos
en 2 de 3), acá los tres seeds apuntan consistentemente en la misma
dirección y con magnitud similar.

### Confirmación en Kaggle: nuevo récord

Se generó la submission (`ranker_20260901-173000_6a-fuente-coleido-knn.csv`)
y se confirmó: **0.05262 en Kaggle -- nuevo récord del proyecto**
(+1.56% sobre 0.05181, +0.00081 absoluto). El salto absoluto es
mayor al de la ronda anterior (+0.00041 con resumen) pese a que la
mejora local fue más chica -- otro recordatorio de que la relación
entre el tamaño del efecto local y el salto puntual en una sola
submission de Kaggle es ruidosa en las dos direcciones, no solo
subestimando mejoras reales (como pasó con resumen) sino también
sobreestimándolas puntualmente.

`"ranker"` (35 features, 6 fuentes de candidatos) pasa a ser el modelo
de referencia del proyecto.

## Embeddings semánticos (sentence-transformers) para el perfil de contenido — descartado

### Objetivo

Curiosidad del usuario después de discutir TF-IDF vs. embeddings de un
LLM en abstracto: TF-IDF hace matching léxico exacto, así que dos
resúmenes que dicen lo mismo con palabras distintas ("asesinato en una
mansión inglesa" / "crimen en una casa señorial") quedan lejos en ese
espacio pese a ser semánticamente casi iguales. La hipótesis era que un
embedding semántico capturaría esa similitud, y que eso ayudaría
justo a la 5ª fuente de candidatos (similitud de resumen), que se
agregó para traer libros ~11x menos populares que las otras 4 fuentes
fallan en capturar -- el caso de uso donde el vocabulario exacto varía
más.

Se armó como experimento controlado, no como reemplazo directo: pedido
explícito del usuario de un enfoque "contenido y económico" -- un
modelo de `sentence-transformers` local (`paraphrase-multilingual-MiniLM-L12-v2`,
384 dimensiones, multilingüe/español, corre en CPU), sin API paga ni
dependencia de red en tiempo de inferencia, evitando el problema de
reproducibilidad de un proveedor externo que puede cambiar de modelo
entre corridas.

### Implementación

Función hermana de `_calcular_perfil_texto` (`_calcular_perfil_texto_embeddings`),
con la misma interfaz de salida (`tfidf_norm`/`fila_por_libro_texto`/
`perfil_usuario_norm`, embeddings envueltos en `sp.csr_matrix`) -- todo
lo que consume esas matrices (`_generar_candidatos_por_resumen`, el
cálculo de `sim_resumen_historial`) funcionó sin ningún cambio, porque
solo hacen matmul/producto interno sin asumir de dónde salió el vector.
Un parámetro `metodo_texto="tfidf"|"embeddings"` opt-in en
`calcular_features_auxiliares`/`preparar_pipeline`, default sin cambio
de comportamiento. Embeddings cacheados a disco (`data/cache/`,
gitignoreado) porque codificar ~48k resúmenes con un transformer tiene
costo real, a diferencia de fittear TF-IDF.

### Resultado: recall prácticamente igual, NDCG levemente peor -- no se justifica

Un solo seed (42) alcanzó para descartar (mismo criterio que
`n_por_fuente=500`: recall primero, CV completo solo si vale la pena):

| | TF-IDF | Embeddings |
|---|---|---|
| Recall del set de candidatos | 0.5115 | 0.5099 |
| NDCG@20 | 0.118625 | 0.116862 |
| Tiempo de armado del contexto | 805s | 2412s (incluye descarga del modelo + codificar ~48k resúmenes en CPU, ~18 min) |

El dato más importante es el recall: casi no cambió (0.5115→0.5099,
-0.3%), lo que significa que los embeddings traen candidatos casi
idénticos a los de TF-IDF -- contradice de entrada la hipótesis de que
iban a surgir candidatos *distintos* por captar sinónimos/parafraseo.

Test pareado por usuario (8.904 usuarios, mismo split que TF-IDF
porque el split no depende de `metodo_texto`):

```
diferencia media pareada (embeddings - tfidf): -0.001763
SE pareado: 0.001014   -> -1.74 sigma
bootstrap 95% CI: [-0.003725, +0.000247]
P(embeddings > tfidf) = 0.0375
usuarios donde cambia el NDCG: 15.0% (mejora 658, empeora 680)
```

No cruza el umbral estricto de 2 sigma, pero el intervalo de confianza
es casi enteramente negativo y en 96 de cada 100 remuestreos bootstrap
TF-IDF ganó -- evidencia débil pero consistentemente en contra, no a
favor. No cumple ni el criterio mínimo del proyecto para pensar en
correr el CV completo de 3 seeds (positivo en un seed), así que no se
gastó ese tiempo ni una submission de Kaggle.

### Reflexión: por qué probablemente no ayudó

Sin confirmar con más experimentos (queda como hipótesis, no como
hallazgo): un modelo de embeddings de propósito general sobre resúmenes
cortos parece capturar similitud de *tema/género* en un sentido amplio,
más difusa que el matching léxico específico (nombres propios,
escenarios, vocabulario puntual de la trama) que parece ser la señal
que efectivamente funciona en `sim_resumen_historial`. Si el "próximo
libro" de un usuario tiende a compartir vocabulario concreto con lo que
ya leyó (misma saga, mismo autor con estilo reconocible, mismo
subgénero con jerga propia) más que un tema general, TF-IDF explota
justo esa especificidad y un embedding semántico la difumina al
promediar hacia una representación más genérica.

Código revertido en su totalidad (`ranker.py`, `pyproject.toml`,
`.gitignore`, dependencia `sentence-transformers`) después del
resultado -- a diferencia de país/franja de nacimiento (que se dejaron
sin usar en el código "por si sirve con otro enfoque"), acá no quedó
nada en el repo. Si se retoma la idea en el futuro, un camino más
prometedor que "reemplazar TF-IDF" sería agregarlo como *feature
adicional* (no como reemplazo) o probar con un modelo de embeddings
más grande/afinado al dominio literario -- ninguno de los dos se probó
esta ronda.
