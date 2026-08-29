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
