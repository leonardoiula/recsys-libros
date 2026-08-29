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
