# Estado del arte del proyecto (para explicar a colegas)

Resumen autocontenido, pensado para poder explicarle el proyecto a
alguien que no siguió la sesión a sesión — qué problema resuelve, qué
arquitectura usa, con qué hiperparámetros, cómo se valida y qué falta.
No repite el *por qué* de cada decisión (eso está en
`decisiones.md`/`bitacora.md`) ni el detalle feature por feature (eso
está en `features.md`) — acá es la foto completa, un nivel más arriba.

## El problema

Recomendar libros para una competencia de Kaggle evaluada con
**NDCG@20**: para cada lector hay que entregar un ranking de 20 libros
(el orden importa), y se puntúa según si el libro que ese lector
efectivamente leyó después aparece cerca del principio de esa lista.

Datos: `data/raw/data.db` (sqlite), tres tablas —
`interacciones(id_lector, id_libro, fecha, rating)`,
`lectores(id_lector, nombre, genero, vive_en, nacimiento)`,
`libros(id_libro, titulo, autor, genero, editorial, anio_edicion, isbn,
resumen, img_src)`. Volumen: 461.408 interacciones, 11.285 lectores
(10.673 con actividad), 128.743 libros en catálogo (48.137 con al menos
una interacción). La matriz usuario-libro tiene **99.91% de sparsity**.

## Récord actual

**0.05262 de NDCG@20 en Kaggle** (2026-09-01), con el modelo
`"ranker"` de `src/recsys/models/ranker.py`. Progresión completa de
todos los modelos probados, con el score real de Kaggle al lado del
local (siempre hay que mirar los dos, ver sección de validación):

| Modelo | NDCG@20 local | NDCG@20 Kaggle |
|---|---|---|
| Popularidad global (bayesiana) | 0.006620 | 0.01024 |
| Popularidad segmentada (género → franja → global) | 0.013719 | 0.01558 |
| ALS (factors=128, reg=0.1, rating crudo) | 0.094406 ± 0.001359 (CV 3 seeds) | 0.03864 |
| Ranker, 3 fuentes de candidatos (26 features) | 0.109735 ± 0.003719 | 0.04831 |
| Ranker, 4 fuentes (+autor ya leído, 29 features) | 0.117495 ± 0.002562 | 0.05140 |
| Ranker, 5 fuentes (+similitud de resumen, 32 features) | 0.120547 ± 0.002674 | 0.05181 |
| Ranker, 6 fuentes (+co-lectura ítem-ítem/kNN, 35 features) | 0.121983 ± 0.002949 | **0.05262 (récord actual)** |

## Arquitectura: candidatos + reranking

No es un solo modelo — son dos etapas, el patrón estándar de sistemas
de recomendación de producción ("*retrieval + ranking*"):

1. **Generación de candidatos**: seis fuentes independientes proponen,
   cada una, hasta 150 libros por usuario (`n_por_fuente=150`). Se unen
   sin duplicar — cada candidato queda marcado con qué fuente(s) lo
   propusieron.
2. **Reranking**: un `LGBMRanker` (LightGBM, gradient boosted trees)
   toma la unión de esos candidatos, cada uno con 35 features, y
   aprende a reordenarlos mejor de lo que cualquier fuente por sí sola
   podría.

Ninguna fuente por sí sola alcanza; en vez de combinarlas a mano (ej.
promediar scores), se entrena un modelo supervisado para que aprenda
cómo pesarlas. El filtrado colaborativo (ALS) sigue siendo la columna
vertebral de los candidatos — LightGBM es la capa de reranking encima,
no un reemplazo.

### Las 6 fuentes de candidatos

| # | Fuente | Qué propone | Código |
|---|---|---|---|
| 1 | **ALS** (filtrado colaborativo) | Top-150 según `implicit.recommend`, con el rating explícito (1-10) usado directamente como confianza | `models/als.py` |
| 2 | **Popularidad global** | Top-150 del ranking bayesiano global | `models/popularity.py` |
| 3 | **Popularidad por género preferido** | Top-150 dentro del género literario que más leyó el usuario | `models/popularity_segmentada.py` |
| 4 | **Autores ya leídos** | Hasta 20 libros sin leer por cada autor que el usuario ya leyó, rankeados por popularidad global | `ranker.generar_candidatos_con_features` |
| 5 | **Similitud de resumen** | Top-150 de *todo* el catálogo con resumen (~48k libros) más parecido al perfil TF-IDF de lectura del usuario — no depende de cuánta gente más leyó el libro | `ranker._generar_candidatos_por_resumen` |
| 6 | **Co-lectura ítem-ítem (kNN)** | Top-150 por score de co-lectura contra el historial del usuario (reusa la matriz de co-ocurrencia `cooc` ya calculada para la feature `score_coleido`) — sigue siendo colaborativo, pero trae candidatos que ALS no trae | `ranker.generar_candidatos_con_features` |

Total de candidatos por usuario: unión de las 6 fuentes, media ~695
(hasta ~820 para usuarios pesados).

## Modelos e hiperparámetros

### ALS (`models/als.py`)

`implicit.als.AlternatingLeastSquares` sobre la matriz sparse
usuario×libro, con el rating explícito (1-10) usado directamente como
confianza (no binarizado).

| Hiperparámetro | Valor |
|---|---|
| `factors` | 128 |
| `regularization` | 0.1 |
| `iterations` | 20 |
| `alpha` (peso `1+alpha*rating`) | `None` → confianza = rating crudo |
| `seed` | 42 |

Esta config es la que **mejor score dio en Kaggle real** (0.03864), no
la que mejor NDCG local dio: una búsqueda con `optuna` (30 trials)
encontró `factors=256, regularization=0.128, alpha=4.718` que mejoraba
el NDCG local +11.5% pero empeoraba Kaggle a 0.03341 — sobreajuste al
split usado en el sweep. Lección que quedó como norma del proyecto:
nunca confiar en una mejora de un solo split/seed sin confirmar con
validación cruzada, y desconfiar de mejoras grandes en un solo sweep.

### Popularidad (`models/popularity.py`)

Score bayesiano con shrinkage:
`score = (n/(n+C))·avg_rating + (C/(C+n))·m`, con `C = n.mean()`
(calculado sobre los datos recibidos, nunca hardcodeado) y `m` = rating
promedio global. Evita que un libro con 2 interacciones a 10 le gane a
uno con 500 interacciones a 8.5.

### Popularidad segmentada (`models/popularity_segmentada.py`)

Mismo score bayesiano, pero calculado sobre subconjuntos: género
literario preferido del usuario (52 categorías granulares
normalizadas, o 10 "macro-géneros" de dominio para las features del
ranker), género declarado del lector (Mujer/Hombre/desconocido), país,
franja de nacimiento (país y franja se probaron como features del
ranker y se descartaron — ver `decisiones.md`).

### LightGBM (`LGBMRanker`, dentro de `models/ranker.py`)

| Hiperparámetro | Valor |
|---|---|
| `objective` | `lambdarank` |
| `num_leaves` | 31 |
| `learning_rate` | 0.05 |
| `n_estimators` | 200 |
| `random_state` | 42 |

Hiperparámetros **conservadores a propósito**, sin tuneo agresivo: se
probó tunear con `optuna` tres veces (sobre bases de features
distintas) y las tres veces la mejora fue menor que el ruido entre
seeds — nunca se adoptó. `lambdarank` optimiza una aproximación
diferenciable del orden dentro de cada grupo de candidatos de un mismo
usuario (no accuracy/logloss por fila), que es justo lo que necesita
NDCG.

## Features (35 en total, catálogo completo en `features.md`)

Agrupadas por qué miden — cada fuente de candidatos aporta 3 columnas
(`score_*`/`rank_*`/`en_*`, el score y la posición que le dio esa
fuente, y si la propuso), y encima hay features "cruzadas" que no
vienen de ninguna fuente en particular:

- **Score/rank/en de cada una de las 6 fuentes** (18 features).
- **Volumen bruto**: cuántas interacciones tiene el libro y el usuario.
- **Autor**: ¿el usuario ya leyó a este autor?, ¿cuántos libros suyos?
- **Editorial**: historial del usuario con la editorial + tamaño del
  catálogo de la editorial (propiedad del libro, no del usuario).
- **Año de edición**: diferencia contra el año promedio que lee el
  usuario.
- **Diversidad de género**: cuántos géneros distintos leyó el usuario.
- **Recencia**: días desde la última interacción del usuario.
- **Co-lectura ítem-ítem** (`score_coleido`): cuánta gente que leyó lo
  mismo que este usuario también leyó el candidato (matriz de
  co-ocurrencia `X.T @ X`, sparse, ~3s para todo el catálogo).
- **Resumen/texto**: similitud coseno TF-IDF entre el resumen del
  candidato y el perfil de lectura del usuario.
- **Macro-género**: popularidad del candidato pooleada a 10 familias de
  género (en vez de las 52 granulares) + qué tan seguido lee el usuario
  ese macro-género.
- **Señales cruzadas lector↔libro**: popularidad segmentada por género
  *declarado* del lector, afinidad de la cohorte del mismo género por
  macro-género, edad del lector cuando se publicó el candidato.

Todas se calculan **solo con el tramo de entrenamiento** (nunca con
datos que el ranker vea como etiqueta), excepto la metadata estática
del libro (autor/editorial/año/resumen), que no depende del split.

## Cómo se valida localmente

### Split de tres niveles (no un split simple)

Como el ranker usa scores de ALS/popularidad como *features*, esos
scores no pueden salir de datos que el ranker vea como etiqueta — si no,
memoriza en vez de aprender a combinar señales. Por eso hay tres
tramos, no dos:

1. **`train_candidatos`**: fitea ALS + popularidad + popularidad por
   género (las fuentes de la etapa 1) y calcula todas las features
   auxiliares.
2. **`train_ranker`**: etiquetas conocidas (la interacción más reciente
   de cada usuario dentro de este tramo) para entrenar el `LGBMRanker`.
3. **`test_final`**: hold-out aislado para medir NDCG@20 del pipeline
   completo (solo existe en evaluación local, no en producción).

Cada tramo sale de un **split leave-one-out temporal** (`split_train_val`
en `data.py`): se retiene la interacción *más reciente por fecha* de
cada usuario, no una muestra aleatoria — un split aleatorio filtra
información del futuro hacia train y sobreestima el NDCG local
fuertemente (medido: 0.260 con split aleatorio vs. 0.123 con el
corregido, mismo modelo).

### Validación cruzada sobre 3 seeds, no un split único

Todo el pipeline del ranker se evalúa con `scripts/evaluate_ranker.py`
sobre 3 seeds (42, 7, 123), mirando media *y* desvío — nunca un solo
número. Motivo real: un sweep de ALS que solo miró un split mejoró el
NDCG local +11.5% pero empeoró Kaggle -13.5% (sobreajuste al ruido de
ese split en particular).

### Métrica de diagnóstico: recall del set de candidatos

`scripts/recall_candidatos.py` mide qué fracción de los libros objetivo
de validación está *presente* entre los candidatos generados — el
techo absoluto de NDCG que puede lograr el reranking, sin importar qué
tan bueno sea LightGBM. Permite decidir sobre la etapa 1 (candidatos)
en minutos en vez de correr el CV completo (~20-25 min) para cada idea.
Regla aprendida: **no toda ganancia de recall se traduce en NDCG** — si
los candidatos nuevos no traen señal distinguible (solo más volumen),
el ranking se vuelve más difícil y el NDCG puede no acompañar (pasó al
subir `n_por_fuente` de 150 a 500: +30% recall, +0.7% NDCG).

### Test pareado por usuario (el criterio con más poder estadístico)

`scripts/comparar_features_pareado.py` compara dos configuraciones
sobre el **mismo contexto/seed** (mismo split, mismo fit de ALS) y mide
la diferencia de NDCG **por usuario**, no promedios independientes —
~5x más poder estadístico que comparar contra el desvío entre 3 seeds.
Se adoptó después de encontrar que varias mejoras que parecían
"confirmadas" (positivas en los 3 seeds, o incluso confirmadas en
Kaggle) eran estadísticamente indistinguibles de ruido con este test.

### La brecha local-vs-Kaggle

El número local **sobreestima** bastante el de Kaggle en términos
absolutos — se investigó y es principalmente varianza de muestra chica
en Kaggle (~832 usuarios de referencia, métrica donde la mayoría de
usuarios da NDCG=0) más una diferencia real de población: los usuarios
de `ejemplo.csv` tienen mucha más actividad (mediana 95 interacciones)
que el promedio de la evaluación local (mediana 9). Conclusión práctica
del proyecto: **comparar dentro del proyecto** (ranker vs. ALS solo, o
una versión de features vs. otra) es mucho más confiable que comparar
los valores absolutos entre sí.

### Criterio para decidir si gastar una submission de Kaggle

Cada submission es un recurso limitado (Kaggle limita cuántas por día).
El criterio vigente, validado explícitamente por el usuario tras
acertar en varios casos esta sesión: **si una mejora es positiva en los
3 seeds individualmente** (aunque no supere el desvío entre seeds),
vale la pena confirmarla con una submission real en vez de descartarla
solo por no pasar el umbral estricto. Si además pasa el test pareado,
mejor todavía — pero no es requisito excluyente.

## Qué se probó y no funcionó (para no repetirlo)

- **Tunear hiperparámetros de LightGBM** con `optuna`: probado 3 veces
  sobre bases de features distintas, las 3 veces la mejora fue menor
  que el ruido entre seeds. No se adopta más.
- **País del usuario (`vive_en`) y franja de nacimiento** como features
  del ranker: ambas empeoraron en 2 de 3 seeds. Quedó código de soporte
  sin usar en `popularity_segmentada.py` por si sirve con otro enfoque.
- **Subir `n_por_fuente` de 150 a 500**: +30% de recall de candidatos,
  pero el NDCG casi no se movió (+0.7%, dentro del ruido) y el costo de
  cómputo casi se duplicó. Descartado.
- **Modelos secuenciales (SASRec/GRU4Rec)**: descartados por los datos,
  no por costo — 67.5% de los gaps entre interacciones consecutivas son
  0 días y 45.7% de los usuarios tiene más de una interacción en su
  fecha máxima, así que el "orden" dentro de un día es en gran parte
  arbitrario, justo lo que un modelo secuencial necesitaría aprender.
- **LightFM / dos torres / factorization machines**: reemplazarían a
  ALS (que no es el cuello de botella) para meter metadata en el
  embedding — esa metadata ya está en las features del ranker, donde
  LightGBM la usa con más libertad. Alto costo de implementación, poca
  ganancia esperada.

## Próximos pasos (agenda completa en `decisiones.md`)

El hallazgo central de las últimas sesiones: **el cuello de botella es
el generador de candidatos, no el reranking ni el modelo colaborativo**.
Atacar el recall de candidatos (agregar fuentes nuevas) dio las tres
mejoras más grandes y mejor validadas del proyecto (autor ya leído:
+5.9% en Kaggle; similitud de resumen: +0.8%; co-lectura ítem-ítem/kNN:
+1.56%), mientras que agregar más features sobre el mismo set de
candidatos viene con retornos decrecientes. Pero la señal se está
agotando: cada fuente nueva sube el recall pero baja un poco la
eficiencia de ranking (NDCG/recall) — la de kNN fue la más chica de las
tres, con la caída de eficiencia más marcada (-10%, vs. -22% del caso
ya descartado de `n_por_fuente=500`). Ideas pendientes, en orden de
prioridad:

1. Adaptar `scripts/comparar_features_pareado.py` (o un script nuevo)
   para comparar dos *generadores de candidatos* completos, no solo
   subconjuntos de `FEATURES` sobre el mismo contexto — hoy no hay
   forma barata de aislar si una fuente nueva de candidatos ayuda antes
   de correr el CV completo de 3 seeds (~20-25 min).
2. Features ponderadas por recencia (autor/editorial/co-lectura
   pesando más lo reciente, no todo el historial parejo).
3. Revisar si conviene refitear la etapa 1 (ALS/popularidad) sobre
   todos los datos después de entrenar el ranker, en vez de sobre
   `train_candidatos` como hoy en `submit.py`.

## Dónde mirar para más detalle

- **`experiments/features.md`**: catálogo completo, feature por
  feature, con qué mide y qué pasa si falta.
- **`experiments/decisiones.md`**: qué se decidió en cada punto,
  qué se conserva y qué quedó pendiente de revisar.
- **`experiments/bitacora.md`**: el razonamiento completo de cada
  decisión, con los números de cada experimento.
- **`experiments/log.csv`**: historial tabular de cada corrida (NDCG
  local y de Kaggle) para comparar entre sí.
- **`experiments/modelo_actual.md`**: el mismo modelo descrito con más
  detalle técnico, más el análisis dedicado de "¿cambiar de paradigma?"
  con las mediciones que lo respaldan.
