# Estado del arte — recsys-libros

Punto de entrada para retomar el proyecto: el modelo actual, cómo se valida, qué se probó
y no funcionó, y el problema abierto. El razonamiento completo ronda por ronda está
**congelado** en `experiments/legacy/` (ver `experiments/legacy/README.md`); no hace falta
leerlo para retomar contexto.

**Récord: 0.06753 de NDCG@20 en Kaggle** (2026-09-10, ventana rodante multi-corte `n_cortes=5`).

---

## El problema

Competencia de Kaggle: recomendar libros, evaluada con **NDCG@20**. Para cada lector se
entrega un ranking de 20 libros (el orden importa) y se puntúa según dónde cae el libro
que ese lector efectivamente leyó después.

Datos (`data/raw/data.db`): 461.408 interacciones, 10.673 lectores con actividad, 128.743
libros (48.137 con ≥1 interacción). Matriz usuario×libro **99.91% vacía**. Rating explícito
1–10. Metadata de libro: autor, editorial, año, género, resumen.

El set de evaluación de Kaggle (`data/raw/ejemplo.csv`) son **~832 usuarios sesgados a
alta actividad** (mediana ~95 interacciones vs ~9 en la evaluación local).

---

## El modelo actual (`--model ranker`, `src/recsys/models/ranker.py` + `submit.py`)

Dos etapas: **6 fuentes de candidatos → `LGBMRanker`**.

### Etapa 1: 6 fuentes de candidatos

Cada fuente propone hasta `n_por_fuente=150` libros sin leer por usuario, **salvo autor**
(`n_por_fuente_autor=500`). Unión deduplicada: media **~780** candidatos/usuario (hasta
~1140 para heavy users). Cada candidato queda marcado con qué fuente(s) lo propusieron.

| # | Fuente | Qué propone |
|---|---|---|
| 1 | **ALS** | top-150 de `implicit` (filtrado colaborativo) |
| 2 | **Popularidad global** | top-150 del ranking bayesiano global |
| 3 | **Popularidad por género preferido** | top-150 dentro del género que más leyó el usuario |
| 4 | **Autores ya leídos** | hasta 20 libros por autor que el usuario ya leyó (por popularidad global), hasta **500** en total por usuario |
| 5 | **Similitud de resumen** | top-150 del catálogo con resumen más parecido al perfil TF-IDF del usuario — no depende de popularidad |
| 6 | **Co-lectura ítem-ítem (kNN)** | top-150 por score de co-lectura contra el historial (matriz `X.T @ X`) |

Limitación: las fuentes 1/5/6 solo alcanzan usuarios con fila en la matriz de ALS; sin
historial → fallback a popularidad.

### ALS (`models/als.py`)

`implicit.als.AlternatingLeastSquares`, `factors=128, regularization=0.1, iterations=20`,
`seed=42`. Confianza = **rating crudo** (`alpha=None`; se probó `1+alpha*rating` tuneado y
empeoró en Kaggle).

**BM25 en la matriz** (`fit_als(..., bm25=(10.0, 0.75))`, constante `BM25_ALS` en
`ranker.py`): `implicit.nearest_neighbours.bm25_weight` se aplica **solo a la copia que va
a `modelo.fit()`** — la matriz que devuelve `fit_als` sigue con el rating crudo (así el
filtro de ya-leídos y la co-ocurrencia ítem-ítem no cambian). Baja el peso de los power
users (11,5% de usuarios = 64% de la señal) y de los libros muy leídos en la factorización.

### Etapa 2: LightGBM (`models/ranker.py`, `fit_ranker`)

`LGBMRanker`, `objective="lambdarank"`, `num_leaves=31, learning_rate=0.05,
n_estimators=200`, `random_state=42`. Hiperparámetros **conservadores a propósito**:
optuna probado 3 veces (bases de features distintas), las 3 dentro del ruido.

**Entrenamiento con ventana rodante multi-corte** (`N_CORTES_RANKER` en `ranker.py`,
`N_CORTES_RANKER_SUBMISSION = 5` en `submit.py` — estrategia 1 de `estrategias.md`, récord
2026-09-10): además del ejemplo `(historial → x_{m-1})` por usuario, se agregan los cortes
`x_{m-2}…x_{m-5}`. El corte `j` predice `x_{m-j}` con **su propia etapa 1**
(ALS/popularidad/género/co-lectura/TF-IDF) fiteada solo sobre `x_1…x_{m-1-j}` y sus features
sobre ese mismo historial — cada corte es un next-item genuino, sin leakage, **1 positivo
por grupo**. Los usuarios con poco historial se caen solos de los cortes profundos.
`test_final` y el recall del set de candidatos no cambian.

Progresión CV 3 seeds (positivo en los 3 en cada salto) / Kaggle: `n_cortes=1` 0.134117 →
`=3` **0.136561** (Kaggle 0.06316 → **0.06667**) → `=5` **0.137854** (Kaggle → **0.06753**).
`=7` **0.137783** — **plano** (media baja un pelo, positivo solo en 1/3): el CV plateaua en
profundidad 5, es la config de producción. Sin regresión por bucket de actividad — de hecho
los buckets casuales (2-4, 5-9) *mejoran* a más profundidad (más diversidad de
profundidad-de-historial en el entrenamiento → mejor generalización a usuarios de historial
corto). `n_interacciones_usuario` sube en importancia.

Distinto del `n_val_ranker` fallido (ver más abajo): aquel metía N positivos en un grupo y
compartía una etapa 1. `N_CORTES_RANKER` queda en **1** de default (dev rápido: un contexto
`n_cortes=5` pesa ~11 GB y tarda ~34 min/seed); `submit.py` y `evaluate_ranker.py` usan 5.
La unión de cortes se arma volcando cada `X_j` a un `.npy` temporal y leyéndolos con `mmap`
a un array `float32` preasignado (`_ensamblar_dataset_ventana_rodante`) — pico ~1× el tamaño
de la unión (~10 GB medido para `n_cortes=7`, ~16 GB de sistema libre), no ~2× como un
`pd.concat` de todas las particiones (que hacía OOM). `RANKER_LOG_RSS=1` imprime el RSS en
cada corte.

### 39 features

- **Score / rank / en de cada una de las 6 fuentes** (18).
- **Volumen**: interacciones del libro y del usuario (2).
- **Autor / editorial**: ¿ya leyó a este autor/editorial?, cuántos libros; + tamaño del
  catálogo de la editorial (propiedad del libro) (5).
- **Año**: diferencia contra el año promedio que lee el usuario (1).
- **Diversidad / recencia del usuario**: géneros distintos leídos; días desde la última
  interacción (2).
- **Co-lectura / resumen**: `score_coleido` (co-ocurrencia); `sim_resumen_historial`
  (TF-IDF coseno contra el perfil) (2).
- **Macro-género**: popularidad del candidato pooleada a 10 familias de dominio + qué tan
  seguido lee el usuario ese macro-género (2).
- **Señales cruzadas lector↔libro**: popularidad segmentada por género *declarado* del
  lector, afinidad de la cohorte por macro-género, edad del lector al publicarse el libro (3).
- **Recencia-ponderadas** (4): variantes de autor/editorial/co-lectura/resumen que pesan
  más lo que el usuario leyó hace poco (`peso = 1/log2(rank+2)`).

Catálogo detallado: `experiments/features.md`.

### refit de etapa 1 (`submit.py`)

Para la submission real, ALS/popularidad/género/features auxiliares se **entrenan sobre
`train_candidatos`** para entrenar el ranker (evita leakage de etiqueta), pero se
**REFITEAN sobre `interacciones` completo** antes de generar los candidatos finales (usa la
última interacción de cada usuario como señal, no solo para filtrar). La generación de
candidatos va **por lotes de usuarios** (`TAMANO_LOTE_USUARIOS`, evita `ArrayMemoryError`).

---

## Récord y progresión

| Modelo | NDCG@20 local | NDCG@20 Kaggle |
|---|---|---|
| Popularidad global (bayesiana) | 0.006620 | 0.01024 |
| Popularidad segmentada (género → franja → global) | 0.013719 | 0.01558 |
| ALS (factors=128, reg=0.1, rating crudo) | 0.094406 ± 0.00136 (CV 3 seeds) | 0.03864 |
| Ranker, 3 fuentes (26 features) | 0.109735 ± 0.00372 | 0.04831 |
| Ranker, 4 fuentes (+autor ya leído, 29 feat) | 0.117495 ± 0.00256 | 0.05140 |
| Ranker, 5 fuentes (+similitud de resumen, 32 feat) | 0.120547 ± 0.00267 | 0.05181 |
| Ranker, 6 fuentes (+co-lectura kNN, 35 feat) | 0.121983 ± 0.00295 | 0.05262 |
| Ranker, +recencia (4 feat) + refit de etapa 1 (39 feat) | 0.143336 ± 0.00290 (a `n_por_fuente=75`) | 0.06149 |
| Ranker, +BM25 en la matriz de ALS (`K1=10, B=0.75`) | 0.132313 ± 0.00219 (`n_por_fuente=150`) | 0.06182 |
| Ranker, +presupuesto de autor (`n_por_fuente_autor=500`) | 0.134117 ± 0.00096 | 0.06316 |
| Ranker, +ventana rodante multi-corte (`n_cortes=3`) | 0.136561 ± 0.00178 | 0.06667 |
| Ranker, ventana rodante `n_cortes=5` | 0.137854 ± 0.00148 | **0.06753** |

Nota: los NDCG locales solo son comparables **dentro** del mismo `n_por_fuente` (la fila de
recencia/refit se midió a 75 por memoria; las dos siguientes a 150).

`experiments/log.csv` tiene una fila por corrida con la nota completa.

---

## Cómo se valida

- **Split leave-one-out TEMPORAL** (`split_train_val`, `n_val=1`): se retiene la
  interacción *más reciente por fecha* de cada usuario. Un split aleatorio filtra futuro y
  sobreestima ~2× (0.260 vs 0.123, mismo modelo).
- **Split de 3 niveles para el ranker**: `train_candidatos` (fitea señales de etapa 1) /
  `train_ranker` (etiquetas para el `LGBMRanker`) / `test_final` (hold-out para NDCG@20).
  El tercer tramo solo existe en evaluación local.
- **CV sobre 3 seeds (42, 7, 123)** — media *y* desvío, nunca un solo split
  (`scripts/evaluate_ranker.py`). Un sweep de ALS sobre un split único mejoró el NDCG local
  +11,5% y empeoró Kaggle −13,5%.
- **Test pareado por usuario** (`scripts/comparar_features_pareado.py` /
  `comparar_generadores_pareado.py`): compara dos configs sobre el **mismo contexto/seed** y
  mide la diferencia de NDCG por usuario. **~5× más poder** que el desvío entre 3 seeds — es
  el gatekeeper real. Varias mejoras "positivas en los 3 seeds" resultaron ruido con este test.
- **`recall_de_candidatos`** (`scripts/recall_candidatos.py`): fracción de objetivos
  presentes entre los candidatos = techo duro del reranker (**~0.535** hoy). Regla: **no
  toda ganancia de recall se traduce en NDCG** — candidatos sin señal distinguible solo
  hacen más difícil el ranking (pasó con `n_por_fuente=500`).
- **Kaggle**: ~832 usuarios, la mayoría con NDCG=0; SE ≈ 0.0065 → una sola submission no
  distingue configs que difieren <~0.01. Confiar en el CV local para lo fino; usar Kaggle
  para confirmar dirección. Criterio para gastar una submission: positivo en los 3 seeds
  (aunque no supere el desvío), mejor si además pasa el test pareado.
- **NDCG ponderado por la actividad de `ejemplo.csv`** + **desglose por bucket de
  actividad** (`evaluate_ranker.py`): diagnóstico de generalización ("¿ayuda a los heavy
  users sin dañar a los casuales?"), no cambia decisiones. Reponderar no cambia el signo de
  ninguna comparación.

---

## Qué se probó y NO funcionó (para no repetirlo)

- **Tunear LightGBM con optuna** — 3 veces, siempre dentro del ruido entre seeds.
- **ALS tuneado con optuna** (`factors=256, reg=0.128, alpha=4.718`) — +11,5% local,
  **peor** en Kaggle (0.03341 vs 0.03864). Sobreajuste a un split.
- **País (`vive_en`) y franja de nacimiento** como features — empeoran en 2/3 seeds.
- **`n_por_fuente=500` global** (todas las fuentes) — +30% recall, NDCG plano, 2× cómputo.
- **`n_por_fuente_autor > 500`** — `800` dio +0,84 σ incremental sobre `500` (ruido), +67%
  candidatos/usuario.
- **Presupuesto propio para resumen / co-lectura** (`n_por_fuente_resumen/coleido`) —
  co-lectura sube el recall del set combinado (0.5152→0.5804) pero el NDCG no acompaña
  (`nfc=500` peor que `nfc=300`), eficiencia de ranking −10,7%. No tienen la patología de
  asignación de autor ni candidatos de alta precisión. Revertido; queda
  `scripts/probe_presupuesto_fuentes.py`.
- **Features de corroboración entre fuentes** (`n_fuentes_candidato` = conteo de fuentes
  que proponen el candidato; `rank_min_candidato`) — el diagnóstico mostró correlación
  fuerte (fuentes coincidentes: 1,31 fuera del top-20 vs 2,63 dentro) pero el test pareado
  dio **negativo** (`n_fuentes_candidato` −1,42 σ). La señal ya está en los `rank_*`/`score_*`
  individuales; el conteo agregado es redundante y proxy ruidoso de popularidad.
- **7ª fuente por editorial ya leída** (mirror de autor) — 50,6% de los targets son de
  editorial ya leída, pero el recall casi no se movió: catálogos de editorial dispersos,
  sus libros populares ya los traían ALS/popularidad. Candidatos redundantes.
- **7ª fuente por similitud usuario-usuario** (kNN de usuarios) — CV 2/3 seeds positivo,
  **regresión en Kaggle** (0.06017 vs 0.06149). Revertida.
- **Embeddings semánticos (`sentence-transformers`) vs TF-IDF** para el perfil de
  contenido — recall casi igual (0.5115→0.5099), NDCG peor (−1,74 σ pareado). Casi no
  traen candidatos distintos.
- **Modelos secuenciales (SASRec / GRU4Rec)** — descartados por los datos: 67,5% de los
  gaps entre interacciones consecutivas son 0 días, el "orden" intradía es arbitrario.
- **LightFM / dos torres / factorization machines** — reemplazarían a ALS (que no es el
  cuello de botella) para meter metadata en el embedding; esa metadata ya está en las
  features del ranker.
- **Rutear usuarios livianos a popularidad por género** (`als.recomendar_hibrido`) — ALS
  le gana a género en **todos** los buckets de actividad, incluso con 1 interacción.
- **Supervisión más densa del reranker, versión ingenua** (`n_val_ranker=3`: los 3 libros
  más recientes de cada usuario como positivos **en un solo grupo**, con **una sola etapa 1**
  compartida) — paired test seed=42 **catastrófico, −10,8 σ** (NDCG 0.133 → 0.113). Dos
  causas: (a) subir `n_val_ranker` también achica `train_candidatos`, degradando la etapa 1
  (recall 0.535 → 0.499); (b) lambdarank con 3 positivos "del pasado reciente" desdibuja el
  objetivo — `test_final` es EL libro siguiente, no "cualquiera de los últimos 3".
  Revertido. **La versión bien hecha SÍ funcionó** (ventana rodante multi-corte, récord
  0.06753 — ver "El modelo actual · Etapa 2"): 1 positivo por grupo y **una etapa 1 propia
  por corte** que solo achica su propio historial. La lección del fallo: no es que "más
  supervisión" esté mal, es que hay que darle a cada ejemplo su estado de etapa 1 correcto.
- **Filtrar los ejemplos de entrenamiento del reranker por historial mínimo**
  (`min_hist` sobre las interacciones usables del usuario en `train_candidatos` — el ~20%
  de los ejemplos vienen de usuarios con ≤3 libros de historial, features casi ruido) —
  paired test seed=42 **monótono y negativo**: `min_hist=3` −1,99 σ, `=5` −3,72 σ, `=10`
  −6,19 σ, `=20` −11,9 σ. El daño se concentra en los usuarios FINOS de test (~14% del
  set) y en los objetivos populares. Como el `LGBMRanker` no usa el ID de usuario, esos
  ejemplos "ruidosos" son la única señal sobre cómo rankear para un usuario del que se
  sabe poco; sacarlos = sobreajuste a patrones de usuario rico, peor generalización.
  Misma lección que `recomendar_hibrido`. Revertido; queda
  `scripts/diagnostico_historial_ranker.py`.
- **Features relativas dentro del usuario / cascade** (`score_als_pct_usuario`,
  `score_coleido_reciente_pct_usuario` = percentil del score entre los candidatos del
  mismo usuario; `n_candidatos_usuario` = tamaño del campo) — el paso barato del ángulo
  "cascade / re-rank en dos pasos". Paired test seed=42: **todas juntas −0,09 σ (plano
  total)**, cada una sola entre −1,44 σ y +0,26 σ (ruido). Sin patrón por decil de
  popularidad. Cierra también el cascade completo: si las features relativas/listwise que
  un cascade explotaría no dan señal en el modelo de una etapa, dos `LGBMRanker`s no lo
  cambian. LightGBM no está perdiéndose esta clase de señal. Revertido.
- **Metadata de serie/saga derivada de los títulos** (`scripts/eda_series.py`) — falla por
  **cobertura**, no por mecanismo: solo el 2,0% de los libros con interacción tiene un
  patrón de serie limpio, y de los 8904 test targets solo el **0,8%** es "el siguiente de
  una saga que el usuario viene leyendo". Techo de ganancia ~+0,0005 en Kaggle. No se
  implementó el parser.
- **Seed-bag del `LGBMRanker`** (`N_BAG_RANKER`, `n_bag` en `fit_ranker`, `_RankerBag` —
  N modelos con `random_state`/`subsample`/`colsample` distintos, promedio de scores) —
  paired test seed=42 +1,77 σ, CV 3 seeds **+1,64% positivo en las 3** (0.134117→0.136318),
  ponderado por actividad +5,7%. Pero **plano en Kaggle** (0.06307 vs 0.06316, dentro del
  ruido de una submission ~0.0065). El mecanismo no puede empeorar en esperanza → el
  código **queda** (`comparar_ensamble_pareado.py`, `_RankerBag`) con `N_BAG_RANKER=1` de
  default (dev rápido) y opt-in a 5 para submissions finales. El valor real de la
  estrategia de ensamble está en un blend de familias de modelos distintas, no en el
  seed-bag solo. Ver `experiments/estrategias.md`.
- **Retrieval aprendido como 7ª fuente de candidatos** (estrategia 2 de `estrategias.md`;
  `scripts/tune_retrievers.py`) — se optimizaron con optuna 4 familias de `implicit` (BPR,
  LMF, cosine-kNN, BM25-kNN) maximizando el recall@200 **complementario a ALS** (recall de
  `ALS-top200 ∪ retriever-top200`). Ganó **LMF** (`factors=33, lr=0.985, reg=1.73,
  iter=105`): recall complementario 0.4048 → 0.5025 en el split standalone. Cableado como
  7ª fuente (mismo patrón que autor/resumen/co-lectura, +3 features, 39→42): recall del set
  completo 0.5346 → 0.5547 (seed 42, nfa=500); CV 3 seeds NDCG@20 0.134117 → 0.134412
  (**+0,22%, dentro del ruido**); test pareado seed 42 **+1,48 σ / P=0.93** (borderline).
  **Kaggle: 0.06164 vs 0.06316 — regresión.** Mismo patrón que la 7ª fuente usuario-usuario
  y `n_por_fuente=500`: el recall sube pero los candidatos nuevos son ruido que el
  `LGBMRanker` no distingue y **desplazan** candidatos mejores en el set de Kaggle (heavy
  users). El cuello de botella no es el recall crudo sino el recall *distinguible*.
  Revertido el cableado (`ranker.py`/`submit.py`/tests, `models/lmf.py`); queda
  `scripts/tune_retrievers.py` (la optimización, reutilizable).

---

## La forma de U por popularidad del objetivo (cerrada como techo estructural)

**Estado (2026-09-09): dada por cerrada.** 7 ángulos, ninguno la mueve — ver el detalle
más abajo. El diagnóstico apunta a que no es un problema de features/representación:
LightGBM no está ignorando señal disponible, y hay un componente aleatorio irreducible.
Retomarla solo tendría sentido con una fuente de datos genuinamente nueva (metadata de
serie/saga), que es un mini-proyecto de payoff incierto.

De los usuarios cuyo objetivo **sí está entre los candidatos** (~4760, recall 0.535), el
reranker lo mete al top-20 solo el **~43%** de las veces. Y la relación entre popularidad
del objetivo y P(top-20) tiene **forma de U**: los peores son los de **popularidad MEDIA**.

| decil de popularidad del objetivo | pop. mediana | P(top-20) |
|---|---|---|
| 0 (menos popular) | 3 | 0.47 |
| 2 | 51 | 0.37 |
| **4** | **180** | **0.31** ← peor |
| 6 | 527 | 0.43 |
| 9 (más popular) | 1449 | 0.63 |

Es pérdida en el **ranking**, no en la generación (los candidatos están). Dentro de la
franja media, el discriminador más fuerte del éxito es que **varias fuentes coincidan** en
el objetivo — pero agregarlo como feature no ayuda (ya está en los `rank_*`/`score_*`).

**Los fracasos de la franja media NO son "casi aciertos"** (diagnóstico 2026-09-09): de
los 939 fracasos en deciles 3-5, solo el 24% quedan en posición predicha 20-49; el 53% en
100+ (mediana ~100). El modelo tiene ~100 candidatos igual de plausibles y el correcto
está perdido en esa sopa — una feature-empujón o calibración por decil no sirven. (En
contraste, los fracasos de la franja ALTA sí son near-misses: 49% en posición 20-49.) Hay
además un componente **irreducible**: para un heavy user que lee amplio, "cuál es EL
próximo libro" tiene decenas de respuestas igual de válidas, y la franja media es donde
esa incertidumbre es máxima.

**7 ángulos, ninguno la mueve** (detalle en "Qué se probó y NO funcionó"): (1) cobertura
de candidatos, (2) BM25 en ALS, (3) presupuesto de autor, (4) features de corroboración
(`n_fuentes_candidato`, −1,42 σ), (5) supervisión más densa (`n_val_ranker=3`, −10,8 σ),
(6) filtrar el entrenamiento por historial (`min_hist`, monótono negativo hasta −11,9 σ),
(7) features relativas dentro del usuario / cascade (−0,09 σ, plano). Los que tocan el
entrenamiento salen fuerte negativos; los que agregan features salen ruido. **No es un
problema de representación de features** — LightGBM no está ignorando señal disponible.
Sumado a la incertidumbre irreducible de la franja media, se toma como **techo
estructural** de este approach (2 etapas + LightGBM sobre estas features + este dataset).

Herramientas de diagnóstico: `scripts/diagnostico_posicion_popularidad.py`,
`scripts/diagnostico_franja_media.py`, `scripts/diagnostico_cap_autor.py` (apuntadas al
modelo de producción, `n_por_fuente_autor=500`).

---

## Cómo correr

- `uv run pytest` — suite (122 tests).
- `uv run python -m src.recsys.submit --model ranker` — genera el CSV en
  `outputs/submissions/` (usa `--tag` para un sufijo descriptivo; los nombres nunca se
  pisan). Entrena con ventana rodante `n_cortes=5` (`N_CORTES_RANKER_SUBMISSION`).
- `uv run python scripts/evaluate_ranker.py` — CV 3 seeds + NDCG por bucket de actividad +
  `feature_importances_` (`N_CORTES = 5`, refleja producción; ~34 min/seed en frío).
- `uv run python scripts/recall_candidatos.py` — recall del set + posición del objetivo.
- `scripts/comparar_features_pareado.py` / `comparar_generadores_pareado.py` — test pareado
  (editar `FEATURES_A`/`FEATURES_B` o `FUENTES_A`/`FUENTES_B`).
- `scripts/comparar_cortes_pareado.py` — test pareado `n_cortes=1` vs `3` (ventana rodante,
  estrategia 1 — **aplicada**, récord 0.06753 con `n_cortes=5`). Editar `CORTES` para otras
  profundidades; con `ctx_base` (`n_cortes=1`) vivo, `n_cortes=5` es límite de RAM.
- `scripts/tune_retrievers.py` — optuna sobre retrievers colaborativos alternativos (BPR /
  LMF / cosine-kNN / BM25-kNN) maximizando recall@200 complementario a ALS. Escribe
  `data/cache/tune_retrievers.json`. (Estrategia 2 — probada y descartada, ver arriba.)
- Familias `diagnostico_*.py` (posición/popularidad del objetivo, franja media,
  presupuesto de autor, historial de entrenamiento del ranker), `screen_*.py` (barridos
  de presupuesto/BM25), `probe_*.py`.

**Cache de contexto**: `preparar_pipeline_cacheado` guarda el contexto en `data/cache/`
(~3 GB c/u con `n_cortes=1`, ~7 GB con `=3`, ~11 GB con `=5`, gitignored). La clave = `seed`
+ `n_por_fuente*` + `n_cortes` + hash de los bytes de `ranker.py` → editar `ranker.py`
invalida todo. Para **barridos**: un proceso por valor (los contextos no se liberan bien
entre iteraciones y agotan la RAM). `evaluate_ranker.py` con `n_cortes=5` corre los 3 seeds
en un proceso porque libera cada contexto antes del siguiente.

---

## Dónde mirar más detalle

- **`experiments/estrategias.md`** — análisis de estrategias de mayor calibre para superar
  el plateau (ventana rodante de entrenamiento, retrieval aprendido, ensamble, rating
  predicho, retrieval por grafo), con recomendación priorizada.
- **`experiments/legacy/`** — historia congelada al 2026-09-07: `bitacora.md` (narrativa
  ronda por ronda), `decisiones.md` (tabla numerada #1–26 + investigación abierta del
  límite del reranking), `modelo_actual.md` (técnico + análisis "¿cambiar de paradigma?").
- **`experiments/log.csv`** — una fila por corrida (NDCG local vs Kaggle + nota completa).
- **`experiments/features.md`** — catálogo feature por feature.
- **`experiments/eda.md`** — análisis exploratorio (cola larga, calidad de datos, géneros,
  demografía).
