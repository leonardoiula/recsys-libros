# Estado del arte — recsys-libros

Punto de entrada para retomar el proyecto: el modelo actual, cómo se valida, qué se probó
y no funcionó, y el problema abierto. El razonamiento completo ronda por ronda está
**congelado** en `experiments/legacy/` (ver `experiments/legacy/README.md`); no hace falta
leerlo para retomar contexto.

**Récord: 0.06316 de NDCG@20 en Kaggle** (2026-09-07).

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
| Ranker, +presupuesto de autor (`n_por_fuente_autor=500`) | 0.134117 ± 0.00096 | **0.06316** |

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

---

## Problema abierto: la forma de U por popularidad del objetivo

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

Atacada esta sesión desde: cobertura de candidatos, BM25 en ALS, presupuesto de autor,
features de corroboración — **ninguna la mueve**. El diagnóstico apunta a **"falta señal"**
para esa franja, no a "el modelo la ignora". Ángulos sin probar: un objetivo del reranker
distinto, un modelo/tratamiento separado para la franja media, o aceptarlo como techo
estructural de este approach (2 etapas + LightGBM sobre estas features).

Herramientas de diagnóstico: `scripts/diagnostico_posicion_popularidad.py`,
`scripts/diagnostico_franja_media.py`, `scripts/diagnostico_cap_autor.py` (apuntadas al
modelo de producción, `n_por_fuente_autor=500`).

---

## Cómo correr

- `uv run pytest` — suite (118 tests).
- `uv run python -m src.recsys.submit --model ranker` — genera el CSV en
  `outputs/submissions/` (usa `--tag` para un sufijo descriptivo; los nombres nunca se
  pisan).
- `uv run python scripts/evaluate_ranker.py` — CV 3 seeds + NDCG por bucket de actividad +
  `feature_importances_`.
- `uv run python scripts/recall_candidatos.py` — recall del set + posición del objetivo.
- `scripts/comparar_features_pareado.py` / `comparar_generadores_pareado.py` — test pareado
  (editar `FEATURES_A`/`FEATURES_B` o `FUENTES_A`/`FUENTES_B`).
- Familias `diagnostico_*.py` (forma de U, franja media, presupuesto de autor),
  `screen_*.py` (barridos de presupuesto/BM25), `probe_*.py`.

**Cache de contexto**: `preparar_pipeline_cacheado` guarda el contexto en `data/cache/`
(~3 GB c/u, gitignored). La clave = `seed` + `n_por_fuente*` + hash de los bytes de
`ranker.py` → editar `ranker.py` invalida todo. Para **barridos**: un proceso por valor
(los contextos de ~3-4 GB no se liberan bien entre iteraciones y agotan la RAM).

---

## Dónde mirar más detalle

- **`experiments/legacy/`** — historia congelada al 2026-09-07: `bitacora.md` (narrativa
  ronda por ronda), `decisiones.md` (tabla numerada #1–26 + investigación abierta del
  límite del reranking), `modelo_actual.md` (técnico + análisis "¿cambiar de paradigma?").
- **`experiments/log.csv`** — una fila por corrida (NDCG local vs Kaggle + nota completa).
- **`experiments/features.md`** — catálogo feature por feature.
- **`experiments/eda.md`** — análisis exploratorio (cola larga, calidad de datos, géneros,
  demografía).
