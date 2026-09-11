# Estado del arte — recsys-libros

Punto de entrada para retomar el proyecto: cómo funciona el sistema, cómo se valida, qué se
probó y no funcionó, y el problema abierto. Es lo único que hace falta leer para retomar
contexto. El razonamiento ronda por ronda hasta 2026-09-07 está **congelado** en
`experiments/legacy/` (ver `experiments/legacy/README.md`); lo posterior está en
`experiments/log.csv` (una fila por corrida).

**Récord: 0.06824 de NDCG@20 en Kaggle** (2026-09-10).

---

## El problema

Competencia de Kaggle: recomendar libros, evaluada con **NDCG@20**. Para cada lector se
entrega un ranking de 20 libros (el orden importa) y se puntúa según dónde cae el libro que
ese lector efectivamente leyó después.

Datos (`data/raw/data.db`): 461.408 interacciones, 10.673 lectores con actividad, 128.743
libros (48.137 con ≥1 interacción). Matriz usuario×libro **99,91 % vacía**. Rating explícito
1–10. Metadata de libro: autor, editorial, año, género, resumen.

El set de evaluación de Kaggle (`data/raw/ejemplo.csv`) son **~832 usuarios sesgados a alta
actividad** (mediana ~95 interacciones vs ~9 en la evaluación local). Span mediano
primera→última interacción por usuario: **26 días** (la mayoría lee en una ráfaga y se va).

---

## Cómo funciona el sistema (`--model ranker`)

Ranker de **dos etapas**: 6 fuentes heurísticas generan un pool de candidatos por usuario,
un `LGBMRanker` los reordena.

```
 interacciones --> split leave-one-out TEMPORAL por usuario  (x_1 .. x_m, m = la ultima)
                        |
        +---------------+--------------------------------+
        v               v                                v
  train_candidatos   train_ranker                     test_final
  (fitea etapa 1)    (etiqueta = x_{m-1};             (x_m; solo local,
        |             ventana rodante: tambien          para NDCG@20)
        |             x_{m-2} .. x_{m-5})
        v
  +--------------------- ETAPA 1 - candidatos ----------------------+
  |  1  ALS (implicit, factors=128, BM25 en la matriz)             |
  |  2  popularidad global (score bayesiano)                       |
  |  3  popularidad del genero preferido del usuario               |   union dedup.
  |  4  libros de autores ya leidos (<=20/autor, <=500 total)      +--> ~780 cand/usuario
  |  5  similitud de resumen (TF-IDF perfil <-> catalogo)          |   marcados por fuente
  |  6  co-lectura item-item kNN  (X . cooc, 1 hop)                |
  +--------------------------------+------------------------------+
                                   |  + 40 features por (usuario, candidato)
                                   v
  +--------------------- ETAPA 2 - LGBMRanker ---------------------+
  |  objective=lambdarank, 200 arboles, hiperparametros conserv.  |
  |  entrenado con ~37M filas  (ventana rodante n_cortes=5)       +--> top-20 por usuario
  +--------------------------------------------------------------+

 Produccion (submit.py): la etapa 1 se REFITEA sobre `interacciones` completo antes de
 generar los candidatos finales (usa la ultima interaccion de cada usuario como senal).
 Cold-start (usuario sin fila en ALS) -> fallback a popularidad global.
```

### Etapa 1 — las 6 fuentes

Cada una propone hasta `n_por_fuente=150` libros sin leer, **salvo la 4 (autor)** que tiene
presupuesto propio `n_por_fuente_autor=500`. Las fuentes 1/5/6 solo alcanzan usuarios con
fila en la matriz de ALS.

| # | Fuente | Qué propone |
|---|---|---|
| 1 | **ALS** | top-150 de `implicit` |
| 2 | **Popularidad global** | top-150 del ranking bayesiano |
| 3 | **Popularidad por género preferido** | top-150 dentro del género que más leyó el usuario |
| 4 | **Autores ya leídos** | ≤20 libros/autor leído (por popularidad global), ≤500 total/usuario |
| 5 | **Similitud de resumen** | top-150 del catálogo con `resumen` más parecido al perfil TF-IDF |
| 6 | **Co-lectura ítem-ítem (kNN)** | top-150 por `X·cooc` (`cooc[i,j]` = # usuarios que leyeron `i` y `j`) |

**ALS** (`models/als.py`): `factors=128, reg=0.1, iters=20`, confianza = rating crudo
(`alpha=None` — se probó tuneado y empeoró en Kaggle). **BM25 en la matriz**
(`BM25_ALS=(10.0, 0.75)`): `bm25_weight` solo sobre la copia que va a `.fit()` — baja el
peso de los power users (11,5 % de usuarios = 64 % de la señal) en la factorización sin
tocar el filtro de ya-leídos ni `cooc`.

### Etapa 2 — LGBMRanker (`fit_ranker`)

`objective="lambdarank"`, `num_leaves=31, learning_rate=0.05, n_estimators=200`.
Hiperparámetros **conservadores a propósito** (optuna probado 3 veces, siempre ruido).

**Ventana rodante multi-corte** (`N_CORTES_RANKER_SUBMISSION=5`, `N_CORTES=5` en
`evaluate_ranker.py`; `N_CORTES_RANKER=1` de default para el dev). Además del ejemplo
`(historial → x_{m-1})` por usuario, se agregan los cortes `x_{m-2}…x_{m-5}`: **el corte `j`
predice `x_{m-j}` con su propia etapa 1** fiteada solo sobre `x_1…x_{m-1-j}` y sus features
sobre ese mismo historial — cada uno es un next-item genuino, sin leakage, **1 positivo por
grupo**. ~2,8×–5× la supervisión (7.932 → ~37M filas). `test_final` y el recall del set de
candidatos no cambian. El CV sube monótono hasta `n_cortes=5` y **plateaua ahí** (`=7`
plano). Los buckets casuales (2-4, 5-9 interacciones) *mejoran* a más profundidad. Es el
lever que más movió Kaggle (0.06316 → 0.06753). Detalle en `experiments/log.csv`.

La unión de los cortes se arma volcando cada `X_j` a un `.npy` temporal y leyéndolos con
`mmap` a un array `float32` preasignado (`_ensamblar_dataset_ventana_rodante`) — pico ~1× el
tamaño de la unión (~10 GB para `n_cortes=7`), no ~2× como un `pd.concat`.

### 40 features (`experiments/features.md` = catálogo)

- **score / rank / en de las 6 fuentes** (18) · **volumen** (interacciones libro/usuario, 2)
- **autor / editorial** (¿ya lo leyó?, cuántos, + tamaño del catálogo de la editorial, 5)
- **año** (diferencia contra el promedio del usuario, 1) · **diversidad/recencia del usuario** (2)
- **co-lectura / resumen**: `score_coleido` (1 hop), `sim_resumen_historial` (2)
- **`score_difusion_candidato`** (1): difusión **2-hop** sobre `cooc` **podada** (aristas de
  <8 co-lectores fuera — el grafo crudo tiene 73M aristas y a 1 hop ya toca el 43 % del
  catálogo; sin podar solo difumina) y row-normalizada, `Σ_{t=1}^{2} 0,85ᵗ (Hᵗ)[u,j]`.
  Reweightea la señal colaborativa por cercanía en cadena. Pareado +1,17 σ, Kaggle 0.06753 →
  **0.06824** (chico, confianza modesta; ambos instrumentos coinciden en dirección). Ver
  `MIN_COREAD_PPR`/`K_DIFUSION`.
- **macro-género** (popularidad pooleada a 10 familias + frecuencia en el historial, 2)
- **señales cruzadas lector↔libro** (popularidad por género declarado del lector, afinidad
  de cohorte, edad al publicarse, 3)
- **recencia-ponderadas** (4): variantes de autor/editorial/co-lectura/resumen que pesan más
  lo leído hace poco.

---

## Récord y progresión

| Modelo | NDCG@20 local | Kaggle |
|---|---|---|
| Popularidad global (bayesiana) | 0.006620 | 0.01024 |
| Popularidad segmentada (género → franja → global) | 0.013719 | 0.01558 |
| ALS solo | 0.094406 ± 0.00136 | 0.03864 |
| Ranker 3 fuentes | 0.109735 ± 0.00372 | 0.04831 |
| Ranker 4 fuentes (+autor) | 0.117495 ± 0.00256 | 0.05140 |
| Ranker 5 fuentes (+resumen) | 0.120547 ± 0.00267 | 0.05181 |
| Ranker 6 fuentes (+co-lectura) | 0.121983 ± 0.00295 | 0.05262 |
| +recencia (4 feat) + refit de etapa 1 | 0.143336 ± 0.00290 (nf=75) | 0.06149 |
| +BM25 en la matriz de ALS | 0.132313 ± 0.00219 (nf=150) | 0.06182 |
| +presupuesto de autor (`nfa=500`) | 0.134117 ± 0.00096 | 0.06316 |
| +ventana rodante `n_cortes=3` | 0.136561 ± 0.00178 | 0.06667 |
| ventana rodante `n_cortes=5` | 0.137854 ± 0.00148 | 0.06753 |
| +`score_difusion_candidato` | pareado +1,17 σ | **0.06824** |

Los NDCG locales solo comparan **dentro** del mismo `n_por_fuente`.

---

## Cómo se valida (en orden de confianza)

1. **Test pareado por usuario** (`comparar_features_pareado.py` / `comparar_generadores_pareado.py`
   / `comparar_cortes_pareado.py`): dos configs sobre el **mismo contexto/seed**, diferencia
   de NDCG por usuario. **~5× el poder** del desvío entre 3 seeds — el gatekeeper real.
   Varias mejoras "positivas en los 3 seeds" resultaron ruido acá. **Medir a `nfa=500`**
   (config de producción): a `nfadef` engaña (el episodio de rating dio +1,67 σ a `nfadef` y
   −2,34 σ a `nfa=500`).
2. **CV sobre 3 seeds (42, 7, 123)** — media *y* desvío (`evaluate_ranker.py`). Nunca un
   solo split: un sweep de ALS sobre un split único mejoró el NDCG local +11,5 % y empeoró
   Kaggle −13,5 %.
3. **Kaggle**: ~832 usuarios, SE ≈ 0.0065 → una submission no distingue configs que difieren
   <~0.01. Confiar en el pareado/CV para lo fino, Kaggle para confirmar **dirección**.
   Criterio para gastar una submission: pareado positivo (aunque sea borderline). Casos
   donde local y Kaggle **discreparon en dirección** = rechazar (seed-bag: pareado +1,77 σ,
   Kaggle en contra).
4. **`recall_de_candidatos`** (`recall_candidatos.py`): fracción de objetivos en el pool =
   techo duro del reranker (**~0.535**). Regla dura: **subir recall no se traduce en NDCG**
   si los candidatos nuevos no son *distinguibles* (pasó con `n_por_fuente=500`, LMF, 7ª
   fuente usuario-usuario).
5. **NDCG ponderado por actividad + bucket de actividad** (`evaluate_ranker.py`): diagnóstico
   de generalización, no cambia decisiones.

**Splits**: leave-one-out **temporal** (`split_train_val`, `n_val=1`) — un split aleatorio
filtra futuro y sobreestima ~2×. Para el ranker, **3 niveles**: `train_candidatos` (fitea
etapa 1) / `train_ranker` (etiquetas) / `test_final` (hold-out, solo local).

---

## Qué se probó y NO funcionó (para no repetirlo)

### Tuning / candidatos

- **Optuna sobre LightGBM** — 3 veces, siempre ruido. **Optuna sobre ALS** (`factors=256,
  alpha=4.718`) — +11,5 % local, **peor** en Kaggle (sobreajuste a un split).
- **País / franja de nacimiento** como features — empeoran en 2/3 seeds.
- **`n_por_fuente=500` global** — +30 % recall, NDCG plano. **`n_por_fuente_autor>500`** —
  `800` +0,84 σ sobre `500` (ruido). **Presupuesto propio resumen/co-lectura** — sube recall,
  NDCG no acompaña.
- **Features de corroboración** (`n_fuentes_candidato`, `rank_min_candidato`) — pareado
  **−1,42 σ**. La señal ya está en los `rank_*`/`score_*` individuales.
- **7ª fuente por editorial ya leída** — recall no se movió (catálogos dispersos, sus
  populares ya los traían ALS/popularidad).
- **7ª fuente por similitud usuario-usuario (kNN)** — CV 2/3 seeds positivo, **regresión en
  Kaggle** (0.06017 vs 0.06149).
- **Retrieval aprendido como 7ª fuente** (estrategia 2; `scripts/tune_retrievers.py` optimizó
  BPR/LMF/cosine-kNN/BM25-kNN por recall complementario a ALS — ganó **LMF**). Cableado:
  recall del set 0.5346 → 0.5547, CV +0,22 % (ruido), pareado **+1,48 σ**, **Kaggle 0.06164
  vs 0.06316 — regresión**. El recall sube pero los candidatos nuevos son ruido que el
  ranker no distingue y **desplazan** candidatos mejores. Revertido; queda `tune_retrievers.py`.
- **Difusión multi-hop como 7ª fuente** (estrategia 5, parte 2 — la feature
  `score_difusion_candidato` ya estaba adoptada, récord 0.06824 sin cambios). Mismo patrón
  que coleido (6ª fuente): top-`n_por_fuente` de la propagación multi-hop como candidatos
  nuevos. Recall 0.5346 → 0.5398 (+1 %, ~9 candidatos extra/usuario), pero pareado 7 vs 6
  **−0,80 σ, P(mejora)=0.22** (0.133997 → 0.133329) — **4ª fuente nueva que falla así**
  (editorial, usuario-usuario, LMF, ahora difusión). Revertido el bloque de fuente; la
  feature queda intacta. La estrategia 5 cierra del todo: funciona como feature, no como
  fuente — la señal sirve para *puntuar*, no para *ampliar el pool*.

### Features del reranker

- **Embeddings semánticos vs TF-IDF** para el perfil — recall igual, NDCG **−1,74 σ**.
- **Features relativas dentro del usuario / cascade** (percentiles de score entre los
  candidatos del usuario) — pareado **−0,09 σ** (plano). Cierra el cascade completo: si esa
  señal listwise no aparece en un modelo de 1 etapa, dos `LGBMRanker`s no la crean.
- **Feature de rating predicho** (estrategia 4; `models/rating.py`, ALS de feedback explícito
  `μ + b_u + b_i + p_u·q_i`, RMSE in-sample 0.97 vs 1.82). `rating_predicho_candidato` +
  `rating_medio_usuario`. Pareado seed 42: **+1,67 σ a `nfadef`** pero
  **−2,34 σ / P=0.005 a `nfa=500`** (`rating_predicho_candidato` sola, 0.133014 → 0.131131)
  — **significativamente dañina** a config real. "Le pondría ≥8" no discrimina *cuál* de los
  muchos next-reads plausibles elige el usuario. Revertida (`models/rating.py` eliminado).

### Entrenamiento del reranker

- **`n_val_ranker=3` (versión ingenua)**: los 3 libros más recientes como positivos **en un
  solo grupo**, con **una sola etapa 1** compartida — pareado **−10,8 σ**. Causas: (a)
  achica `train_candidatos`, degrada la etapa 1; (b) lambdarank con 3 positivos desdibuja el
  objetivo. **La versión bien hecha SÍ funcionó** — ventana rodante multi-corte (1 positivo
  por grupo, una etapa 1 propia por corte). Lección: no es que "más supervisión" esté mal,
  hay que darle a cada ejemplo su estado de etapa 1 correcto.
- **Filtrar el entrenamiento por historial mínimo** (`min_hist`) — pareado **monótono
  negativo** (`min_hist=20` → −11,9 σ). Los ejemplos de usuarios finos son la única señal
  sobre cómo rankear para un usuario del que se sabe poco; sacarlos = sobreajuste a usuario
  rico.
- **Seed-bag del `LGBMRanker`** (`N_BAG_RANKER`, `_RankerBag`) — pareado **+1,77 σ**, CV
  **+1,64 % positivo en los 3**, pero **plano en Kaggle** (0.06307 vs 0.06316) — y local vs
  Kaggle discreparon en dirección. Código **queda** opt-in (`N_BAG_RANKER=1` de default, 5
  para submissions finales). El valor real del ensamble estaría en blend de familias
  distintas, no en 5 copias del mismo modelo.

### Otras direcciones

- **Modelos secuenciales (SASRec / GRU4Rec)** — 67,5 % de los gaps entre interacciones son 0
  días: el "orden" intradía es arbitrario.
- **LightFM / dos torres / FM** — reemplazarían a ALS (que no es el cuello de botella); su
  metadata ya está en las features.
- **Rutear usuarios livianos a popularidad por género** (`recomendar_hibrido`) — ALS le gana
  a género en **todos** los buckets, incluso con 1 interacción.
- **Metadata de serie/saga por títulos** — falla por **cobertura**: 2,0 % de los libros con
  patrón limpio, 0,8 % de los targets son "el siguiente de una saga". Techo ~+0,0005.

---

## La forma de U por popularidad del objetivo (techo estructural, cerrada 2026-09-09)

De los usuarios cuyo objetivo **sí está entre los candidatos** (~4760, recall 0.535), el
reranker lo mete al top-20 solo el **~43 %** de las veces, y la relación con la popularidad
del objetivo tiene **forma de U** — los peores son los de **popularidad media**:

| decil de popularidad del objetivo | pop. mediana | P(top-20) |
|---|---|---|
| 0 (menos popular) | 3 | 0.47 |
| 2 | 51 | 0.37 |
| **4** | **180** | **0.31** ← peor |
| 6 | 527 | 0.43 |
| 9 (más popular) | 1449 | 0.63 |

Es pérdida en el **ranking**, no en la generación. Los fracasos de la franja media **no son
near-misses**: de 939 fracasos en deciles 3-5, el 53 % queda en posición predicha 100+
(mediana ~100) — el modelo tiene ~100 candidatos igual de plausibles y el correcto está
perdido en esa sopa. Hay además un componente **irreducible**: para un heavy user que lee
amplio, "cuál es EL próximo libro" tiene decenas de respuestas válidas.

**7 ángulos, ninguno la mueve**: cobertura de candidatos, BM25 en ALS, presupuesto de autor,
features de corroboración (−1,42 σ), `n_val_ranker=3` (−10,8 σ), `min_hist` (hasta −11,9 σ),
features relativas / cascade (−0,09 σ). Los que tocan el entrenamiento salen fuerte
negativos; los que agregan features salen ruido. **No es un problema de representación** —
LightGBM no ignora señal disponible. Techo estructural de este approach.

Diagnóstico: `scripts/diagnostico_posicion_popularidad.py`, `diagnostico_franja_media.py`,
`diagnostico_cap_autor.py`.

---

## Qué queda por probar (`experiments/estrategias.md`)

Se acabó lo barato: LMF-feature, corroboración, relativas, rating, difusión-como-fuente —
todo ruido o negativo. Solo `score_difusion_candidato` como **feature** movió algo (y poco).
Lo que queda es más caro:

- **Estrategia 6 — BERT4Rec / masked-item** como fuente + feature de "interés actual". Mayor
  ceiling, mayor costo/riesgo (461k interacciones es poca data para un transformer).
- Blend de **familias de modelos distintas** (no seed-bag).

El lever que rindió de verdad esta ronda fue **más señal de entrenamiento** (ventana
rodante). El set de features y de fuentes de candidatos está saturado — **4/4 intentos de
7ª fuente fallaron** (editorial, usuario-usuario, LMF, difusión), incluso cuando la señal
de base ya era buena. La barrera no es "encontrar más candidatos", es que el `LGBMRanker`
no logra distinguir cuáles de los candidatos extra son buenos.

---

## Cómo correr

- `uv run pytest` — suite (124 tests).
- `uv run python -m src.recsys.submit --model ranker [--tag ...]` — CSV en
  `outputs/submissions/` (nombres con timestamp, nunca se pisan). Entrena con
  `n_cortes=5` + refit de etapa 1. ~40 min.
- `uv run python scripts/evaluate_ranker.py` — CV 3 seeds (`N_CORTES=5`, ~34 min/seed en
  frío) + NDCG por bucket + `feature_importances_`.
- `uv run python scripts/recall_candidatos.py` — recall del set + posición del objetivo.
- `scripts/comparar_features_pareado.py` / `comparar_generadores_pareado.py` /
  `comparar_cortes_pareado.py` — tests pareados (editar las listas / `FUENTES_*` / `CORTES`).
  A `nfa=500`.
- `scripts/tune_retrievers.py` — optuna sobre retrievers alternativos (estrategia 2,
  descartada).
- Familias `diagnostico_*.py`, `screen_*.py`, `probe_*.py`.

**Cache de contexto** (`preparar_pipeline_cacheado`, `data/cache/`, gitignored): ~3 GB con
`n_cortes=1`, ~7 GB con `=3`, ~11 GB con `=5`. Clave = `seed` + `n_por_fuente*` + `n_cortes`
+ hash de los bytes de `ranker.py` (editar `ranker.py` invalida todo). En cada llamada barre
los contextos con hash de código viejo, así el caché no acumula sin techo. Para barridos: un
proceso por valor (los contextos no se liberan bien entre iteraciones).

---

## Dónde mirar más detalle

- **`experiments/estrategias.md`** — estrategias de mayor calibre + recomendación priorizada.
- **`experiments/log.csv`** — una fila por corrida (NDCG local vs Kaggle + nota completa),
  desde 2026-09-07 en adelante.
- **`experiments/features.md`** — catálogo feature por feature.
- **`experiments/legacy/`** — historia congelada al 2026-09-07: `bitacora.md` (narrativa),
  `decisiones.md` (#1–26), `modelo_actual.md` (técnico + "¿cambiar de paradigma?").
- **`experiments/eda.md`** — análisis exploratorio.
