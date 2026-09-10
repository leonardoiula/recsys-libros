# Estrategias para superar el plateau (récord actual 0.06753)

Análisis para decidir el próximo paso grande. Escrito con el modelo en el plateau 0.06316
(6 fuentes → `LGBMRanker`, 7 ángulos contra la forma de U fallados, generación de
candidatos por heurísticas agotada). **La estrategia 1 (ventana rodante) lo rompió: Kaggle
0.06316 → 0.06667 (`n_cortes=3`) → 0.06753 (`=5`)** (ver más abajo). Las estrategias 2 y 3
se probaron y no rindieron.

## Dónde está la pérdida

1. **El techo de recall del set de candidatos es ~0.535.** El 46% de las veces el libro
   objetivo **no es ni candidato** — ningún reranker lo puede recuperar. Esta es, de lejos,
   la restricción dominante.
2. El reranker captura ~1/4 de ese techo (NDCG@20 local ~0.133 sobre un techo ~0.535).
   Hay margen, pero acotado por (1).
3. La tarea real, medida sobre los held-out: **76% de los "próximos libros" están
   calificados ≥7, 62% ≥8** (mediana 8). O sea el objetivo es "el próximo libro que el
   usuario lee **y le gusta**", no cualquier lectura.
4. ~~El `LGBMRanker` entrena con **7.932 queries** (1 por usuario). Es muy poca supervisión
   para 39 features.~~ **Saldado por la estrategia 1**: la ventana rodante multi-corte lo
   subió a ~37M filas / ~37k queries con `n_cortes=5` (récord 0.06753). El CV plateaua ahí
   (`n_cortes=7` plano) — la supervisión ya no es el cuello de botella.

## Ya descartado (no re-intentar)

- Los 7 ángulos contra la forma de U (ver `estado_del_arte.md`).
- **SASRec / GRU4Rec autoregresivos**: 67,5% de los gaps entre interacciones son 0 días →
  el orden intradía es ruido (análisis en `legacy/modelo_actual.md`).
- **LightFM / two-tower como reemplazo de ALS para meter metadata en el embedding**: esa
  metadata ya está en las features del ranker (`legacy/modelo_actual.md`). *Ojo*: esto NO
  descarta two-tower como modelo de **retrieval** (ver estrategia 2).

---

## Estrategias candidatas

Ordenadas por (leverage × probabilidad) / esfuerzo.

### 1. Entrenamiento del reranker con ventana rodante (multi-corte)  ·  APLICADA (récord 0.06753)

**Estado (2026-09-10): implementada y adoptada — el mayor salto desde recencia+refit.**
`N_CORTES_RANKER` en `ranker.py` (default 1, dev), `N_CORTES_RANKER_SUBMISSION = 5` en
`submit.py`. Además del ejemplo `(historial → x_{m-1})` por usuario, se agregan los cortes
`x_{m-2}…x_{m-5}`: el corte `j` predice `x_{m-j}` con **su propia etapa 1** fiteada solo
sobre `x_1…x_{m-1-j}` y sus features sobre ese mismo historial — next-item genuino, sin
leakage, **1 positivo por grupo**. Los usuarios con poco historial se caen solos de los
cortes profundos. `test_final` y el recall del set de candidatos no cambian.

- **Resultado**: test pareado seed 42 (`n_cortes=3` vs `1`) **+2,32 σ / P=0.985**.
  Progresión CV 3 seeds (positivo en los 3 en cada salto) / Kaggle: `n_cortes=1` 0.134117 →
  `=3` **0.136561** (Kaggle 0.06316 → **0.06667**) → `=5` **0.137854** (Kaggle → **0.06753**).
  Sin regresión por bucket — los buckets **casuales** (2-4, 5-9) *mejoran* a más
  profundidad: más diversidad de profundidad-de-historial en el entrenamiento → mejor
  generalización a usuarios de historial corto. `n_interacciones_usuario` sube en importancia.
- **Por qué el `n_val_ranker` barato había fallado** (−10 σ): metía N positivos en un grupo
  y compartía **una sola** etapa 1 (fiteada sobre datos recientes) para todos los cortes →
  leakage en las etiquetas viejas + `train_candidatos` global achicado. La versión buena da
  a cada corte su etapa 1 correcta.
- **Memoria**: el armado de la unión vuelca cada corte a un `.npy` temporal y lo libera de
  RAM ni bien lo arma (`_ensamblar_dataset_ventana_rodante`); al final se leen con `mmap` a
  un único array `float32` F-contiguo del que la `DataFrame` toma vistas sin copiar. Pico
  ~1× el tamaño de la unión (~10 GB medido para `n_cortes=7`, con ~16 GB de sistema libre),
  no ~2× como un `pd.concat` de todas las particiones. `n_cortes` alto ya no hace OOM.
- **Profundidad óptima ≈ 5**: `n_cortes=7` CV 3 seeds 0.137783 vs 0.137854 de `=5` — **plano**
  (media baja un pelo, positivo solo en 1/3). Los cortes `x_{m-6}`/`x_{m-7}` están más lejos
  del target real y vienen de un slice cada vez más angosto de usuarios de historial largo.
  Producción queda en **`n_cortes=5`**. Un `n_cortes` adaptativo por usuario (más cortes solo
  para los de historial largo) queda como idea, pero el plateau sugiere poco upside.

### 2. Retrieval aprendido (dual-encoder / two-tower como fuente de candidatos)  ·  esfuerzo alto, riesgo medio

**Estado (2026-09-10): probada la vía barata (retriever colaborativo alternativo), falló.**
`scripts/tune_retrievers.py` optimizó con optuna 4 familias de `implicit` (BPR, LMF,
cosine-kNN, BM25-kNN) maximizando el recall@200 *complementario a ALS*. Ganó **LMF**
(`factors=33, lr=0.985, reg=1.73, iter=105`): recall de `ALS ∪ LMF` 0.4048 → 0.5025
standalone. Cableado como 7ª fuente: recall del set completo 0.5346 → 0.5547, CV 3 seeds
+0,22% (ruido), test pareado +1,48 σ (borderline), **Kaggle 0.06164 vs 0.06316 —
regresión**. Mismo patrón que `n_por_fuente=500` y la 7ª fuente usuario-usuario: sube el
recall pero los candidatos nuevos son ruido que el ranker no distingue y desplazan
candidatos mejores. **El techo que importa no es el recall crudo, es el recall
distinguible.** Revertido; queda `scripts/tune_retrievers.py`. Un two-tower entrenado con
texto (lo de abajo) sigue sin probarse, pero la expectativa baja: si un retriever
colaborativo bien tuneado no convierte, un dual-encoder tiene que traer candidatos de
*otra naturaleza* (contenido puro) para justificar el build.

User-tower (encode el historial: mean/attention sobre embeddings de libros) + item-tower
(encode metadata + `resumen` con un encoder de texto congelado). Se entrena con el
próximo-libro real como positivo + negativos in-batch/sampleados; en inferencia, top-K por
vecino más cercano (ANN). **Feeda candidatos que las 6 heurísticas no encuentran** →
ataca el techo de recall 0.535 directamente.

- **Por qué podría ganar**: es la capa de retrieval estándar del recsys moderno; el
  cuello de botella es exactamente el recall.
- **Costo**: entrenar una red chica (o user-pooling aprendido sobre embeddings de texto
  congelados). Alto.
- **Riesgo**: con 461k interacciones puede no superar a ALS. El experimento de embeddings
  semánticos vs TF-IDF (fallido) fue como *feature*, no como retriever entrenado — no lo
  descarta pero baja la expectativa.

### 3. Ensamble / blend  ·  esfuerzo bajo, riesgo muy bajo

**Estado (2026-09-09): seed-bag probado, resultado dentro del ruido.** `n_bag=5` en
`fit_ranker` (ver `N_BAG_RANKER`): CV 3 seeds +1,64% (positivo en las 3), ponderado por
actividad +5,7%, pero Kaggle plano (0.06307 vs 0.06316). El código queda (opt-in a 5 para
submissions finales, default 1). **Pendiente el paso que rinde de verdad**: blend de
**familias de modelos distintas** (no 5 copias del mismo `LGBMRanker`) — p.ej.
rank-average o meta-learner sobre {ranker, ALS solo, popularidad por género reciente,
ranking content-only, un 2º ranker con features/objetivo distintos}.

La entrega es **un solo `LGBMRanker`**. Casi toda solución competitiva de Kaggle es un
blend:
- **Seed-bag del propio `LGBMRanker`**: 5 modelos con distinto `random_state`/subsample,
  promediar scores → mata la varianza de ajuste (el proyecto ya midió ~0.0006 de ruido de
  semilla en ALS; LightGBM tiene algo similar).
- **Rank-average o meta-learner** sobre {ranker actual, ALS solo, popularidad por género
  reciente, ranking content-only, un 2º ranker con features/seed distintos}.

- **Magnitud esperada**: +2–5% relativo, de forma confiable. Bajísimo riesgo.

### 4. Señal explícita de rating predicho  ·  esfuerzo bajo-medio, riesgo bajo

76% de los held-out están rateados ≥7. El proyecto usa el rating solo como *confianza* de
ALS. Un modelo que **prediga el rating** que un usuario le daría a un libro (BiasedMF /
SVD++ sobre la matriz 1-10, o un GBDT `P(rating ≥ 8)`) da una señal alineada con el
objetivo real ("leído **y** gustado"). Como fuente de candidatos (top predicted-rating sin
leer) y como feature del reranker.

- **Por qué podría ganar**: señal ortogonal y hoy desaprovechada; alineada con la métrica.
- **Magnitud esperada**: modesta pero real.

### 5. Retrieval por grafo (personalized PageRank / random walks)  ·  esfuerzo bajo-medio, riesgo bajo

Co-lectura es ítem-ítem a **1 hop**. Personalized PageRank desde el historial del usuario
sobre el grafo bipartito usuario-ítem (o el grafo ítem-ítem) propaga 2-3 hops y alcanza
libros relevantes que las fuentes de 1 hop no ven — sobre todo la franja media (libros de
nicho conectados por una cadena). Iteración de potencia sobre matriz dispersa, barato.

- **Magnitud esperada**: bump de recall, posiblemente significativo.

### 6. Modelo secuencial revisado (masked-item / BERT4Rec)  ·  esfuerzo alto, riesgo medio-alto

El descarte de secuenciales fue sobre el orden *fino*. Pero (a) el **masked-item modeling**
(predecir un libro enmascarado a partir del resto, bidireccional) es mucho más robusto al
orden ruidoso que la autoregresión estricta, y (b) el **drift inter-mensual** (los usuarios
pasan por fases de género/autor) es señal real que las features de "bag of historial"
capturan solo groseramente. Encoder tipo BERT4Rec → fuente de candidatos + una feature de
"embedding de interés actual".

- **Riesgo**: entrenar un transformer sobre 461k interacciones es poca data.

---

## Recomendación

1. ~~estrategia 1 — ventana rodante~~ **APLICADA y agotada: Kaggle 0.06316 → 0.06667
   (`n_cortes=3`) → 0.06753 (`=5`), récord.** El reranker estaba hambriento (7.932 queries);
   darle supervisión next-item con cada corte a su estado de etapa 1 correcto fue el lever
   estructural más grande. El CV plateaua en profundidad 5 (`=7` plano). El OOM se resolvió
   volcando cada corte a disco. **No queda upside claro en esta dirección.**
2. ~~estrategia 3 — seed-bag~~ **probada: +1,64% CV, plano en Kaggle.** Código opt-in.
3. ~~estrategia 2 — retrieval aprendido (vía barata)~~ **probada: recall +0,02, regresión en
   Kaggle.** El cuello de botella es el recall *distinguible*, no el crudo.
4. **Lo que queda**: estrategia 4 (rating predicho) y 5 (grafo / PPR) como fuentes/features
   baratas — pero la lección de la 2 acota: solo valen si traen señal *ortogonal*, no más
   recall del mismo tipo. Y estrategia 6 (BERT4Rec / masked-item) como fuente de candidatos
   + feature de "interés actual" — ahora con más razón, dado que subir la supervisión
   secuencial (estrategia 1) rindió.

**Observación meta** (parcialmente saldada): el proyecto había sobre-invertido en el *set
de features* del reranker y sub-invertido en cuántos datos ve. La estrategia 1 corrigió lo
segundo. Sigue abierto: (b) calidad del pool de candidatos y (c) ensamblado de familias
distintas.
