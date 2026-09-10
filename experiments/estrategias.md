# Estrategias para superar el plateau (0.06316)

Análisis para decidir el próximo paso grande. El modelo actual (6 fuentes → `LGBMRanker`)
está en un plateau: 7 ángulos contra la forma de U fallaron, la generación de candidatos
por heurísticas está agotada, y el score de Kaggle (0.06316) no es competitivo-alto.

## Dónde está la pérdida

1. **El techo de recall del set de candidatos es ~0.535.** El 46% de las veces el libro
   objetivo **no es ni candidato** — ningún reranker lo puede recuperar. Esta es, de lejos,
   la restricción dominante.
2. El reranker captura ~1/4 de ese techo (NDCG@20 local ~0.133 sobre un techo ~0.535).
   Hay margen, pero acotado por (1).
3. La tarea real, medida sobre los held-out: **76% de los "próximos libros" están
   calificados ≥7, 62% ≥8** (mediana 8). O sea el objetivo es "el próximo libro que el
   usuario lee **y le gusta**", no cualquier lectura.
4. El `LGBMRanker` entrena con **7.932 queries** (1 por usuario). Es muy poca supervisión
   para 42 features.

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

### 1. Entrenamiento del reranker con ventana rodante (multi-corte)  ·  esfuerzo medio, riesgo bajo

Hoy: 1 ejemplo (`historial → próximo`) por usuario. Con ventana rodante, para cada
usuario se predice la interacción *i* a partir de las 1..*i*-1, para varios *i* (p.ej. los
últimos 5-10 cortes). **~434.000 pares posibles (55× la supervisión actual)**; incluso con
5 cortes/usuario son ~40k queries (5×). Cada corte es una tarea genuina de next-item con
sus features calculadas sobre historial estrictamente anterior — **sin leakage** (distinto
del experimento `n_val_ranker=3`, que metía varios positivos "del pasado reciente" en un
solo grupo y desdibujaba el objetivo).

- **Por qué podría ganar**: los GBDT-rankers escalan bien con más grupos de query; el
  modelo actual está claramente hambriento (7.932 queries, 42 features). Es el lever más
  grande sin tocar.
- **Costo**: recalcular features por corte es la parte cara (dependen del estado del
  historial). Factible con pocos cortes por usuario (los últimos N).
- **Magnitud esperada**: potencialmente del orden de la ronda recencia+refit (+16,9%).

### 2. Retrieval aprendido (dual-encoder / two-tower como fuente de candidatos)  ·  esfuerzo alto, riesgo medio

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

1. **Ya (bajo riesgo, casi gratis)**: estrategia 3 — seed-bag del `LGBMRanker` + un blend
   simple. Una tarde, +2-5% esperado.
2. **La apuesta grande**: estrategia 1 — ventana rodante. El reranker está hambriento de
   datos y es lo más barato de los cambios estructurales.
3. **Si 1 y 3 no alcanzan y hay ganas de un build real**: estrategia 2 — retrieval
   aprendido, que ataca el techo de recall de verdad.
4. **En paralelo, como fuentes/features nuevas baratas**: 4 (rating predicho) y 5 (grafo).

**Observación meta**: el proyecto sobre-invirtió en el *set de features* del reranker (42,
ablacionadas exhaustivamente) y sub-invirtió en (a) cuántos datos ve el reranker, (b) qué
tan bueno es el pool de candidatos (retrieval aprendido), (c) ensamblado. Ahí es donde
ganan las soluciones de nivel competición.
