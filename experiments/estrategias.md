# Estrategias para superar el plateau (récord actual 0.06824)

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

### 4. Señal explícita de rating predicho  ·  probada como feature, NO ayuda

**Estado (2026-09-10): probada como feature, revertida.** `models/rating.py` (ALS de
feedback explícito `mu + b_u + b_i + p_u·q_i` sobre los ratings 1–10, RMSE in-sample 0.97
vs 1.82). Features `rating_predicho_candidato` + `rating_medio_usuario` (el sesgo del usuario
aislado — lo único que faltaba, `score_popularidad` ya es el sesgo del libro). Test pareado
seed 42: **+1,67 σ** con candidatos `nfadef`, pero **−2,34 σ / P=0.005** con `nfa=500`
(config de producción, `rating_predicho_candidato` sola) — no es borderline, es
**significativamente dañina** a la config real. El +1,67 σ era ruido dependiente de config.
Interpretación: el ranking de "qué se lee próximo" ya está dominado por señal
colaborativa/recencia; "le pondría ≥8" no discrimina *cuál* next-read elige un usuario (lee
muchos libros que ratearía 7–9), y a `nfa=500` se vuelve un casi-constante que empuja al
ranker lejos de la señal buena. Ver el detalle en `estado_del_arte.md`. Como **fuente de
candidatos** tiene aún menos chance — las 7ª fuentes fallaron 3 veces.

### 5. Retrieval por grafo (difusión multi-hop)  ·  APLICADA como feature (récord 0.06824)

**Estado (2026-09-10): probada como feature, adoptada con confianza modesta.**
`score_difusion_candidato` (`ranker.py`, `MIN_COREAD_PPR`/`K_DIFUSION`/`ALPHA_DIFUSION`):
`Σ_{t=1}^{2} 0,85ᵗ (Hᵗ)[u,j]` sobre el grafo de co-lectura **podado** y row-normalizado, con
`H⁰` = historial binario del usuario. Es la extensión 2-hop de `score_coleido` (1 hop).

- **La poda importa**: el grafo crudo tiene 73M aristas y a 1 hop ya alcanza el 43% del
  catálogo — sin podar, la difusión solo difumina y cuesta ~74s/1000 usuarios. Podando a
  `≥8` co-lectores (2,1% de las aristas, asociaciones con evidencia real) baja a
  ~3,5s/1000. Con `K=2` en vez de 3.
- **Feature-only** (candidatos de las 6 fuentes, sin cambiar el pool). Test pareado seed 42
  `nfa=500`: **+1,17 σ, P=0.88** (borderline, por debajo del ~1,5 σ que suele pedirse).
  Kaggle 0.06753 → **0.06824** (+0,0007, chico — dentro del SE de una submission por sí
  solo, pero **misma dirección** que el pareado, a diferencia de seed-bag). Se adopta por
  eso; confianza modesta.
- **Como fuente de candidatos** (top-N por `score_difusion`): no probado; las 7ª fuentes
  fallaron 3 veces, pero acá la señal *sí* mostró algo, así que queda como posible
  siguiente paso.

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
4. ~~estrategia 4 — rating predicho (feature)~~ **probada: +1,67 σ a `nfadef` pero −0,35 σ
   a `nfa=500`, revertida.** "Le pondría ≥8" no discrimina *cuál* next-read elige el usuario.
5. ~~estrategia 5 — difusión 2-hop (feature)~~ **APLICADA: pareado +1,17 σ, Kaggle 0.06753 →
   0.06824.** Efecto chico, confianza modesta, pero ambos instrumentos coinciden. Pendiente:
   probarla como **fuente de candidatos** (top-N por `score_difusion`) — es la única señal
   nueva que mostró algo.
6. **Lo que queda**: estrategia 6 (BERT4Rec / masked-item) como fuente de candidatos +
   feature de "interés actual" — mayor ceiling, mayor costo/riesgo (461k interacciones es
   poca data para un transformer). **Casi todo lo barato se probó**; lo demás son builds
   grandes de payoff incierto.

**Observación meta**: el proyecto había sobre-invertido en el *set de features* del reranker
(y esta ronda lo confirmó: LMF-feature, corroboración, relativas, rating predicho — todas
ruido; solo la difusión 2-hop movió algo, y apenas). El lever que rindió de verdad fue
**más señal de entrenamiento** (estrategia 1). Sigue abierto sin idea barata clara: (b)
calidad del pool de candidatos (retrieval que traiga
candidatos *distinguibles*, no solo más recall) y (c) ensamblado de familias distintas.
