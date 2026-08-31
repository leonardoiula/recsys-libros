# Modelo actual: ranker de dos etapas (v3, 29 features)

Resumen autocontenido del modelo de referencia del proyecto, pensado
para poder evaluarlo "desde afuera" (¿vale la pena seguir con LightGBM,
o conviene probar algo más orientado a sistemas de recomendación?) sin
tener que reconstruir el contexto leyendo todo `bitacora.md`. No repite
el *por qué* de cada decisión — eso está en `decisiones.md`/`bitacora.md`
— acá solo describe *qué* es el modelo tal como está hoy en el código.

Récord actual: **0.05140 en Kaggle** (NDCG@20), confirmado
2026-08-31. Código: `src/recsys/models/ranker.py`.

## Idea básica: candidatos + reranking

No es un solo modelo, son dos etapas:

1. **Generación de candidatos**: cuatro fuentes separadas (un modelo de
   filtrado colaborativo + dos baselines de popularidad + libros de
   autores ya leídos) proponen, cada una, hasta 150 libros por usuario
   (`n_por_fuente=150` en el código). Se unen sin duplicar.
2. **Reranking**: un `LGBMRanker` (LightGBM, gradient boosted trees)
   toma la unión de esos candidatos, cada uno con 29 features
   (incluyendo el score/ranking que le dio cada fuente de la etapa 1),
   y aprende a reordenarlos mejor de lo que cualquier fuente individual
   hace sola.

La razón de este diseño: ninguna fuente por sí sola alcanzaba, y en vez
de ensamblar a mano (ej. promediar scores), se dejó que un modelo
supervisado aprenda cómo combinarlas — ver `decisiones.md` sección 6.

Es un patrón común en sistemas de recomendación de producción reales
("*retrieval + ranking*" / "*candidate generation + learning-to-rank*"),
no un reemplazo de un modelo de recsys por LightGBM: el filtrado
colaborativo (ALS) sigue siendo la fuente principal de candidatos;
LightGBM es la capa de reranking encima.

## Etapa 1: fuentes de candidatos

### ALS (filtrado colaborativo, `models/als.py`)

`implicit.als.AlternatingLeastSquares` sobre una matriz dispersa
usuario×libro, con el **rating explícito (1-10) usado directamente como
confianza** (no rating binarizado). Hiperparámetros (`fit_als` default,
confirmados como los mejores en Kaggle real, no solo en NDCG local):

| Hiperparámetro | Valor |
|---|---|
| `factors` | 128 |
| `regularization` | 0.1 |
| `iterations` | 20 |
| `alpha` | `None` (confianza = rating crudo, no `1+alpha*rating`) |
| `seed` | 42 |

Candidatos: `implicit.recommend` top-150 por usuario
(`filter_already_liked_items=True`, más un filtro explícito extra
contra libros ya leídos — ver docstring de `recomendar_por_usuario` en
`als.py`).

### Popularidad global (`models/popularity.py`)

Score bayesiano con shrinkage: `score = (n/(n+C))·avg_rating + (C/(C+n))·m`,
con `C = n.mean()` (calculado sobre los datos recibidos, nunca
hardcodeado) y `m` = rating promedio global. Candidatos: top-150 del
ranking global.

### Popularidad por género preferido del usuario (`models/popularity_segmentada.py`)

Mismo score bayesiano, pero calculado *solo* con las interacciones del
género literario que más leyó cada usuario (52 categorías granulares
normalizadas). Candidatos: top-150 dentro de ese género.

### Libros de autores ya leídos (4ª fuente, agregada 2026-08-31)

Para cada autor que el usuario ya leyó, hasta 20 libros sin leer de ese
autor (`n_por_autor=20`), rankeados por el score de popularidad
**global** (no se refittea un score bayesiano por autor -- la mayoría
tiene muy pocas interacciones para un shrinkage propio confiable).
Motivada por medir que 28.6% de los targets de validación son de un
autor ya leído -- señal que antes solo existía como *feature*
(`en_autor_leido`/`n_libros_autor_leidos`, ver más abajo) y nunca
proponía candidatos nuevos por sí sola. Topeada además a `n_por_fuente`
(150) candidatos **totales** por usuario, priorizando los autores que
más leyó -- sin este tope, usuarios con cientos de autores leídos
llegaban a más de 5.000 candidatos de esta fuente sola (problema real
de memoria encontrado al implementarla, no solo teórico).

**Total de candidatos por usuario**: unión de las 4 fuentes (media
~486, hasta ~600 para usuarios pesados) — cada candidato queda marcado
con `en_<fuente>` (1/0) y su `score_<fuente>`/`rank_<fuente>` si esa
fuente lo propuso, `0`/sentinel si no.

## Etapa 2: LightGBM (`LGBMRanker`)

| Hiperparámetro | Valor |
|---|---|
| `objective` | `lambdarank` |
| `num_leaves` | 31 |
| `learning_rate` | 0.05 |
| `n_estimators` | 200 |
| `random_state` | 42 |

Son **hiperparámetros conservadores, elegidos a propósito sin tunear
agresivo**: el tuneo con `optuna` se probó **tres veces** sobre bases de
features distintas (`scripts/tune_ranker.py`) y las tres veces dio una
mejora menor que el ruido entre seeds — nunca se adoptó. En producción
nunca se pasa `eval_set`, así que el early stopping soportado por
`fit_ranker` no se usa en la práctica: siempre entrena las 200 rondas
completas.

## Qué variable optimiza el ranker

Hay dos niveles distintos, fáciles de confundir:

1. **Label de entrenamiento (`y`)**: binario por candidato — `1` si es
   el libro que el usuario realmente leyó después (la interacción más
   reciente, reservada aparte como `train_ranker`), `0` para el resto
   de sus candidatos. Los candidatos de un mismo usuario forman un
   *grupo* (`group` en `LGBMRanker.fit`) — `lambdarank` optimiza (una
   aproximación diferenciable de) el orden **dentro de cada grupo**, no
   accuracy/logloss de cada fila por separado.
2. **Métrica real del proyecto**: NDCG@20 — descuento `1/log2(pos+2)`
   normalizado por IDCG, evaluado con split leave-one-out temporal
   (última interacción de cada usuario por fecha). Localmente se mide
   con validación cruzada sobre 3 seeds (42, 7, 123); el número que
   manda es el de Kaggle real, sobre un conjunto de test oculto
   (`data/raw/ejemplo.csv` tiene 832 usuarios de referencia del mismo
   tipo).

`lambdarank` es una aproximación a NDCG (no lo optimiza exactamente),
pero está diseñado específicamente para eso — a diferencia de entrenar
un clasificador binario normal (que optimizaría separar 1s de 0s, no
ordenar).

## Split de tres niveles (por qué no es un split simple)

Como el ranker usa el score de ALS/popularidad como *features*, esos
scores no pueden salir de datos que el ranker vea como etiqueta (si no,
memoriza en vez de aprender a combinar señales). Por eso:

- `train_candidatos`: fitea ALS + popularidad + popularidad por género
  (las 3 fuentes de la etapa 1) y calcula las features auxiliares.
- `train_ranker`: etiquetas conocidas (última interacción de cada
  usuario dentro de este tramo) para entrenar el `LGBMRanker`.
- `test_final`: hold-out aislado para medir NDCG@20 del pipeline
  completo (solo en evaluación local, no existe en producción).

## Features (29 en total)

Catálogo completo con el detalle de cada una en `experiments/features.md`.
Resumen agrupado:

- **Candidatos de ALS** (3): `score_als`, `rank_als`, `en_als`.
- **Candidatos de popularidad global** (3): `score_popularidad`,
  `rank_popularidad`, `en_popularidad`.
- **Candidatos de popularidad por género preferido** (3): `score_genero`,
  `rank_genero`, `en_genero`.
- **Candidatos de autores ya leídos** (3, ronda más reciente):
  `score_autor_candidato`, `rank_autor_candidato`, `en_autor_candidato`
  -- distintas de `en_autor_leido`/`n_libros_autor_leidos` (historial,
  sin importar qué fuente propuso el candidato). Un test pareado mostró
  que estas 3 features específicas no aportan por sí solas (0.44 sigma,
  no significativo) -- la mejora real viene de los candidatos nuevos
  que trae la fuente, no de trackear explícitamente que vinieron de
  ahí. Se mantienen igual, no restan.
- **Volumen bruto** (2): `n_interacciones_libro`, `n_interacciones_usuario`.
- **Autor** (2): `en_autor_leido`, `n_libros_autor_leidos`.
- **Año de edición** (1): `anio_edicion_dif` (vs. promedio de lectura
  del usuario).
- **Diversidad de género** (1): `n_generos_distintos_usuario`.
- **Recencia** (1): `dias_desde_ultima_interaccion_usuario`.
- **Co-lectura ítem-ítem** (1): `score_coleido`.
- **Editorial** (3): `en_editorial_leida`, `n_libros_editorial_leidos`
  (historial del usuario), `n_libros_editorial_catalogo` (tamaño del
  catálogo de la editorial, propiedad del libro).
- **Resumen/texto** (1): `sim_resumen_historial` (TF-IDF + coseno).
- **Macro-género** (2): `popularidad_genero_macro_candidato`,
  `frecuencia_genero_macro_usuario` (10 familias de dominio, pooled).
- **Señales cruzadas lector↔libro** (3, ronda más reciente):
  `popularidad_genero_lector_candidato` (popularidad segmentada por
  género *declarado* del lector — Mujer/Hombre/desconocido, no el
  género literario), `frecuencia_genero_macro_por_genero_lector`
  (afinidad de *cohorte* por macro-género, no historial individual),
  `edad_lector_al_publicarse` (`anio_edicion − nacimiento`).

Todas se calculan **solo con `train_candidatos`**, salvo la metadata
estática del libro (autor/editorial/año/resumen, que no depende del
split).

## Números actuales (para comparar)

| | NDCG@20 local (CV 3 seeds) | NDCG@20 Kaggle |
|---|---|---|
| ALS solo | 0.094406 ± 0.001359 | 0.03864 (mejor config ALS confirmada) |
| Ranker (26 features, 3 fuentes de candidatos) | 0.109735 ± 0.003719 | 0.04855 |
| Ranker (29 features, 4 fuentes -- +autor ya leído) | 0.117495 ± 0.002562 | **0.05140 (récord actual)** |

La 4ª fuente (autor) dio la mejora más grande y mejor validada de la
sesión: recall del set de candidatos 0.394→0.445 (medido con
`scripts/recall_candidatos.py`), CV positivo en los 3 seeds por un
margen 5-10x mayor que cualquier resultado anterior, y +5.9% real en
Kaggle -- ver `decisiones.md` sección 12 y `bitacora.md`.

Ojo: el número local **sobreestima bastante** el de Kaggle en términos
absolutos (investigado en `bitacora.md` — principalmente varianza de
muestra chica en Kaggle, ~832 usuarios, con una métrica donde la
mayoría de los usuarios da NDCG=0). Comparar *dentro* del proyecto
(ranker vs. ALS solo, o una versión de features vs. otra) es más
confiable que comparar los valores absolutos entre sí.

## Contexto para pensar el cambio de paradigma

Algunos datos neutrales, sin recomendar una dirección — la decisión de
si conviene seguir invirtiendo en features sobre este pipeline o probar
algo estructuralmente distinto es tuya:

- **Dataset**: 461,408 interacciones, 11,285 lectores (10,673 con
  actividad), 128,743 libros en catálogo (48,137 con al menos una
  interacción). Sparsity de la matriz usuario-libro: **99.91%**.
- **Patrón de esta sesión y las anteriores**: tunear hiperparámetros de
  LightGBM se probó 3 veces (bases de features distintas) y las 3
  veces dio una mejora menor que el ruido. Agregar features de dominio
  movió la aguja varias veces pero con retornos decrecientes
  (autor/año/género/recencia: +15.3% en Kaggle; género macro + tamaño
  de editorial: +3.7-4.5% cada una; señales cruzadas lector↔libro:
  +0.5%, luego mostrado como ruido con un test pareado -- ver más
  abajo). **Atacar el generador de candidatos en vez de seguir sumando
  features rompió ese patrón**: la 4ª fuente de candidatos (autor ya
  leído) dio +5.9% en Kaggle, la segunda mejora más grande del proyecto,
  con la validación estadística más sólida de todas (recall medido
  aparte, CV positivo en los 3 seeds por un margen enorme, test pareado
  que aisló el mecanismo real). Confirma que el cuello de botella real
  es el **recall del generador de candidatos**, no la capacidad del
  modelo de reranking ni la cantidad de features.
- Esta arquitectura (candidatos de ALS + reranking supervisado) ya es
  un patrón de recsys real, no un modelo genérico de ML aplicado a la
  fuerza.

---

## Recomendación: ¿cambiar de paradigma? (análisis dedicado, 2026-08-31)

Encargado a un modelo separado (Opus) con instrucciones de medir en vez
de opinar en abstracto — corrió diagnósticos reales contra el código
del proyecto (scripts quedaron en el scratchpad de la sesión, no en el
repo). Resumen del dictamen:

**Recomendación: no migrar de paradigma. El cuello de botella medible
no es el modelo de reranking ni el de filtrado colaborativo — es el
generador de candidatos.**

### El embudo pierde más de lo que el reranking puede recuperar

Descomponiendo el NDCG local (seed 42, 8.904 usuarios de `test_final`):

| | valor |
|---|---|
| Target presente en `train_candidatos` (techo absoluto de cualquier generador) | **0.931** |
| Recall del set de candidatos actual (unión de 3 fuentes, ~419/usuario) | **0.394** |
| NDCG@20 del ranker | 0.1063 |
| Fracción del techo que captura el ranker | 0.270 |

El NDCG es aproximadamente `0.394 × 0.270`. Las últimas rondas de
features movieron el segundo factor menos del 2%; el primero está a
0.394 de un techo de 0.931 — ahí está el margen real.

Ese factor es muy sensible a cambios simples y baratos:

| set de candidatos | recall | candidatos/usuario |
|---|---|---|
| unión actual, `n_por_fuente=150` | 0.414 | 705 |
| unión actual, `n_por_fuente=500` | 0.559 | 2.292 |
| + 4ª fuente "libros de autores ya leídos" (top-20/autor), n=150 | 0.507 | 936 |
| unión + autor, n=500 | **0.620** | 2.487 |
| solo la fuente autor | 0.268 | 268 |

**28,6% de los targets de validación son libros de un autor que el
usuario ya leyó** — autor hoy solo existe como *feature*
(`en_autor_leido`/`n_libros_autor_leidos`), que solo puede activarse si
ALS o popularidad ya trajeron el libro como candidato. No está en el
generador. Si la eficiencia de ranking (0.27) se mantuviera, recall
0.62 daría NDCG ~0.167 (+57%) — no se va a sostener igual con más
candidatos (ranking más difícil), pero incluso un tercio de eso es 5-10x
más grande que cualquier cosa medida en las últimas 4 rondas de esta
sesión.

### Los modelos secuenciales quedan descartados por los datos, no por costo

Se midió la estructura temporal antes de recomendar nada:

- **67,5% de los gaps entre interacciones consecutivas son 0 días**;
  gap mediano 0, p75 = 4 días.
- Span temporal mediano por usuario: 26 días (mediana 9 interacciones)
  — la mayoría lee "todo junto".
- **45,7% de los usuarios tiene más de una interacción en su fecha
  máxima** — el propio ground truth ("el próximo libro") se decide hoy
  por un desempate aleatorio dentro de un lote del mismo día.

SASRec/GRU4Rec modelan el *orden* de la secuencia; acá el orden
intra-día es arbitrario y es la mayoría de la señal — pedirle a un
modelo secuencial que aprenda un orden que el dataset no tiene. Lo que
sí hay es **coherencia local sin orden** (mismo autor que la
interacción inmediata anterior: 12,2% vs. 4,4% al azar, 2,8x; mismo
género: 30,3% vs. 22,4%, 1,35x) — se captura con features ponderadas
por recencia sobre el último lote, no con un modelo secuencial. Ojo:
popularidad *reciente* (ventana global) es peor que all-time (recall en
top-150: 0.194 vs. 0.245) porque los targets están repartidos en 16
años (solo 12% cae en 2024) — cada usuario tiene su propio "ahora"; si
se explora esto, la ventana debe ser relativa al propio usuario.

**LightFM / dos torres / factorization machines reemplazarían a ALS,
que no es el problema.** Su valor agregado sobre ALS es meter metadata
en el embedding — metadata que ya está en las 26 features del ranker,
donde LightGBM la usa con más libertad (no lineal). Efecto esperado:
del orden de lo que se viene midiendo, por debajo de la resolución del
instrumento actual. Costo de implementación: alto. Mala relación
costo/beneficio comparado con atacar el generador de candidatos.

### Hallazgo incómodo: las últimas 3 "confirmaciones" podrían ser ruido

Se reimplementó la comparación de las 3 señales cruzadas (26 vs. 23
features) sobre el mismo contexto/seed, con un **test pareado por
usuario** (en vez de comparar promedios de 3 seeds independientes):

```
NDCG@20 26 features: 0.106833      NDCG@20 23 features: 0.106919
diferencia media pareada: -0.000086   SE pareado: 0.000770
bootstrap 95% CI: [-0.001499, +0.001411]     P(diferencia > 0) = 0.46
usuarios donde cambia el NDCG: 13.3%  (mejoran 573, empeoran 611)
```

Con ~4,9x más precisión que el criterio actual (comparar contra el
desvío entre 3 seeds), el efecto de las 3 features nuevas es
**estadísticamente cero** (mejoran y empeoran usuarios en proporciones
similares — churn, no mejora real). Y el "+0.5% en Kaggle" que
confirmó esta ronda son 0.00024 absolutos = 0.04 desvíos estándar del
ruido de Kaggle. Los tres "casos límite" confirmados esta sesión
(género macro +4.5%, editorial +3.7%, señales cruzadas +0.5%) son 0.3,
0.3 y 0.04 SE respectivamente — **ninguno es evidencia estadística
real**, pese a estar documentados como "confirmados en Kaggle". Lo que
veníamos leyendo como "retornos decrecientes de las features de
dominio" parece ser, en buena parte, el momento en que el efecto real
cayó por debajo de la resolución del instrumento de medición.

Esto no significa que esas features estén mal (no hay evidencia de que
*empeoren* tampoco) — significa que el criterio de decisión usado
hasta ahora (comparar contra el desvío entre 3 seeds, o confirmar con
una sola submission de Kaggle) no tiene poder estadístico suficiente
para distinguir señal real de ruido en este rango de efectos.

### Next steps, en orden de prioridad

1. ~~**Fuente de candidatos por autor** (4ª fuente): top-20 libros por
   popularidad de cada autor leído por el usuario, filtrando leídos.~~
   **✅ HECHO Y CONFIRMADO (2026-08-31)**: implementada con un tope
   total de `n_por_fuente` candidatos por usuario (necesario -- sin
   tope, usuarios con muchos autores leídos generaban miles de
   candidatos y rompían la corrida por memoria). Recall 0.394 → 0.445.
   CV positivo en los 3 seeds, +5.9% en Kaggle -- la segunda mejora más
   grande de todo el proyecto. Ver `decisiones.md` sección 12.
2. **Subir `n_por_fuente` de 150 a 500** y medir recall del set de
   candidatos y NDCG por separado (cambio de una línea; sube el costo
   de cómputo de `preparar_pipeline` de ~300s a ~15-20 min/seed).
   Combinado con (1): recall medido en 0.620 en el diagnóstico
   original (antes de agregar el tope de memoria a la fuente de autor
   -- revalidar el número exacto ahora que esa fuente está topeada).
3. **Item-item kNN / EASE^R como 5ª fuente de candidatos** (no solo
   como feature): ya se calcula `cooc = X.T @ X` para `score_coleido`
   en ~3s; normalizado y usado como generador (top-N vecinos del
   historial reciente) suele empatar o ganarle a ALS en datos
   implícitos así de dispersos, y trae candidatos que ALS no trae.
   Antes que cualquier LightFM/dos torres.
4. **Decidir sobre etapa 1 con recall del set de candidatos, no con
   NDCG del pipeline completo** — se mide en 17-76s por variante (vs.
   ~300s de contexto completo), sin gastar submissions. Solo cuando el
   recall sube vale la pena correr el pipeline completo.
5. **Features ponderadas por recencia** (última fecha / últimas N
   interacciones) de las señales que hoy poolean todo el historial
   parejo (autor, editorial, co-lectura, `sim_resumen_historial`) — la
   forma correcta de aprovechar la señal temporal en estos datos, dado
   que un modelo secuencial no aplica. Va *después* de (1)-(3): es un
   factor de eficiencia de ranking, no de recall.
6. **Revisar el refit de etapa 1 en `submit.py`**: hoy fitea ALS/
   popularidad sobre `train_candidatos` (sin la interacción más
   reciente de cada usuario, ~10.673 interacciones, la parte más
   informativa) para generar los candidatos finales de producción. La
   práctica estándar es refitear etapa 1 sobre todos los datos después
   de entrenar el ranker sobre el split — testeable localmente sin
   leakage (comparar señales fitteadas en `train_candidatos` vs.
   `train_candidatos_full`, evaluando siempre en `test_final`).
7. **No volver a tunear LightGBM.** 3/3 intentos fallidos, y ahora se
   entiende por qué: el efecto que se buscaba (~0.0013) está por
   debajo del error estándar no pareado (0.0038) con el que se medía.

---

## Recomendación: ¿incorporar agentes de IA al flujo de trabajo? (mismo análisis)

**El 80% de la ganancia de velocidad no viene de agentes de IA — viene
de cachear el contexto de candidatos y paralelizar los seeds, que es
ingeniería determinista.** Los sub-agentes valen para tres cosas
puntuales: implementar features en paralelo (worktrees) mientras corre
una evaluación, auditar leakage antes de quemar una corrida, y
replicar un resultado de forma independiente antes de gastar una
submission. Envolver un `for seed in seeds` en un LLM no agrega nada.

### Lo que no es un problema de agentes

- **Los 3 seeds corren en serie** (`scripts/evaluate_ranker.py` es un
  `for seed in SEEDS`) pudiendo correr en paralelo con 2-3 workers de
  proceso (~20-25 min → ~10 min) — pero antes de paralelizar hay que
  subir el archivo de paginación y bajar el pico de memoria del TF-IDF
  (`tfidf_norm[filas_texto].multiply(...)`, procesarlo por chunks): dos
  corridas simultáneas dieron `MemoryError` en la máquina de esta
  sesión.
- **El contexto se recalcula para cada ablation sin hacer falta.** Ya
  existe la separación `preparar_pipeline` (~300s, no depende de
  hiperparámetros de LightGBM) / `evaluar_con_params` (~22s). Falta el
  paso obvio: el contexto también sirve para comparar *subconjuntos de
  features* del ranker, no solo hiperparámetros — el test pareado de la
  sección anterior entrenó 26 y 23 features sobre el mismo contexto en
  ~1 minuto en vez de repetir dos corridas completas de ~20-25 min cada
  una. Cacheando el contexto a disco por seed (con **todas** las
  columnas candidatas, incluidas las experimentales), cada ablation
  pasa de 20-25 min a ~1 min.
- **El criterio de decisión actual tiene poco poder estadístico** (ver
  hallazgo de arriba). Dos arreglos concretos, sin agentes:
  - Test pareado por usuario en vez de "mejora > desvío entre seeds":
    medido, SE no pareado 0.003775 vs. SE pareado 0.000770 → ~4,9x más
    poder, efectos detectables desde ~0.0015 en vez de ~0.0075.
  - Ponderar la evaluación local por la población real de Kaggle: los
    832 usuarios de `ejemplo.csv` tienen mediana de 95 interacciones;
    la evaluación local promedia 8.904 usuarios con mediana 9 — un
    usuario con 200+ interacciones pesa ~5,3x más en Kaggle que en el
    promedio local sin ponderar. La bitácora ya hizo este reponderado a
    mano una vez para ALS; falta meterlo en el pipeline de CV.

### Dónde sí sirven sub-agentes (con contexto cacheado, el cuello de
botella pasa a ser implementar/auditar, no evaluar)

1. **Worktrees paralelos para ideas independientes** (ej. fuente por
   autor / item-item kNN como fuente / features ponderadas por
   recencia): 2-3 sub-agentes implementando en paralelo contra el mismo
   contexto cacheado, revisión humana de los diffs, una sola corrida
   determinista evalúa todas las variantes en conjunto y pareadas.
2. **Auditor de leakage dedicado y adversarial**: un sub-agente cuyo
   único trabajo es leer el diff de una feature nueva y buscar uso de
   datos posteriores al corte — este proyecto ya tuvo dos episodios
   reales de este tipo (split aleatorio con leakage temporal: NDCG
   0.260 → 0.123 al corregirlo; necesidad del split de tres niveles).
   Corrida perdida: 20-25 min. Auditor: ~1 min.
3. **Replicación independiente antes de gastar una submission**: un
   sub-agente que no vio la implementación reimplementa la *medición*
   (no la feature) y reporta el número pareado — es exactamente lo que
   se hizo en este análisis, y encontró que el "récord confirmado" de
   esta sesión es indistinguible de ruido.

### Dónde no

- Envolver la ejecución de los 3 seeds en agentes (es
  `joblib`/`multiprocessing`, no un problema de LLM).
- Decidir si se gasta una submission, o elegir taxonomías de dominio
  (queda humano — y con la agenda de arriba se necesitan *menos*
  submissions, no más).
- Interpretar resultados límite con más agentes: el modo de falla
  histórico del proyecto no es falta de análisis, es **exceso de
  narrativa sobre ruido** (los tres casos límite, con explicación
  causal convincente cada uno, resultaron ser efecto real cero). Un
  enjambre de agentes produce más historias plausibles por hora, no
  más señal — la defensa correcta es estadística (test pareado,
  población ponderada, umbrales pre-registrados), no más razonamiento.
- Tuneo de hiperparámetros paralelizado con agentes: está muerto por
  evidencia (3/3), paralelizarlo no cambia eso.

### Next steps concretos

1. Cachear el contexto de `preparar_pipeline` a disco por seed con
   **todas** las columnas de features (incluidas las experimentales);
   que `fit_ranker`/la predicción reciban la lista de features como
   parámetro en vez de leer `ranker.FEATURES` global. Habilita
   ablations de ~1 min.
2. Cambiar el criterio de decisión en `evaluation.py`: NDCG por usuario
   + diferencia media pareada + CI bootstrap, y NDCG ponderado por la
   distribución de actividad de `ejemplo.csv` reportado al lado del
   plano. Pre-registrar el umbral (ej. adoptar si el CI pareado del
   95% excluye el 0 en la población ponderada).
3. Subir el archivo de paginación, bajar el pico de memoria del TF-IDF,
   recién entonces paralelizar los seeds con 2-3 workers.
4. Con eso andando, abrir la agenda del generador de candidatos con
   2-3 worktrees paralelos, auditoría de leakage por sub-agente, y una
   sola submission al final del bloque — no una por idea.
