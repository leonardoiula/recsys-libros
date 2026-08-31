# Modelo actual: ranker de dos etapas (v3, 26 features)

Resumen autocontenido del modelo de referencia del proyecto, pensado
para poder evaluarlo "desde afuera" (¿vale la pena seguir con LightGBM,
o conviene probar algo más orientado a sistemas de recomendación?) sin
tener que reconstruir el contexto leyendo todo `bitacora.md`. No repite
el *por qué* de cada decisión — eso está en `decisiones.md`/`bitacora.md`
— acá solo describe *qué* es el modelo tal como está hoy en el código.

Récord actual: **0.04855 en Kaggle** (NDCG@20), confirmado
2026-08-31. Código: `src/recsys/models/ranker.py`.

## Idea básica: candidatos + reranking

No es un solo modelo, son dos etapas:

1. **Generación de candidatos**: tres fuentes separadas (un modelo de
   filtrado colaborativo + dos baselines de popularidad) proponen,
   cada una, hasta 150 libros por usuario. Se unen sin duplicar
   (`n_por_fuente=150` en el código).
2. **Reranking**: un `LGBMRanker` (LightGBM, gradient boosted trees)
   toma la unión de esos candidatos, cada uno con 26 features
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

**Total de candidatos por usuario**: unión de las 3 fuentes (hasta 450,
deduplicados) — cada candidato queda marcado con `en_<fuente>` (1/0) y
su `score_<fuente>`/`rank_<fuente>` si esa fuente lo propuso, `0`/sentinel
si no.

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

## Features (26 en total)

Catálogo completo con el detalle de cada una en `experiments/features.md`.
Resumen agrupado:

- **Candidatos de ALS** (3): `score_als`, `rank_als`, `en_als`.
- **Candidatos de popularidad global** (3): `score_popularidad`,
  `rank_popularidad`, `en_popularidad`.
- **Candidatos de popularidad por género preferido** (3): `score_genero`,
  `rank_genero`, `en_genero`.
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
| Ranker (26 features) | 0.110112 ± 0.003411 | **0.04855 (récord actual)** |

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
  veces dio una mejora menor que el ruido. Agregar features de dominio,
  en cambio, sí movió la aguja varias veces (autor/año/género/recencia:
  +15.3% en Kaggle; género macro + tamaño de editorial: +3.7-4.5% cada
  una; señales cruzadas lector↔libro: +0.5%). Esto sugiere que el
  cuello de botella hasta ahora fue **señal/features**, no la capacidad
  del modelo de reranking en sí.
- Esta arquitectura (candidatos de ALS + reranking supervisado) ya es
  un patrón de recsys real, no un modelo genérico de ML aplicado a la
  fuerza. Si se evalúa un cambio de paradigma, las alternativas típicas
  para pensar (sin que esto sea una recomendación) suelen ser: modelos
  híbridos colaborativo+contenido nativos (ej. LightFM), factorization
  machines, modelos de dos torres/embeddings aprendidos end-to-end, o
  modelos secuenciales que usen el orden temporal del historial (ej.
  GRU4Rec/SASRec) — el dataset tiene 16 años de historial con fecha por
  interacción, señal temporal que hoy solo se usa de forma agregada
  (recencia, año de edición), no secuencial.
