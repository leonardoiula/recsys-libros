# Estado del arte — recsys-libros

Punto de entrada para retomar el proyecto: cómo funciona el sistema, cómo se valida, qué se
probó y no funcionó, y el problema abierto. Es lo único que hace falta leer para retomar
contexto. El razonamiento ronda por ronda hasta 2026-09-07 está **congelado** en
`experiments/legacy/` (ver `experiments/legacy/README.md`); lo posterior está en
`experiments/log.csv` (una fila por corrida).

**Récord: 0.06824 de NDCG@20 en Kaggle** (2026-09-10).

**IMPORTANTE (2026-09-24)**: los NDCG *locales* de este documento (split por usuario) sobreestiman
Kaggle ~2-2.7× por una fuga temporal poblacional en `split_train_val`, no solo ruido de muestra
chica — ver "Validación con corte temporal global" más abajo antes de interpretar cualquier
número local como "cerca de lo que daría en Kaggle". Siguen siendo válidos para comparar
configs *entre sí* (el gatekeeper real sigue siendo el pareado + CV), no como estimación
absoluta.

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
   −2,34 σ a `nfa=500`) — **`comparar_generadores_pareado.py` no lo pasaba en absoluto hasta
   2026-09-24** (bug real, no solo un riesgo teórico: cualquier fuente nueva evaluada con ese
   script, LMF/usuario-usuario incluidas, pudo estar comparándose a `nfadef` sin que nadie lo
   notara). `REFIT_PARA_TEST=True` (mismo día) ahora es el default en los 4 scripts de este
   punto y en `evaluate_ranker.py`: antes medían contra una etapa 1 sin el refit que
   `submit.py` siempre hace, una diferencia ya confirmada en +12,4 % NDCG local
   (`comparar_refit_etapa1.py`, 2026-09-02) y nunca vuelta a aplicar hasta ahora. Cuesta ~2×
   el armado del contexto.
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
   de generalización. Cierra solo una PARTE de la brecha con Kaggle (ver sección siguiente) —
   no descarta que un config sea distinto en dirección, pero no hay que leerlo como "esto ya
   explica por qué el local es más alto que Kaggle".
6. **`evaluate_global_cutoff.py`** (nuevo, 2026-09-24): NDCG@20 bajo un split de corte
   temporal **global** en vez de por usuario — ver "Validación con corte temporal global"
   más abajo. Más lento (~7-8 min) y con menos usuarios de test (811) que el pareado/CV, así
   que no reemplaza al gatekeeper del punto 1 para decisiones finas; sirve para chequear si
   una señal que gana localmente sigue ganando bajo un protocolo sin fuga temporal, sobre
   todo para ideas que el punto 1 alguna vez aprobó pero Kaggle rechazó (LMF, usuario-usuario).

**Splits**: leave-one-out **temporal** (`split_train_val`, `n_val=1`) — un split aleatorio
filtra futuro y sobreestima ~2×. Para el ranker, **3 niveles**: `train_candidatos` (fitea
etapa 1) / `train_ranker` (etiquetas) / `test_final` (hold-out, solo local). **OJO**: este
split sigue siendo por-usuario, así que todavía tiene la fuga temporal poblacional descripta
abajo — sigue siendo el instrumento correcto para el gatekeeper del punto 1 (tiene ~10× el N
de usuarios y por lo tanto mucho más poder estadístico), pero su NDCG *absoluto* no es
comparable a Kaggle sin corregir.

---

## Validación con corte temporal global (nuevo, 2026-09-24)

**Hallazgo**: la brecha NDCG local-vs-Kaggle (0.137854 local vs 0.06753 Kaggle para la misma
config, por ejemplo) no es principalmente ruido de muestra chica de Kaggle (SE≈0.0065 solo
explica diferencias <~0.02) ni solo composición de actividad (reponderar por la actividad de
`ejemplo.csv` deja **0.106754 vs 0.06316 real, +69% de gap residual** — medido recargando el
contexto cacheado de la ronda BM25+nfa=500 y reponderando con `evaluar_ndcg_ponderado_por_actividad`).
La causa que sí cierra la mayor parte de la brecha: **`split_train_val` es leave-one-out POR
USUARIO, sin ningún corte de fecha global** — ALS/popularidad/co-lectura se fitean sobre TODA
la población de `train_candidatos` sin restricción temporal. Para un usuario mediano de la
validación local (su propio corte cae ~2018-01-01), **el 30.9% de TODO el dataset es
cronológicamente posterior a su punto de validación** — señal de "futuro" (tendencias,
libros que se pusieron de moda después) que ningún sistema prospectivo real tendría. Kaggle
evalúa específicamente a los usuarios activos hasta el final del rango de fechas del dataset
(mediana de última interacción visible 2024-10-25 en `ejemplo.csv`, donde solo el 0.5% del
dataset es posterior) — ahí esa fuga casi no existe. Es consistente que el gap relativo
post-reponderación (~+69%) sea casi idéntico para el ALS-solo de la era v2 (`log.csv` fila 8:
reponderado 0.064008 vs Kaggle 0.03864, +65.6%) y para el ranker de 40 features de hoy — un
artefacto del protocolo, no de cuánto aprendió un modelo en particular.

**`scripts/evaluate_global_cutoff.py`** (nuevo) mide el NDCG@20 bajo un split de corte
temporal **global** (`recsys.data.split_temporal_global`) en vez de por usuario: la etapa 1
se fitea SOLO con datos anteriores a un corte de calendario, sin importar de qué usuario sea
cada interacción — como un sistema prospectivo real. Ventana rodante de 5 cortes globales
(`2022-01 / 2023-01 / 2023-07 / 2024-01 / 2024-04 / 2024-07`, mismo espíritu que
`N_CORTES_RANKER` pero con fechas de calendario en vez de un corte relativo por usuario): cada
ventana `(cortes[i], cortes[i+1]]` fitea su PROPIA etapa 1 con datos `<= cortes[i]` y etiqueta
con la primera interacción de cada usuario en esa ventana — 4.471 etiquetas en total. El corte
final (2024-07-01) se eligió porque da una población de test (811 usuarios, 45.0% en el bucket
de actividad 100+) muy parecida a la de Kaggle (832 usuarios, 46.9% en 100+) sin haber
ajustado a mano para que coincidiera. Antes de generar los candidatos de test, la etapa 1 se
REFITEA sobre todo lo disponible hasta el corte final (mismo criterio que `submit.py`).

| Modelo | NDCG@20, corte GLOBAL (811 usuarios) | referencia |
|---|---|---|
| Popularidad global | 0.004365 | vs 0.01024-0.01558 Kaggle (v0/v1, otra era del proyecto) |
| ALS solo | 0.017508 | vs 0.03864 Kaggle (v2, sin BM25) |
| **Ranker (2 etapas, ventana rodante)** | **0.049061*** | **vs 0.06316-0.06824 Kaggle (config comparable)** |

\* Corrida única (seed=42, sin CV) de la primera vez que se armó esta ventana rodante.
**Resultó ser un outlier** — el CV de 3 seeds hecho más tarde el mismo día da
**0.046116 ± 0.000129** como número de referencia real (ver sección "CV multi-seed +
tuning de LightGBM bajo corte global" más abajo). Queda el 0.049061 acá tal cual salió
en su momento, por fidelidad histórica de esta sección -- para cualquier comparación
nueva, usar 0.046116.

El ranker sigue ganándole a ALS solo por un margen sano (+51% relativo) y ALS le sigue
ganando a popularidad (4×) — el orden se preserva, no es un pipeline roto. El número absoluto
(0.049) queda en el mismo orden de magnitud que Kaggle real por primera vez (vs 2-2.7× más
alto con el split por usuario) — evidencia fuerte de que el mecanismo identificado (fuga
temporal poblacional) es la explicación correcta de la brecha, no solo una hipótesis. Un
primer intento con un solo corte (T1=2022-01 → T2=2024-07, sin ventana rodante) había dado
solo 0.026522 -- **más bajo** que Kaggle, no más alto -- por dos motivos identificados y
corregidos: (a) sub-entrenamiento del ranker con apenas ~2.070 etiquetas, y (b) un gap de 2.5
años entre la etapa 1 con la que el ranker aprende (fiteada solo hasta T1) y la etapa 1
"fresca" con la que se lo evalúa (refiteada hasta T2) -- un desajuste de distribución que el
split por usuario nunca tiene (ahí train/refit difieren en un puñado de filas, nunca años).
La ventana rodante resuelve ambos a la vez. Sigue quedando un 22-28% de gap sin cerrar contra
Kaggle real -- candidato más probable: más densidad de cortes (esta ronda solo probó 1 vs 5
ventanas, sin punto intermedio para ver dónde plateaua, y el propio `N_CORTES_RANKER` por
usuario todavía mejoraba de 3 a 5 cortes).

**Bug real encontrado armando esto** (ya corregido, con test de regresión): pasar un string
de fecha a `pd.Timestamp(...)`/`pd.to_datetime(...)` sin `format="%d-%m-%Y"` explícito lo
parsea `MM-DD-YYYY` (convención US) en vez de `DD-MM-YYYY` (el formato real de la columna
`fecha` en todo el proyecto) — `"01-07-2024"` se leía como 7 de enero en vez de 1 de julio.
`split_temporal_global` fuerza el formato correcto cuando `fecha_corte` es un string; cualquier
código nuevo que reciba una fecha como parámetro debe hacer lo mismo.

**Qué implica para el resto del documento**: los NDCG locales de la tabla de arriba ("Récord y
progresión") siguen siendo válidos para lo que fueron diseñados — comparar configs *entre sí*
bajo el mismo protocolo, con mucho más poder estadístico que Kaggle o que
`evaluate_global_cutoff.py` (811 usuarios vs ~10.673) — pero su valor ABSOLUTO no es una
estimación de Kaggle. El patrón repetido de "sube el recall, NDCG local plano/positivo, Kaggle
regresión" (editorial, usuario-usuario, LMF, difusión-como-fuente — 4/4 intentos de 7ª fuente)
es compatible con que esas señales exploten específicamente el tipo de correlación temporal
poblacional que este protocolo elimina — candidatas naturales para re-testear con
`evaluate_global_cutoff.py` antes de asumir que están cerradas para siempre.

**Cómo correr**: `uv run python scripts/evaluate_global_cutoff.py` (~7-8 min, un solo seed —
no hay CV multi-seed todavía, es la limitación más importante de esta primera versión).
`scripts/comparar_global_pareado.py` (nuevo) hace el mismo tipo de test PAREADO por usuario
que `comparar_generadores_pareado.py`/`comparar_features_pareado.py`, pero sobre
`evaluate_global_cutoff.construir_y_evaluar` en vez del split por usuario — ~15 min por
comparación (dos configs completas, sin cache a disco). Acepta tanto `FUENTES_A`/`FUENTES_B`
(subconjuntos de `FUENTES_CANDIDATOS`) como `FEATURES_A`/`FEATURES_B` (subconjuntos de
`FEATURES`).

**Sesión 2026-09-24 — qué se aprendió usándolo para re-testear ideas viejas:**

- **LMF y usuario-usuario, rechazo confirmado con alta confianza.** Ver sus entradas en
  "Qué se probó y NO funcionó" más abajo — LMF en particular es el caso más claro: pareado
  por-usuario *incluso corregido* lo mostraba "casi positivo" (+1,56 σ), pero el corte
  global dio **-1,09 σ** — negativo. Confirma que un retriever *aprendido* que explota
  co-ocurrencia poblacional es exactamente el tipo de señal que la fuga temporal infla.
- **Las features "límite" (género macro, señales cruzadas lector↔libro, difusión)
  SÍ sostienen su valor** — sacarlas empeora bajo el corte global (+1,58 σ *a favor* de
  mantenerlas, más fuerte que el pareado por-usuario que las adoptó originalmente). La fuga
  temporal no infla todo por igual: perjudica selectivamente a **fuentes de candidatos**
  que aprenden patrones de co-ocurrencia poblacional (LMF, probablemente usuario-usuario),
  no a **features** que puntúan candidatos ya elegidos con señal demográfica/de
  contenido/de diseminación. **No hay que revertir nada del feature set de producción** —
  al contrario, esto lo confirma con más solidez de la que tenía.
- **Densidad de la ventana rodante — sin señal clara.** Duplicar los cortes (5→10,
  2022→2021, ~4.470→~7.960 etiquetas) **bajó** el NDCG (0.049061→0.045499) en vez de
  subirlo, al revés de lo que predeciría la analogía con `N_CORTES_RANKER`. Sin CV/pareado
  todavía (un seed, n=811) no alcanza para saber si es ruido o un efecto real — hipótesis
  más plausible: los cortes de `N_CORTES_RANKER` son siempre relativos al historial propio
  de cada usuario (nunca "viejos"), mientras que los cortes globales de 2021 sí lo son en
  términos absolutos. Quedó en 5 cortes (ya validado, más barato). Sigue quedando el
  22-28 % de gap contra Kaggle real sin explicación cerrada.

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
  Kaggle** (0.06017 vs 0.06149). **Re-testeada 2026-09-24** bajo el pareado con los dos bugs
  de protocolo corregidos (`REFIT_PARA_TEST=True` + `nfa=500`, ninguno estaba wireado en
  `comparar_generadores_pareado.py` la primera vez) para descartar que el rechazo original
  fuera un falso negativo — **rechazo confirmado**: 0,34 σ, P(mejora)=0,63, CI cruza cero,
  728 mejoran vs 723 empeoran (~50/50). Recall +1,5%, NDCG plano — mismo patrón que las
  otras 3. No es una historia de protocolo roto, la señal genuinamente no está. Revertida
  de nuevo en su totalidad. Ver `experiments/log.csv`, fila `2026-09-24,ranker,...,0.150768`.
- **Retrieval aprendido como 7ª fuente** (estrategia 2; `scripts/tune_retrievers.py` optimizó
  BPR/LMF/cosine-kNN/BM25-kNN por recall complementario a ALS — ganó **LMF**). Cableado:
  recall del set 0.5346 → 0.5547, CV +0,22 % (ruido), pareado **+1,48 σ**, **Kaggle 0.06164
  vs 0.06316 — regresión**. El recall sube pero los candidatos nuevos son ruido que el
  ranker no distingue y **desplazan** candidatos mejores. Revertido; queda `tune_retrievers.py`.
  **Re-testeado 2026-09-24** bajo dos protocolos: pareado por-usuario corregido
  (`REFIT_PARA_TEST=True`+`nfa=500`) da +1,56 σ, P=0,94 — firma casi idéntica a la
  original, los bugs de protocolo no eran la causa. Pero bajo `comparar_global_pareado.py`
  (corte temporal global, sin fuga por-usuario) da **-1,09 σ, P=0,13 — negativo**. Es el
  caso más claro de la sesión del mecanismo de fuga: un retriever *aprendido* que explota
  co-ocurrencia poblacional se beneficia de la señal de "futuro" que el split por usuario
  filtra sin querer, y el pareado por-usuario (aunque esté bien configurado) no lo puede
  ver. Tres señales alineadas (Kaggle histórico, pareado corregido borderline, corte
  global negativo) → rechazo confirmado con alta confianza. Revertido de nuevo en su
  totalidad. Ver `experiments/log.csv`.
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
- **BERT4Rec / masked-item como feature** (estrategia 6; `models/sequential.py`, `torch`
  CPU-only, eliminado tras el descarte). Encoder MLM sobre el historial cronológico
  (vocab podado a 18,7k libros con ≥3 interacciones, secuencias de hasta 50 ítems),
  `score_bert4rec_candidato` = softmax en la posición `[MASK]` final (truco estándar de
  inferencia). Costo real: ~223s/fit, ~290s inferencia batched para 10.673 usuarios.
  Pareado seed 42, `nfa=500`: **−1,44 σ, P(mejora)=0.077** (0.132900 vs 0.133997) — no pasa
  el gatekeeper. Mismo patrón que LMF/rating/difusión-fuente: 461k interacciones / mediana 9
  por usuario resultó, como se anticipó, poca data para un transformer. Revertida
  íntegramente, no llegó a CV ni Kaggle.
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

Se acabó lo barato **y lo caro**: LMF-feature, corroboración, relativas, rating,
difusión-como-fuente, BERT4Rec — todo ruido o negativo. Solo `score_difusion_candidato`
como **feature** movió algo (y poco).

**2026-09-24 — cerrada la última pendiente**: blend de familias de modelos distintas
(ALS + LGBMRanker vía Reciprocal Rank Fusion, `scripts/probar_blend_global.py`, sobre el
protocolo de corte global) — **rechazo contundente: -4,18 σ, P(mejora)=0,0000, CI
[-0.029,-0.010]**, el más claro de toda la sesión. Explicación: el `LGBMRanker` ya ve
`score_als`/`rank_als` como 2 de sus 40 features — ya aprendió a extraer y pesar lo que
ALS aporta junto con las otras 38 señales. Un RRF sin ponderar le da a ALS (2,6× más
débil solo) el mismo peso que al ranker completo, diluyendo una señal ya-aprendida y
superior con una versión más pobre de la misma información. No se probó una variante
ponderada (mucho más peso al ranker) — el argumento teórico de redundancia sugiere un
techo bajo ahí, no pareció buen uso del tiempo. **Con esto no queda ninguna estrategia de
mayor calibre de `estrategias.md` sin al menos un intento serio.**

El lever que rindió de verdad esta ronda fue **más señal de entrenamiento** (ventana
rodante). El set de features y de fuentes de candidatos está saturado — **6/6 intentos de
7ª fuente fallaron** (editorial, usuario-usuario ×2, LMF ×2, difusión — usuario-usuario y
LMF re-testeados 2026-09-24 bajo el protocolo corregido y bajo corte temporal global,
ambos con el mismo veredicto), incluso cuando la señal de base ya era buena, y el build
más grande de la ronda (BERT4Rec) tampoco pasó el pareado. La barrera no es "encontrar
más candidatos" ni "más señal de interés", es que el `LGBMRanker` no logra distinguir
cuáles de los candidatos extra son buenos — confirmado ahora bajo dos protocolos de
validación distintos, no solo el que tenía la fuga temporal.

**CV multi-seed + tuning de LightGBM bajo corte global — cerrados (2026-09-24, mismo día).**
`scripts/tune_ranker_global.py` (nuevo, mismo diseño que `tune_ranker.py`): optuna 10
trials × 2 seeds, confirmación final con 3 seeds (42, 7, 123). **El tuning no ayuda**,
tampoco en este régimen de N mucho más chico (4.471 grupos vs ~37-40k del split por
usuario a `n_cortes=5`) donde se hipotetizaba que el balance sesgo/varianza podría ser
distinto — mejor config de optuna: 0.045523 ± 0.001689, **peor en promedio y no positivo
en los 3 seeds** (pierde en 2 de 3) contra el conservador. Consistente con las 3 rondas
de tuning previas del proyecto bajo el split por usuario. **Los hiperparámetros
conservadores (`num_leaves=31, learning_rate=0.05, n_estimators=200`) quedan
confirmados bajo los dos protocolos de validación que existen ahora.**

El CV de 3 seeds del conservador da el número de referencia real de este protocolo:
**0.046116 ± 0.000129** — muy estable (CV ~0.3%). Reemplaza al 0.049061 original (una
sola corrida sin seed-CV, que resultó ser un outlier — el valor reproducible de
seed=42 es 0.045987, visto en 4 corridas independientes distintas). Con esto el gap
contra Kaggle (0.063-0.068) es de **~32%** (antes se estimaba 22-28% con el número
viejo), y requirió un refactor de `evaluate_global_cutoff.py` en dos fases
(`preparar_contexto_global` cacheado a disco + `evaluar_con_params_global` barato,
mismo patrón que `ranker.preparar_pipeline`/`evaluar_con_params`) para que tunear
hiperparámetros fuera viable en tiempo razonable.

**Qué es genuinamente nuevo y sin probar después de esta sesión** (no "cerrado", solo sin
tiempo/evidencia todavía): por qué más densidad de ventana empeoró en vez de mejorar
(sección anterior, tampoco tiene CV todavía); una variante de RRF ponderada hacia el
ranker. Ninguna de estas tiene el mismo respaldo teórico/histórico que las que sí se
probaron esta sesión — son extensiones, no vías obviamente prometedoras sin probar.

---

## Cómo correr

- `uv run pytest` — suite (158 tests).
- `uv run python -m src.recsys.submit --model ranker [--tag ...]` — CSV en
  `outputs/submissions/` (nombres con timestamp, nunca se pisan). Entrena con
  `n_cortes=5` + refit de etapa 1. ~40 min.
- `uv run python scripts/evaluate_ranker.py` — CV 3 seeds (`N_CORTES=5`, ~34 min/seed en
  frío) + NDCG por bucket + `feature_importances_`.
- `uv run python scripts/evaluate_global_cutoff.py` — NDCG@20 bajo corte temporal GLOBAL
  (ver "Validación con corte temporal global" arriba), ~7-8 min, un seed, sin CV todavía.
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
