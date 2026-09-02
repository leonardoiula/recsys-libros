# Decisiones — qué se conserva, qué se reconsidera

Índice de las decisiones de diseño tomadas en v0/v1/v2, para revisar qué
sigue teniendo sentido, qué se llevó a la versión siguiente y qué quedó
sin resolver. Es un complemento *de referencia rápida* a
`experiments/bitacora.md` (que tiene el razonamiento completo de cada
una) — acá no se repite el "por qué" en detalle, solo se linkea a dónde
está.

Columna **Estado**: es una sugerción de lectura, no un veredicto —
la idea es que la marques vos como ✅ conservar / ❌ sacar / 🔄 revisar.

## Para retomar: próximos pasos pendientes (al 2026-09-01)

Estado actual: `"ranker"` (v3, ALS+género+popularidad+autor/año/diversidad/
recencia+co-lectura+editorial+resumen+popularidad/frecuencia de
macro-género+tamaño de catálogo de editorial+señales cruzadas lector↔libro
**+4ª fuente de autor ya leído+5ª fuente de similitud de resumen+6ª
fuente de co-lectura ítem-ítem (kNN)** → LightGBM, 35 features,
hiperparámetros conservadores) es el modelo de referencia del proyecto,
con **0.05262 confirmado en Kaggle** (récord actual). Ideas concretas
para la próxima sesión, en orden sugerido (ver `experiments/modelo_actual.md`
para el detalle completo):

1. **Seguir atacando el generador de candidatos, no el reranking** —
   confirmado con evidencia real en varias rondas: la 4ª fuente (autor)
   dio la mejora más grande (+5.9% en Kaggle), la 5ª (resumen) una
   mejora más modesta pero real (+0.8% en Kaggle, +2.6% local), la 6ª
   (co-lectura ítem-ítem) la más chica de las tres pero igual
   consistente (+1.56% en Kaggle, +1.19% local, positivo en los 3
   seeds). Pero **no todo lo que sube el recall ayuda**: subir
   `n_por_fuente` a 500 también subió el recall (+30%) y el NDCG no
   acompañó (ver resuelto abajo) -- la ganancia depende de si los
   candidatos nuevos traen una señal *distinguible*, no solo más
   volumen de las mismas fuentes. La 6ª fuente mostró una versión más
   leve del mismo síntoma (recall +12.2%, eficiencia de ranking -10%)
   sin llegar a cancelar la mejora -- la señal se está agotando pero
   todavía no se apagó del todo.
2. **Usar `scripts/recall_candidatos.py` para decidir sobre etapa 1**
   (barato, ~5-8 min) **antes** de correr el CV completo (~20-25 min) —
   pero medir siempre los dos números (recall Y NDCG en un seed), no
   solo recall: subir `n_por_fuente` mostró que recall puede subir
   mucho sin que el NDCG lo acompañe.
3. **Usar `scripts/comparar_features_pareado.py` como criterio
   principal** para decidir si algo nuevo es señal real, no el desvío
   entre 3 seeds — confirmado en 2 rondas (autor y resumen) que tiene
   ~5x más poder estadístico. **Resuelto (sección 17):** se adaptó a
   `scripts/comparar_generadores_pareado.py` (compara dos generadores de
   candidatos completos, no solo subconjuntos de `FEATURES` sobre el
   mismo contexto) y se validó retroactivamente la 6ª fuente (kNN) con
   este método más riguroso.
4. **Investigar más a fondo la brecha local-vs-Kaggle** — sigue sin
   resolverse del todo (ver sección de split y validación local, más
   abajo). La 6ª fuente sumó un data point en la dirección contraria a
   la de resumen: acá el salto en Kaggle (+1.56%) fue *mayor* que la
   mejora local (+1.19%), mientras que en la ronda de resumen fue al
   revés (Kaggle subestimó la mejora local) -- confirma que el ruido de
   una sola submission pega en las dos direcciones, no solo subestima.
5. **Reproducir baselines de rondas anteriores dentro de la misma
   sesión (`git stash`) antes de comparar seed a seed** -- se adoptó
   esta ronda (ver sección 14) después de confirmar que el split es
   100% reproducible entre sesiones (el número de seed=42 coincidió
   exacto con el ya logueado), pero el entrenamiento de LightGBM no lo
   es del todo (ruido de punto flotante, ~0.0001-0.0006) -- vale la
   pena mantener esta práctica para que los diffs por seed sean
   confiables, no solo los promedios.

**Resuelto:** 6ª fuente de candidatos por co-lectura ítem-ítem (kNN)
(32→35 features) -- reusa el batch `co_scores_por_usuario` que ya
armaba `score_coleido`, tomando su top-`n_por_fuente` como candidatos
nuevos en vez de solo puntuar lo que ya trajo otra fuente. Recall
+12.2% (0.4559→0.5115) pero con la misma advertencia que
`n_por_fuente=500` en versión más leve: la eficiencia de ranking bajó
10% (vs. -22% de aquel caso). Baseline de 5 fuentes reproducido en la
misma sesión (`git stash`) para comparar seed a seed en igualdad de
condiciones -- CV 3 seeds positivo en los 3 (+1.19% promedio, la mejora
local más chica confirmada así hasta ahora, pero con magnitudes
consistentes entre seeds). **Confirmado en Kaggle: 0.05262, nuevo
récord** (+1.56% sobre 0.05181) -- el salto en Kaggle fue mayor que la
mejora local, al revés de lo que pasó con resumen. Ver sección 14 y
`bitacora.md`, sección "6ª fuente de candidatos: co-lectura ítem-ítem
(kNN)".

**Resuelto:** 5ª fuente de candidatos por similitud de resumen
(29→32 features) -- co-diseñada con el usuario tras preguntar cuán
importante era recomendar libros "raros" (se midió: los targets que
fallan las otras 4 fuentes son ~11x menos populares que los que sí se
capturan). CV 3 seeds positivo en los 3 (+2.6% promedio), más chico que
autor pero por encima del ruido. **Confirmado en Kaggle: 0.05181,
nuevo récord** (+0.8% sobre 0.05140) -- el salto absoluto es del orden
del ruido de una submission, pero la evidencia local es sólida. Ver
sección 13 y `bitacora.md`, sección "5ª fuente de candidatos: similitud
de resumen".

**Resuelto:** subir `n_por_fuente` de 150 a 500 -- recall del set de
candidatos +30% (0.445→0.578), pero la eficiencia de ranking bajó
(0.258→0.20) y el NDCG casi no se movió (+0.7%, un solo seed, dentro
del ruido), a casi el doble de costo de cómputo. Descartado sin correr
el CV completo (el propio método "recall primero" evitó gastar ese
tiempo). Ver `bitacora.md`, sección "n_por_fuente=150→500: recall sube
fuerte, NDCG no acompaña".

**Resuelto:** 4ª fuente de candidatos por autor ya leído (26→29
features) -- CV 3 seeds positivo en los 3, +7.1% en promedio, 5-10x más
grande que cualquier resultado previo de la sesión. Test pareado
confirmó que la mejora viene de los candidatos extra, no de las 3
features de tracking (0.44 sigma, no significativa por sí solas).
**Confirmado en Kaggle: 0.05140, nuevo récord del proyecto** (+5.9%
sobre 0.04855) -- segunda mejora más grande de todo el proyecto, y la
mejor validada estadísticamente. Ver sección 12 y `bitacora.md`, sección
"4ª fuente de candidatos: libros de autores ya leídos".

**Resuelto:** señales cruzadas lector↔libro (`popularidad_genero_lector_candidato`,
`frecuencia_genero_macro_por_genero_lector`, `edad_lector_al_publicarse`,
26 features en total) -- casi positivo en los 3 seeds (positivo en 2,
apenas negativo en el tercero, muy por debajo del ruido). Se confirmó
con el usuario vía submission real pese al caso límite: **0.04855 en
Kaggle, nuevo récord del proyecto** (+0.5% sobre 0.04831). Ver sección 11
y `bitacora.md`, sección "Señales cruzadas lector↔libro: nuevo récord
del proyecto".

**Resuelto:** franja de nacimiento del usuario como feature del ranker
(`popularidad_franja_candidato`, calco de país) -- mixto y negativo en
2 de 3 seeds, no cumple el criterio de "positivo en los 3 seeds". No se
gastó una submission; se revirtió del ranker (se mantiene un fix de
calidad de datos hecho de paso: `nacimiento == 1910` es un sentinel, no
una década real). Ver sección 10 y `bitacora.md`, sección "Franja de
nacimiento del lector: explorada y descartada".

**Resuelto:** país del usuario (`vive_en`) como feature del ranker
(`popularidad_pais_candidato`, popularidad segmentada por país) --
negativo en 2 de 3 seeds, no cumple el criterio de "positivo en los 3
seeds" que sí justificó género macro/tamaño de editorial. No se gastó
una submission; se revirtió del ranker. Ver sección 9 y `bitacora.md`,
sección "País del usuario (`vive_en`): explorada y descartada".

**Resuelto:** probar 23 features sin `en_editorial_leida`/
`n_libros_editorial_leidos` contra Kaggle -- el ablation dio negativo en
2 de 3 seeds (a diferencia de los casos límite anteriores), así que no
se gastó una submission. Se mantienen las 23 features. Ver sección 8 y
`bitacora.md`, sección "¿Sacar editorial? Ablation resuelve la pregunta
pendiente".

**Resuelto:** el criterio de "positivo en los 3 seeds individualmente"
(sin exigir que supere el desvío) para decidir si gastar una submission
-- el usuario lo confirmó explícitamente ("me parece que el criterio de
los 3 seeds va bien") después de que acertara en los tres casos de esta
sesión (género macro y tamaño de editorial: confirmar valió la pena;
sacar editorial: no confirmar también fue la decisión correcta). Queda
como criterio del proyecto para decidir submissions, no reemplaza la
regla del desvío para decisiones que no impliquen Kaggle (ej.
hiperparámetros, donde si se aplicó y evitó adoptar mejoras de ruido).

**Resuelto:** retomar el tuneo de LightGBM sobre la base de 23 features
-- mixto (positivo en 2 de 3 seeds, negativo en el tercero), mejora
promedio +0.12%, indistinguible de ruido. **Tercera vez** que el tuneo
de hiperparámetros da un resultado así de chico sobre bases de features
distintas -- no se adopta, se mantienen los hiperparámetros
conservadores. Ver `bitacora.md`, sección "Tuneo de LightGBM sobre 23
features: tercera vez que no se adopta".

Ver `experiments/bitacora.md`, sección "Se retoma el ranker", para el
detalle de por qué se priorizaron features sobre hiperparámetros, y
secciones "Co-lectura, editorial, resumen y género macro" y "Tamaño de
catálogo de editorial" para las dos rondas más recientes.

## 1. Split y validación local

| Decisión | Versión | Sigue vigente en | Estado sugerido | Detalle |
|---|---|---|---|---|
| Split leave-one-out **por usuario** (nunca aleatorio global) | v0 | v0, v1, v2 | ✅ conservar | Necesario para poder evaluar a todo usuario (ver `bitacora.md` v0, sección "Lógica del enfoque" punto 1). Nadie lo cuestiona. |
| `n_val=1` temporal (últimas interacciones de cada usuario por `fecha` a val, reemplazó `frac_val=0.2` aleatorio) | v0 → corregido después de v2 | v0, v1, v2 | ✅ **resuelto** | Confirmado con datos: split aleatorio inflaba a ALS de 0.26 a split temporal 0.12 (leakage temporal real), y `frac_val` proporcional inflaba aún más a usuarios pesados. Con el fix, la sobreestimación de v2 sobre Kaggle bajó de +573% a +162.6% (y a +65.6% si se repondera por la actividad real de `ejemplo.csv`). Ver sección "Corrección de metodología" en `bitacora.md`. Sigue quedando un residual de brecha — próximo paso: samplear/ponderar val por la composición real de usuarios de Kaggle. |
| `seed=42` fijo para reproducibilidad | v0 | v0, v1, v2 | ✅ conservar | Sin comentarios en contra; permite comparar versiones sobre el mismo split exacto. |
| NDCG@k con descuento `1/log2(pos+2)`, normalizado por IDCG | v0 | v0, v1, v2 | ✅ conservar | Es la métrica que pide la competencia (NDCG@k) — no es una decisión de diseño discutible, es el objetivo. Lo que sí está en duda es el *split* sobre el que se calcula (ver arriba), no la fórmula de la métrica en sí. |
| `evaluar_ndcg` (un ranking global compartido) vs `evaluar_ndcg_personalizado` (un ranking por usuario) como dos funciones separadas | v0 → v1 | v1, v2 | ✅ conservar | v1 necesitó una variante porque cada usuario tiene su propio ranking; v2 reusa `evaluar_ndcg_personalizado` tal cual. No hay fricción, ambas conviven bien. |

## 2. Filtrado de libros ya leídos

| Decisión | Versión | Sigue vigente en | Estado sugerido | Detalle |
|---|---|---|---|---|
| Nunca recomendar un libro que el usuario ya leyó (`libros_leidos_por_usuario`) | v0 | v0, v1, v2 | ✅ conservar | Es la mejora individual más grande medida hasta ahora en Kaggle: **+7.3%** solo por este filtro (fila `popularity_sin_filtro` vs `popularity` en `log.csv`). Fuera de discusión. |
| En v2, re-filtrar explícitamente contra `libros_leidos` en vez de confiar solo en `filter_already_liked_items=True` de `implicit` | v2 | v2 | ✅ conservar | No es redundancia inocua: se confirmó (probando la librería real) que `implicit.recommend` puede rellenar huecos repitiendo ids ya vistos con un score inválido en vez de descartarlos — el filtro explícito es la única protección real contra eso. Ver `bitacora.md` v2, "Lógica del enfoque" punto de `recomendar_por_usuario`. |

## 3. Popularidad como señal base / fallback

| Decisión | Versión | Sigue vigente en | Estado sugerido | Detalle |
|---|---|---|---|---|
| Score bayesiano de popularidad (`C`, `m` calculados sobre los datos recibidos, nunca hardcodeados) en vez de rating promedio simple | v0 | v0, v1 (por género/franja), v2 (fallback) | ✅ conservar | Se descartó explícitamente el promedio simple por ruido (libros con 1-2 interacciones y rating alto por azar). Sigue siendo el fallback de último recurso en v1 y v2. |
| `C = n.mean()` (en vez de mediana o media geométrica) para el shrinkage bayesiano | v0 | v0, v1 (fallback), v2 (fallback) | ✅ **resuelto — se confirma la media** | Se sospechaba que la media (9.52) era demasiado agresiva frente a una distribución right-skewed (mediana 2) y se sweepeó contra NDCG real: la media ganó por lejos (0.0066 vs 0.0007 geométrica vs 0.0002 mediana). Con leave-one-out estricto, un `C` chico deja subir demasiado a libros de nicho con pocos ratings altos por azar. `C` quedó configurable en `fit_popularity` igual, para no volver a asumir a ciegas. Ver `bitacora.md`. |
| Normalización de `libros.genero` (capitalización/espacios) antes de agrupar | v1 (a partir del EDA) | v1 | ✅ conservar (uso queda acotado a v1) | Se probó explícitamente usar género como fallback de usuarios livianos en v2 (`recomendar_hibrido`) y se midió que ALS le gana en todos los buckets de actividad — no hace falta llevar género a v2. Sigue siendo la base correcta de v1, que se mantiene como modelo aparte. |
| Franja de nacimiento (década, no edad real) como proxy demográfico | v1 (a partir del EDA) | v1 | 🔄 revisar — **código casi muerto** | Se midió en la práctica: aporta candidatos a **0.03% de los usuarios** (2 de 6,770) porque género casi siempre alcanza para llenar el top-20 por sí solo. No se sacó del código (bajo riesgo, cero costo dejarlo), pero no es un criterio que esté aportando nada medible hoy. |
| Cadena de fallback secuencial género → franja → global, con backfill sin duplicar | v1 | v1 | ✅ **resuelto — se confirma que no conviene llevarla a v2** | Se implementó y midió `recomendar_hibrido` (ALS para usuarios activos, género→global para livianos) bajo el split corregido: ALS le gana a género en **todos** los buckets de actividad probados, incluso con 1 sola interacción de historial. La función queda en el código (testeada) pero no se usa en `submit.py` — cualquier umbral de ruteo a género empeora el resultado con este dataset. Ver `bitacora.md`. |

## 4. Arquitectura de código (`submit.py` / modelos)

| Decisión | Versión | Sigue vigente en | Estado sugerido | Detalle |
|---|---|---|---|---|
| `MODELOS: dict[str, Callable[[usuarios, k], dict]]` — cada modelo es una función `(usuarios, k) -> {id_lector: [id_libro,...]}`, ya entrenada con todos los datos | v0 (ranking único) → extendido en v1 (ranking por usuario) | v0, v1, v2 | ✅ conservar | Extensión limpia: v0 devuelve el mismo ranking recortado por usuario, v1/v2 devuelven ranking ya personalizado. `--model` en la CLI queda automático (`choices=sorted(MODELOS)`) — no hubo que tocar el CLI para agregar `als`. |
| Cada `fit_*`/modelo recibe los datos como parámetro (nunca un DB global implícito), para poder reusarse igual con train (validación local) o con todos los datos (submission) | v0 | v0, v1, v2 | ✅ conservar | Es lo que permite que el mismo código sirva para evaluar y para generar la entrega, sin dos implementaciones paralelas. |

## 5. Modelo ALS (v2)

| Decisión | Versión | Estado sugerido | Detalle |
|---|---|---|---|
| Usar el rating explícito (1-10) como confianza de `implicit.als`, vía `confianza = 1 + alpha*rating` (Hu/Koren/Volinsky) | v2 | ✅ **resuelto — mejora confirmada** | Se probó tunear `alpha` (antes fijo en 1:1 con el rating crudo) junto con `factors`/`regularization` vía optuna: +11.5% de NDCG@20 local con `alpha≈4.7`, a costa de un Recall@200 levemente menor (0.398→0.382). Ver `bitacora.md`, sección "Mejoras a ALS". |
| Hiperparámetros `factors=256, regularization=0.128, alpha=4.718, iterations=20` | v2 | ❌ **descartado, se revirtió el default** | Encontrados con optuna (30 trials) sobre un único split local: mejoraban el NDCG local +11.5%, pero dieron **0.03341 en Kaggle, peor que 0.03864** del config anterior. Se revirtió `fit_als` a `factors=128, regularization=0.1, alpha=None` (rating crudo) como default -- ver fila siguiente y `bitacora.md`, sección "Revert de ALS". |
| Validación cruzada (3 seeds) protege contra sobreajustar a un split, pero no corrige sesgo sistemático del proxy local | v2/v3 | ✅ **resuelto — no es sesgo, es ruido de muestra chica en Kaggle** | Se probaron y descartaron dos hipótesis (aleatoriedad de ALS, composición de población) y se encontró la real: con n=832 usuarios en `ejemplo.csv` y una métrica donde el 81% de los usuarios da NDCG=0, el error estándar de una sola submission es ~0.0065 (IC 95% ±0.0127). Las diferencias que veníamos usando para decidir (0.0005 a 0.0052) están todas por debajo de 1 error estándar -- son indistinguibles de ruido, no evidencia real de que una config sea mejor que otra. Ver `bitacora.md`, sección "Investigando el sesgo sistemático". |
| BPR (`implicit.bpr.BayesianPersonalizedRanking`) como alternativa a ALS | v2 | ❌ **descartado** | Tuneado con optuna (30 trials) igual que ALS: perdió en NDCG@20 (0.085 vs 0.113) y en Recall@200 (0.318 vs 0.382). La hipótesis de que un ranking pairwise se ajustaría mejor a NDCG que ALS no se confirmó con este dataset. |
| Fallback a popularidad global (v0) para usuarios sin fila en la matriz (cold start real) | v2 | ✅ conservar | No se disparó en el split local, pero cubre el caso general sin costo. |
| No rutear usuarios livianos de ALS hacia género/franja (v1) | v2 | ✅ **resuelto — se prueba y se descarta** | Se implementó y midió (`als.py::recomendar_hibrido`, no usado en `submit.py`): ALS le gana a género en todos los buckets de actividad, incluso con 1 interacción. La hipótesis de que ALS es poco confiable para usuarios livianos no se sostuvo con datos reales sin leakage. Distinto de la idea de abajo (combinar como *features* de un ranker, no como ruteo excluyente). |
| Combinar ALS + género/popularidad como features de un ranker (LightGBM) en vez de un solo modelo base | v2 → v3 | ✅ **implementado, gana en validación cruzada — pendiente confirmar en Kaggle** | Ver sección 6, modelo v3. |

## 6. Ranker de dos etapas (v3)

| Decisión | Versión | Estado sugerido | Detalle |
|---|---|---|---|
| Split de tres niveles (`train_candidatos`/`train_ranker`/`test_final`) para entrenar el ranker sin leakage | v3 | ✅ conservar | Necesario porque el ranker usa el score de ALS/popularidad/género como *features* -- si esos scores salieran de datos que el ranker ve como etiqueta, memorizaría en vez de aprender a combinar señales. |
| Evaluar con validación cruzada (3 seeds), no un solo split | v3 | ✅ conservar — **principio a mantener para todo modelo futuro** | Restricción explícita del usuario tras el episodio de ALS+optuna. El ranker ganó a ALS solo en los 3 seeds (+3.9% promedio, menor desvío: 0.00178 vs 0.00249) -- señal mucho más confiable que una mejora medida en un solo split. |
| Hiperparámetros conservadores de LightGBM (`num_leaves=31, learning_rate=0.05, n_estimators=200`), sin sweep tipo optuna | v3 | ✅ conservar por ahora | Decisión deliberada para no repetir el sobreajuste al proxy local que tuvo ALS. Si se tunea en el futuro, usar la misma validación cruzada multi-seed. |
| Wireado como modelo nuevo `"ranker"` en `submit.py` (no reemplaza `"als"`) | v3 | ✅ **confirmado en Kaggle, mejora real** | 0.03578, +7.1% sobre su propia base (ALS tuneado, 0.03341) -- a diferencia de ALS+optuna, esta mejora cross-validada SÍ se sostuvo en Kaggle. Sigue -7.4% por debajo del mejor histórico (ALS original, 0.03864) porque hereda esa base deteriorada como señal -- ver `bitacora.md`. |
| Ranker reconstruido sobre el ALS original (revertido) | v3 | 🔄 **inconcluso, no refutado** | Kaggle dio 0.03815 vs 0.03864 de ALS solo -- una diferencia de -0.00049, muy por debajo de 1 error estándar (~0.0065, ver fila de arriba). **No hay evidencia real de que el ranker empeore sobre ALS original** -- el dato es simplemente ruido, ni a favor ni en contra. Corrige la lectura anterior ("hipótesis refutada"), que fue prematura. |
| Revert de ALS a `factors=128/reg=0.1/alpha=None` como default | v2 | 🔄 **probablemente correcto, pero no confirmado con certeza estadística** | La diferencia observada en Kaggle (0.00523) tampoco supera 1 error estándar. Se mantiene el revert porque hay una explicación causal independiente y razonable (sobreajuste de optuna a un único split), no porque la diferencia de Kaggle sea concluyente por sí sola. |
| Agregar features de autor/año de edición/diversidad de género/recencia al ranker | v3 | ✅ **confirmado en Kaggle -- nuevo récord del proyecto** | +9.4% de NDCG local (positivo en los 3 seeds) se tradujo en +15.3% real en Kaggle (0.03864 -> 0.04457). Primera mejora de esta ronda que se confirma limpiamente de punta a punta, sin sorpresas. `"ranker"` pasa a ser el modelo de referencia del proyecto. Ver `bitacora.md`. |
| Tunear hiperparámetros de `LGBMRanker` con optuna (validación cruzada, 2 seeds/trial) | v3 | ❌ **no se adopta -- mejora menor que el ruido** | +1.2% sobre los hiperparámetros conservadores, pero la mejora absoluta (0.0013) es menor que el desvío entre los 3 seeds (0.0015) -- mismo criterio aprendido con Kaggle aplicado localmente: no confiar en diferencias menores que la variabilidad natural. Quedan los hiperparámetros conservadores en producción. |

## 7. Co-lectura, editorial, resumen y género macro (ronda 2026-08-30)

| Decisión | Estado sugerido | Detalle |
|---|---|---|
| `score_coleido` (co-lectura ítem-ítem, matriz dispersa `cooc = X.T @ X` sobre la matriz binaria de ALS) | ✅ conservar | Deja de ser "cara de calcular bien": ~3s para 48k libros/461k interacciones. `feature_importances_` la ubica consistentemente en el grupo de mayor peso (por delante de `score_genero`/`score_popularidad`). |
| `en_editorial_leida`/`n_libros_editorial_leidos` | 🔄 **confirmado en conjunto, pero es la señal más débil** | Solo en Kaggle *junto con* las otras 5 features nuevas (no se probó la combinación sin ella). `feature_importances_` la ubica sistemáticamente al final -- ver "Próximos pasos" arriba. |
| `sim_resumen_historial` (TF-IDF + coseno contra el perfil de lectura del usuario) | ✅ conservar | Igual que co-lectura, entra en el grupo de mayor peso en `feature_importances_` en los 3 seeds. |
| `_normalizar_genero` ignora acentos además de capitalización | ✅ **resuelto** | 54 categorías → 52 reales: 2 pares eran typos de tilde de la misma categoría (`"clásicos"`/`"clasicos"`, `"biografías"`/`"biografiás"`), no géneros distintos. Sin mapa de alias a mano. |
| Macro-taxonomía de género (10 familias de dominio, co-diseñada con el usuario) | ✅ conservar | Ver tabla abajo. Implementada como mapa explícito de 25 categorías (las 9 familias con identidad propia); todo lo demás cae a un catch-all por default. |
| `popularidad_genero_macro_candidato` / `frecuencia_genero_macro_usuario` | ✅ **confirmado en Kaggle -- nuevo récord del proyecto** | Ver fila de abajo. |
| Ranker con las 6 features de esta ronda (co-lectura + editorial + resumen + macro-género, 22 en total) | ✅ **confirmado en Kaggle -- nuevo récord del proyecto** | CV 3 seeds: 0.108740±0.002955, positivo en los 3 seeds individualmente pero *sin* superar el desvío entre seeds (regla estricta habría dicho "no confirmado"). Se decidió confirmar con una submission real en vez de seguir analizando el número local. **0.04658 en Kaggle, +4.5% sobre el récord anterior** (0.04457). Ver `bitacora.md`, sección "Co-lectura, editorial, resumen y género macro". |

### Macro-géneros (taxonomía de dominio)

| Macro-género | Libros | Categorías granulares que agrupa |
|---|---|---|
| Narrativa y clásicos | 14.746 | narrativa, ficción literaria, literatura contemporánea, clásicos de la literatura, novela |
| Ensayo, biografía y no ficción | 7.752 | ensayo, biografías/memorias, no ficción, historia, historia militar, historia del cine, filosofía contemporánea, feminismo y mujer, naturaleza y ciencia |
| Novela negra y suspenso | 6.878 | novela negra/intriga/terror, novela negra |
| Infantil y juvenil | 6.053 | infantil y juvenil, juvenil, lecturas complementarias |
| Histórica y aventuras | 3.905 | histórica y aventuras |
| Fantástico y ciencia ficción | 3.000 | fantástica, ciencia ficción |
| Romántica y erótica | 2.957 | romántica, erótica |
| Cómic y novela gráfica | 2.197 | cómics, novela gráfica |
| Poesía y teatro | 1.234 | poesía/teatro, poesía |
| Práctico y misceláneo (catch-all) | 1.799 | ~27 categorías restantes (humor, autoayuda, cocina, economía, música, deportes, medicina, derecho, idiomas, ...), todas con menos de 460 libros |

## 8. Tamaño de catálogo de editorial (ronda 2026-08-30, parte 2)

| Decisión | Estado sugerido | Detalle |
|---|---|---|
| No aplicar una macro-taxonomía categórica a `editorial` (a diferencia de género) | ✅ **resuelto -- se descarta explícitamente** | 2.762 editoriales entre libros con interacción (vs. 52 de género), cola mucho más larga (91% con <20 libros, 51% con exactamente 1) y sin agrupación temática natural -- "Anagrama" y "Alfaguara" no comparten un "dominio" como sí lo hacen las categorías de género. Forzar 6-10 "familias de editoriales" habría sido artificial. |
| `n_libros_editorial_catalogo` (tamaño del catálogo de la editorial, no del historial del usuario) como feature numérica simple | ✅ **confirmado en Kaggle -- nuevo récord del proyecto** | CV 3 seeds: 0.109735±0.003719, mismo patrón límite que la fila de género macro (positivo en los 3 seeds, no supera el desvío). `feature_importances_` la ubica por delante de `en_editorial_leida`/`n_libros_editorial_leidos` en los 3 seeds. Confirmado con el usuario vía submission real: **0.04831 en Kaggle, +3.7% sobre el récord anterior** (0.04658). Ver `bitacora.md`, sección "Tamaño de catálogo de editorial". |
| Sacar `en_editorial_leida`/`n_libros_editorial_leidos` (las más débiles individualmente) ahora que está `n_libros_editorial_catalogo` | ✅ **resuelto -- no conviene, no se sube a Kaggle** | Ablation (23 → 21 features): CV 3 seeds 0.109287±0.003276, negativo en 2 de 3 seeds -- a diferencia de los casos límite anteriores (siempre positivos), acá la señal local apunta claramente en contra. Aunque son las features más débiles individualmente, no son ruido puro: aportan algo real en conjunto. Se mantienen las 23 features. Ver `bitacora.md`, sección "¿Sacar editorial? Ablation resuelve la pregunta pendiente". |
| "Positivo en los 3 seeds individualmente" como criterio para gastar una submission (sin exigir que supere el desvío) | ✅ **confirmado por el usuario** | Acertó en los tres casos de esta sesión (género macro y tamaño de editorial: confirmar valió la pena; sacar editorial, mixto/negativo: no confirmar también fue correcto). Sigue reservándose la regla del desvío para decisiones que no impliquen Kaggle. |
| Retomar el tuneo de `LGBMRanker` con optuna sobre la base de 23 features (10 trials × 2 seeds, `scripts/tune_ranker.py`, ahora con contexto cacheado por seed -- ver `bitacora.md`) | ❌ **no se adopta -- tercera vez, mejora menor que el ruido** | Mejor config (`num_leaves=7, learning_rate=0.134, n_estimators=280, min_child_samples=82, reg_alpha=0.0012, reg_lambda=7.58`): CV 3 seeds 0.109864±0.003066 vs 0.109735±0.003719 del conservador -- MIXTO (positivo en 2 seeds, negativo en 1), +0.12% de mejora promedio. Tercera vez que el tuneo de hiperparámetros da un resultado así de chico sobre bases de features distintas. Quedan los hiperparámetros conservadores (`num_leaves=31, learning_rate=0.05, n_estimators=200`) en producción. |

## 9. País del usuario (`vive_en`, ronda 2026-08-31)

| Decisión | Estado sugerido | Detalle |
|---|---|---|
| País (98 categorías, `vive_en` parseado a lo que sigue al último " - ") en vez de macro-regiones o saltar la feature | ✅ **co-diseñado con el usuario** | 68% de los lectores vive en España, cola larga de 97 países más. Se descartó agrupar en macro-regiones (España/Latinoamérica/otro) -- se probó país tal cual. |
| "Desconocido" como categoría propia (no fallback a popularidad global) para los ~9% de `vive_en` vacíos o `"¿?"` | ✅ **co-diseñado con el usuario** | En su momento, a diferencia de cómo se manejaba la franja de nacimiento (que descartaba los `NaN`) -- ver sección 10, donde se alineó el mismo criterio para franja. |
| `popularidad_pais_candidato` (popularidad bayesiana del candidato dentro del país del usuario, `popularity_segmentada.fit_popularity_por_pais`) | ❌ **descartada -- no se sube a Kaggle** | CV 3 seeds: 0.109002±0.003161 vs 0.109735±0.003719 de las 23 features confirmadas -- negativo en 2 de 3 seeds (seed=7, seed=123), prácticamente empatado en el tercero (seed=42). No cumple el criterio de "positivo en los 3 seeds" que sí justificó confirmar género macro y tamaño de editorial. `feature_importances_` la ubica en la mitad de la tabla -- no es la señal más floja, pero el conjunto empeora con ella adentro. Revertida de `FEATURES`/`ranker.py` para no regresionar el récord actual (0.04831). Se mantienen `pais_por_usuario`/`fit_popularity_por_pais` en `popularity_segmentada.py` (testeadas, sin usar) por si vale la pena retomar la idea con otro enfoque. Ver `bitacora.md`, sección "País del usuario (`vive_en`): explorada y descartada". |

## 10. Franja de nacimiento del usuario (`nacimiento`, ronda 2026-08-31)

| Decisión | Estado sugerido | Detalle |
|---|---|---|
| `nacimiento == 1910` tratado como "desconocido" (no una década real) | ✅ **co-diseñado con el usuario** | 415 de 438 lectores en la franja "1910s" tienen el valor exacto 1910 (vs. 1-9 casos para 1911-1917) -- patrón consistente con un default de formulario, no con gente real de esa edad en un dataset de lectura activa. |
| Modificar `franja_nacimiento_por_usuario`/`fit_popularity_por_franja_nacimiento` (compartidas con v1) en vez de una copia aislada para el ranker | ✅ **co-diseñado con el usuario** | v1 (`popularity_segmentada.recomendar_por_usuario`, fallback género→franja→global) ahora también trata "desconocido" como categoría propia en vez de saltar directo a popularidad global para ese ~30% de usuarios. No se re-evaluó el NDCG de v1 con este cambio (no es el modelo activo del proyecto), pero es una mejora de calidad de dato, no una regresión esperada. |
| `popularidad_franja_candidato` (calco de `popularidad_pais_candidato`, segmentado por franja de nacimiento del usuario) | ❌ **descartada -- no se sube a Kaggle** | CV 3 seeds: 0.109300±0.002688 vs 0.109735±0.003719 de las 23 features confirmadas -- mixto (positivo en seed=42, negativo en seed=7 y seed=123), media -0.40%. No cumple "positivo en los 3 seeds". Mismo desenlace que país pese a que género del lector/franja está mejor distribuido (menos sesgo a una categoría dominante). Revertida de `FEATURES`/`ranker.py`; se mantiene el fix del sentinel 1910 (independiente de esta feature puntual). Ver `bitacora.md`, sección "Franja de nacimiento del lector: explorada y descartada". |

## 11. Señales cruzadas lector↔libro (ronda 2026-08-31, parte 2)

| Decisión | Estado sugerido | Detalle |
|---|---|---|
| `lectores.genero` es el género DECLARADO del lector (Mujer/Hombre/"-"), no el género literario -- documentado explícitamente en el código para no confundir | ✅ **hallazgo, no decisión** | Mismo nombre de columna (`genero`) que `libros.genero` pero significado completamente distinto -- trampa real detectada antes de implementar nada. |
| Probar sistemáticamente los 3 candidatos de señales cruzadas juntos (en vez de uno a la vez) | ✅ **co-diseñado con el usuario** | Pedido explícito del usuario ("probemos sistematicamente las 3") para acelerar la exploración, dado que el modelo venía sin superar expectativas tras dos rondas fallidas (país, franja). |
| `popularidad_genero_lector_candidato` (popularidad bayesiana segmentada por género declarado, mismo patrón que país/franja) | ✅ **confirmado en Kaggle -- nuevo récord del proyecto** | La más fuerte de las 3 en `feature_importances_` (consistentemente top-10 del set completo en los 3 seeds). A diferencia de país (68% un solo valor) y franja (~30% desconocido), género del lector está bien balanceado (35/33/32%) -- posible explicación de por qué esta ronda sí funcionó donde las dos anteriores fallaron. |
| `frecuencia_genero_macro_por_genero_lector` (afinidad de *cohorte* por macro-género, no historial individual) | ✅ **confirmado en Kaggle (en conjunto) -- señal débil pero no descartable sola** | Una de las dos más débiles del trío en `feature_importances_`. No se probó aislada. |
| `edad_lector_al_publicarse` (`anio_edicion - nacimiento`, cruce directo lector↔libro) | ✅ **confirmado en Kaggle (en conjunto) -- ablation confirma que aporta pese a ser la más floja** | Sacarla en un ablation empeoró el resultado (ver fila siguiente) -- mismo patrón que `en_editorial_leida`/`n_libros_editorial_leidos`: la feature más débil individualmente no es ruido puro. |
| Ranker con las 3 features de esta ronda (26 en total) | ✅ **confirmado en Kaggle -- nuevo récord del proyecto** | CV 3 seeds: 0.110112±0.003411 -- casi positivo en los 3 seeds (positivo en 2, apenas negativo en el tercero, muy por debajo del ruido). Confirmado con el usuario vía submission real pese al caso límite (mismo criterio que género macro/tamaño de editorial). **0.04855 en Kaggle, +0.5% sobre el récord anterior** (0.04831). Tercera vez que un caso límite se confirma real. Ver `bitacora.md`, sección "Señales cruzadas lector↔libro: nuevo récord del proyecto". |
| Ablation: sacar `edad_lector_al_publicarse` (25 features) para intentar que el seed límite cruce a positivo | ❌ **descartado -- empeoró en vez de mejorar** | CV 3 seeds: 0.109897±0.002980, peor que las 26 completas en media y en el seed límite (-0.00084 vs -0.00024). Se revirtió el ablation, quedaron las 26 features completas. |

## 12. 4ª fuente de candidatos: libros de autores ya leídos (ronda 2026-08-31, parte 3)

| Decisión | Estado sugerido | Detalle |
|---|---|---|
| Agregar una 4ª fuente de candidatos (no solo feature) para libros de autores ya leídos, en vez de seguir sumando features sobre las 3 fuentes existentes | ✅ **confirmado en Kaggle -- nuevo récord del proyecto** | Motivada por el análisis de generador de candidatos (`modelo_actual.md`): 28.6% de los targets son de un autor ya leído. Rankeada por popularidad global (no un score bayesiano por autor). CV 3 seeds: 0.117495±0.002562 vs 0.109735±0.003719 de las 26 features -- positivo en los 3 seeds, 5-10x más grande que cualquier resultado previo de la sesión. Recall del set de candidatos: 0.394→0.445. **0.05140 en Kaggle, +5.9% sobre el récord anterior** (0.04855) -- segunda mejora más grande del proyecto (detrás de +15.3% de autor/año/género/recencia) y la mejor validada estadísticamente. `"ranker"` (29 features, 4 fuentes) pasa a ser el modelo de referencia. Ver `bitacora.md`, sección "4ª fuente de candidatos: libros de autores ya leídos". |
| Topear el total de candidatos de esta fuente a `n_por_fuente` (150), priorizando los autores más leídos | ✅ **resuelto -- problema real de memoria encontrado en la verificación** | Sin tope, usuarios con cientos de autores leídos llegaban a 5.304 candidatos de esta fuente sola -- una corrida completa se cortó (`killed`, sin traceback). Con el tope: recall baja levemente (0.473→0.445) pero sigue muy por encima del original, y el pipeline corre de forma estable. |
| `score_autor_candidato`/`rank_autor_candidato`/`en_autor_candidato` (3 features nuevas, 26→29) | 🔄 **mantenidas, pero test pareado dice que no aportan por sí solas** | Comparando el mismo pool de candidatos con vs. sin estas 3 features (test pareado, seed=42): diferencia +0.0004, 0.44 sigma, no significativa -- el ranker ya aprovechaba casi toda la mejora con features existentes (`en_autor_leido`, `n_libros_autor_leidos`, etc.). La mejora real viene de los candidatos, no de las features. Se mantienen igual (no restan, documentan la fuente explícitamente). |
| Subir `n_por_fuente` de 150 a 500 (next step #2 del análisis de paradigma) | ❌ **descartado -- recall sube, NDCG no acompaña** | Recall del set de candidatos: 0.445→0.578 (+30%), pero la eficiencia de ranking (NDCG/recall) bajó de 0.258 a 0.20 -- con ~3x más candidatos por usuario la tarea de rankear se vuelve más difícil, casi cancelando la ganancia. NDCG@20 (un solo seed, alcanza para descartar): 0.114810→0.115601 (+0.7%, ruido). Costo de cómputo casi 2x. No se corrió el CV completo de 3 seeds -- el método "recall primero, CV completo solo si vale la pena" (diseñado esta sesión) evitó gastar ese tiempo. `n_por_fuente` se mantiene en 150. Ver `bitacora.md`, sección "n_por_fuente=150→500: recall sube fuerte, NDCG no acompaña". |

## 13. 5ª fuente de candidatos: similitud de resumen (ronda 2026-08-31, parte 4)

| Decisión | Estado sugerido | Detalle |
|---|---|---|
| Agregar una 5ª fuente de candidatos por similitud de resumen (contenido, no popularidad/colaborativo) | ✅ **confirmado en Kaggle -- nuevo récord del proyecto** | Motivada por medir (co-diseñado con el usuario, quien preguntó cuán importante era recomendar libros "raros"): los targets que las 4 fuentes anteriores fallan en capturar son ~11x menos populares (mediana 21 vs 231 interacciones) que los que sí capturan. Es la única de las 5 fuentes que no depende de cuánta gente más leyó un libro. Recall: 0.445→0.456 (+2.4%). CV 3 seeds: 0.120547±0.002674 vs 0.117495±0.002562 de las 29 features -- positivo en los 3 seeds (+0.00269/+0.00476/+0.00171, media +2.6%), más chico que autor pero por encima del ruido calibrado esta sesión. **0.05181 en Kaggle, +0.8% sobre 0.05140** -- el salto absoluto (0.00041) es del orden del error estándar de una submission (~0.0065), pero la evidencia local es sólida (a diferencia de señales cruzadas, donde ya era límite localmente) -- se interpreta como mejora real subestimada por el ruido de esta muestra puntual, no como que la mejora local fuera ruido. `"ranker"` (32 features, 5 fuentes) pasa a ser el modelo de referencia. Ver `bitacora.md`, sección "5ª fuente de candidatos: similitud de resumen". |
| Procesar en lotes (`TAMANO_LOTE_RESUMEN=500`) para no materializar un producto denso usuarios×libros de una sola vez | ✅ **precaución aplicada preventivamente** | Tras dos problemas de memoria reales esta sesión (autor sin tope, `n_por_fuente=500`). En la práctica no hubo problema de memoria en esta ronda -- la única anomalía de rendimiento (seed=7 tardó ~2.9hs) fue por la PC entrando en reposo, no por memoria/código (ver `bitacora.md`). |
| `score_resumen_candidato`/`rank_resumen_candidato`/`en_resumen_candidato` (3 features nuevas, 29→32) | 🔄 **mantenidas, pero test pareado dice que no aportan por sí solas** | Mismo patrón que autor: test pareado (seed=42) da 0.67 sigma, no significativo -- se solapan con `sim_resumen_historial`, ya existente. La mejora real viene de los candidatos. |

## 14. 6ª fuente de candidatos: co-lectura ítem-ítem (kNN) (ronda 2026-09-01)

| Decisión | Estado sugerido | Detalle |
|---|---|---|
| Agregar una 6ª fuente de candidatos por co-lectura ítem-ítem (kNN), reusando el batch `co_scores_por_usuario` (`X_batch @ cooc`) que ya arma `score_coleido` | ✅ **confirmado en Kaggle -- nuevo récord del proyecto** | Sigue siendo colaborativo (mismo sesgo hacia libros con suficientes interacciones que ALS/popularidad, aunque más suave que popularidad pura), pero trae candidatos que ALS no trae. No hizo falta ningún matmul nuevo -- se tomó el top-`n_por_fuente` del mismo cálculo que ya existía, con `heapq.nlargest` para no ordenar el dict completo de un usuario con mucho historial. Recall: 0.4559→0.5115 (+12.2%), pero eficiencia de ranking (NDCG/recall) bajó de 0.258 a 0.232 (-10%) -- versión más leve del síntoma de `n_por_fuente=500` (que bajó -22% y sí se descartó), sin llegar a cancelar la mejora. Baseline de 5 fuentes reproducido en la MISMA sesión (`git stash`) para comparar seed a seed en igualdad de condiciones -- confirmó que el split es 100% determinístico entre sesiones (seed=42 dio 0.117503, idéntico al ya logueado). CV 3 seeds: 0.121983±0.002949 vs 0.120547±0.002674 de las 32 features -- positivo en los 3 seeds (+0.001122/+0.001628/+0.001560, media +1.19%), magnitudes consistentes entre sí, pero la mejora local más chica confirmada así hasta ahora. **0.05262 en Kaggle, +1.56% sobre 0.05181** (+0.00081 absoluto) -- salto mayor que la mejora local, al revés de lo que pasó con resumen (donde Kaggle subestimó la mejora local). `"ranker"` (35 features, 6 fuentes) pasa a ser el modelo de referencia. Ver `bitacora.md`, sección "6ª fuente de candidatos: co-lectura ítem-ítem (kNN)". |
| `score_coleido_candidato`/`rank_coleido_candidato`/`en_coleido_candidato` (3 features nuevas, 32→35) | 🔄 **mantenidas, sin test pareado dedicado esta ronda** | Mismo patrón de nombres que autor/resumen -- distintas de `score_coleido` (sigue puntuando cualquier candidato). A diferencia de autor/resumen, no se corrió `comparar_features_pareado.py` para aislar si estas 3 features aportan por sí solas o si la mejora viene solo de los candidatos nuevos (el script compara subconjuntos de `FEATURES` sobre un único contexto, no dos generadores de candidatos distintos -- queda pendiente adaptarlo). |

## 15. Embeddings semánticos (sentence-transformers) para el perfil de contenido (ronda 2026-09-01)

| Decisión | Estado sugerido | Detalle |
|---|---|---|
| Reemplazar TF-IDF por embeddings de `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`, local, sin costo/API) en el perfil de contenido que usan `sim_resumen_historial` y la 5ª fuente de candidatos | ❌ **descartado -- no mejora, evidencia débil de que empeora** | Curiosidad del usuario tras discutir TF-IDF vs. embeddings en abstracto: la hipótesis era que un embedding semántico capturaría libros con vocabulario distinto pero significado similar, algo que TF-IDF (matching léxico exacto) no puede por diseño -- justo el tipo de señal que la 5ª fuente busca (targets ~11x menos populares que las otras fuentes capturan). Implementado como función hermana de `_calcular_perfil_texto` (misma interfaz de salida, sin tocar el resto del pipeline) detrás de un parámetro `metodo_texto` opt-in, default sin cambios. Un solo seed (42) alcanzó para descartar: recall del set de candidatos prácticamente igual (0.5115 TF-IDF vs. 0.5099 embeddings, -0.3%) -- la señal más importante, porque significa que los embeddings casi no traen candidatos *distintos* a los de TF-IDF, contra la hipótesis de partida. NDCG@20: 0.118625 vs. 0.116862. Test pareado por usuario: diferencia media -0.001763, -1.74 sigma (no cruza el umbral estricto de 2 sigma, pero el bootstrap 95% CI [-0.003725, +0.000247] es casi todo negativo y P(embeddings > tfidf) = 3.75%). No cumple ni el criterio mínimo del proyecto para escalar a CV completo de 3 seeds (positivo en un seed), así que no se corrió CV ni se gastó una submission. Costo real medido: 2412s (40 min, incluye descarga del modelo + codificar ~48k resúmenes en CPU) vs. 805s (13 min) de TF-IDF -- ni siquiera hubiera sido gratis de mantener. Código revertido en su totalidad (`ranker.py`, `pyproject.toml`, `.gitignore`) tras el resultado -- a diferencia de país/franja de nacimiento (que se dejaron sin usar "por si sirve con otro enfoque"), acá no quedó nada en el repo, solo esta entrada. Hipótesis no confirmada de por qué no ayudó: un modelo de embeddings de propósito general sobre resúmenes cortos parece capturar similitud de tema/género en general, más difusa que el matching léxico específico (nombres, escenarios, vocabulario puntual) que parece ser la señal que efectivamente funciona acá. |

## 16. 7ª fuente de candidatos: libros de editoriales ya leídas (ronda 2026-09-01)

| Decisión | Estado sugerido | Detalle |
|---|---|---|
| Agregar una 7ª fuente de candidatos por editorial ya leída, mismo patrón exacto que la 4ª fuente (autor) | ❌ **descartado -- recall casi no sube, pese a una oportunidad teórica grande** | Motivada por medir (mismo método que justificó autor): 50.6% de los libros objetivo son de una editorial ya leída (más que el 31.9% de autor, control) y 24% son de editorial ya leída **sin** ser también de autor ya leído -- no redundante con la 4ª fuente en el papel. `n_libros_editorial_catalogo` ya era la 3ª feature más importante por `gain`, señal adicional de que reforzaba la hipótesis. Implementado como mirror exacto de la fuente de autor (`score_editorial_candidato`/`rank_editorial_candidato`/`en_editorial_candidato`, 35→38 features, mismo tope total a `n_por_fuente` priorizando las editoriales más leídas). Resultado del chequeo de recall (seed=42, mismo criterio que descartó `n_por_fuente=500` sin correr CV completo): recall 0.5115→0.5167 (+1.0%, muy por debajo del 24pp de oportunidad teórica), NDCG@20 0.118625→0.118847 (+0.19%, muy por debajo incluso del error estándar pareado ~0.0008 del proyecto), eficiencia de ranking 0.2319→0.2300 (-0.8%). No cumplió el criterio mínimo para correr el CV completo de 3 seeds, así que no se corrió ni se gastó una submission. Hipótesis no confirmada de por qué la oportunidad teórica no se tradujo en recall real: a diferencia de autor (catálogos chicos, el top-20 por popularidad global cubre casi todo lo que escribió un autor), una editorial tiene un catálogo mucho más disperso (`decisiones.md` sección 8: 91% de las editoriales con interacciones tiene menos de 20 libros, pero las editoriales grandes tienen colas largas) y sus libros más populares probablemente ya los traían ALS/popularidad global -- la fuente sumó ~60 candidatos/usuario en promedio pero en su mayoría redundantes, no candidatos nuevos que capturaran el target específico. Código revertido en su totalidad (`ranker.py`, `submit.py`, `tests/test_ranker.py`, `features.md`) tras el resultado -- mismo criterio que el experimento de embeddings, no queda nada en el repo salvo esta entrada. |

## 17. Infraestructura: caché de contexto + comparación de generadores de candidatos completos (ronda 2026-09-02)

| Decisión | Estado sugerido | Detalle |
|---|---|---|
| `fuentes_activas` (nuevo parámetro de `generar_candidatos_con_features`/`preparar_pipeline`, `FUENTES_CANDIDATOS = {"als", "popularidad", "genero", "autor", "resumen", "coleido"}`) para apagar una fuente de candidatos entera | ✅ **implementado, sin regresión** | Con `fuentes_activas=None` (default) el comportamiento es idéntico al de antes de este cambio -- solo gatea el bloque que agrega candidatos NUEVOS de una fuente, no las features de historial que ya existían antes de que esa fuente propusiera candidatos (`score_coleido`/`sim_resumen_historial` se siguen calculando siempre). Tests dedicados en `tests/test_ranker.py`. |
| `scripts/comparar_generadores_pareado.py` (nuevo) -- mismo aparato estadístico que `comparar_features_pareado.py`, pero compara dos generadores de candidatos completos (dos contextos, uno por `fuentes_activas`) en vez de dos subconjuntos de `FEATURES` sobre un único contexto | ✅ **implementado y corrido con éxito una vez sobre datos reales** | Resuelve el pendiente anotado tras la ronda de la 6ª fuente (kNN): antes no había forma de aislar con el poder estadístico del test pareado si una fuente *en sí* (no solo sus 3 features de tracking) mueve el NDCG. `ndcg_por_usuario`/`recall_de_candidatos` se extrajeron a `ranker.py` como funciones públicas reusables (antes vivían duplicadas/inline en los scripts). |
| `preparar_pipeline_cacheado` -- cachea a disco (`data/cache/`, gitignored) el contexto de `preparar_pipeline` por `seed`/`n_por_fuente`/`n_por_autor`/`k`/`fuentes_activas` + hash del código de `ranker.py` | 🔄 **implementado y corregido, pero sin número de tamaño/velocidad confirmado sobre datos reales esta sesión** | Un primer intento sin tratamiento especial pesaba 3-4 GB por configuración (`candidatos_test` tiene millones de filas con `id_lector`/`id_libro` repetidos sin deduplicar) y una recarga se colgó. Corregido con `_comprimir_para_cache`/`_descomprimir_de_cache` (esas dos columnas a `category` antes de picklear, revertidas a `str` al cargar) -- >2x de reducción confirmado con un test sobre datos sintéticos de forma realista (`tests/test_cache_pipeline.py`), pero no se pudo confirmar el número final sobre el contexto real completo en esta sesión: cinco intentos de correrlo (background directo y foreground con auto-promoción a background) terminaron en estado `killed` por la infraestructura de la sesión, sin ninguna traza de error del código -- ver `bitacora.md` para el detalle. Pendiente: confirmar corriendo `scripts/recall_candidatos.py` dos veces seguidas fuera de este entorno. |
| Validación retroactiva de la 6ª fuente (co-lectura ítem-ítem/kNN) con el test pareado de generadores | 🔄 **lectura más rigurosa que "positivo en los 3 seeds", pero no concluyente por sí sola** | Única corrida que completó (antes de la corrección del caché): recall y NDCG@20 idénticos a los ya logueados (0.4559→0.5115, 0.117503) -- confirma que `fuentes_activas` reproduce exactamente lo que antes requería revertir código a mano. Test pareado: diferencia +0.001121, 1.35σ, bootstrap 95% CI [-0.000506, +0.002759] (roza el cero), P(mejora)=91%. Más fuerte que los tres casos límite de rondas anteriores que la re-auditoría de `modelo_actual.md` mostró indistinguibles de ruido (0.3σ, 0.3σ, 0.04σ), pero no cruza el umbral estricto de 2σ. Coherente con la confirmación ya obtenida en Kaggle (+1.56%). Ver `bitacora.md`. |

---

## EDA: qué se usó y qué no (para tu feedback)

Resumen rápido de qué hallazgos del EDA (`experiments/eda.md`) terminaron
metidos en el código y cuáles quedaron sin explotar — separado acá porque
lo preguntaste aparte:

**Usados:**
- Sparsity 99.91% / cola larga en usuarios y libros → motivó tanto el
  fallback jerárquico de v1 como, en general, la idea de que un modelo
  con señal usuario-item (v2) tenía mucho margen para mejorar sobre
  popularidad pura.
- Rating 1-10 (no 1-5), con variación real → decisión directa de v2: usar
  el rating como confianza en vez de binarizar.
- Género literario con 100% de cobertura de usuarios y 66→55 valores
  tras normalizar → base entera del modelo v1 (`genero_preferido_por_usuario`,
  normalización de capitalización).
- Franja de nacimiento cubriendo ~70% de lectores (30.4% sin
  `nacimiento`) → segundo fallback de v1, con ese % de cobertura
  esperado y confirmado en la corrida real (71.6% en val).
- Rating promedio parejo entre géneros (6.8–7.6) → validó que el score
  bayesiano por género no iba a distorsionarse raro entre géneros
  distintos.

**Usados (actualización 2026-08-30):**
- `autor`/`editorial` de libros: no se implementó el modelo content-based
  separado que sugería el EDA, pero ambos terminaron como *features* del
  ranker v3 (`en_autor_leido`/`n_libros_autor_leidos`,
  `en_editorial_leida`/`n_libros_editorial_leidos`) -- una realización
  más liviana de la misma idea (reordenar candidatos con esa señal, en
  vez de generarlos con ella).
- `anio_edicion` (antigüedad de la edición): sí se usa, vía
  `anio_edicion_dif` en el ranker (diferencia contra el promedio de lo
  que lee el usuario).
- `resumen` (texto): TF-IDF + similitud coseno contra el perfil de
  lectura del usuario (`sim_resumen_historial`) -- primera vez que se
  usa una señal de texto en el proyecto.

**No usados todavía (siguen sobre la mesa):**
- El 60.76% del catálogo sin metadata — se concluyó que "no afecta al
  modelo" porque son libros sin interacciones, pero no se revisó de
  nuevo con ALS (que sí podría, en teoría, recomendar libros de esa cola
  si tuvieran alguna interacción rara — vale la pena confirmar que no
  está pasando).
- `vive_en` (ubicación del lector) — nunca se usó como señal en ningún
  modelo, ni se investigó en el EDA más allá de la calidad de datos.
- El género v1 (`libros.genero`) granular y la franja de nacimiento
  **siguen sin llegar a v2** (ALS, ver punto 3 de la tabla de arriba).
  El género sí llega a v3 (ranker) por dos caminos: el existente
  (`score_genero`/`rank_genero`, género preferido del usuario) y el
  nuevo macro-género (`popularidad_genero_macro_candidato`/
  `frecuencia_genero_macro_usuario`, ver sección 7) -- pero franja de
  nacimiento sigue sin usarse fuera de v1.
