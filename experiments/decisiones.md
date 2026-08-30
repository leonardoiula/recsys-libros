# Decisiones — qué se conserva, qué se reconsidera

Índice de las decisiones de diseño tomadas en v0/v1/v2, para revisar qué
sigue teniendo sentido, qué se llevó a la versión siguiente y qué quedó
sin resolver. Es un complemento *de referencia rápida* a
`experiments/bitacora.md` (que tiene el razonamiento completo de cada
una) — acá no se repite el "por qué" en detalle, solo se linkea a dónde
está.

Columna **Estado**: es una sugerción de lectura, no un veredicto —
la idea es que la marques vos como ✅ conservar / ❌ sacar / 🔄 revisar.

## Para retomar: próximos pasos pendientes (al 2026-08-30)

Estado actual: `"ranker"` (v3, ALS+género+popularidad+autor/año/diversidad/
recencia+co-lectura+editorial+resumen+popularidad/frecuencia de
macro-género+tamaño de catálogo de editorial → LightGBM, 23 features,
hiperparámetros conservadores) es el modelo de referencia del proyecto,
con **0.04831 confirmado en Kaggle** (récord actual). Ideas concretas
para la próxima sesión, en orden sugerido:

1. **Seguir sumando features de dominio** — con el tuneo de
   hiperparámetros descartado tres veces (ver resuelto abajo), el margen
   real sigue estando ahí, no en LightGBM. Candidatos sin explorar
   todavía: ver `experiments/features.md`, sección "Qué no está"
   (`vive_en`, franja de nacimiento, señales cruzadas lector↔libro).
2. **Investigar más a fondo la brecha local-vs-Kaggle** — sigue sin
   resolverse del todo (ver sección de split y validación local, más
   abajo).

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
