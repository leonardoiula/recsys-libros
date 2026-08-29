# Decisiones — qué se conserva, qué se reconsidera

Índice de las decisiones de diseño tomadas en v0/v1/v2, para revisar qué
sigue teniendo sentido, qué se llevó a la versión siguiente y qué quedó
sin resolver. Es un complemento *de referencia rápida* a
`experiments/bitacora.md` (que tiene el razonamiento completo de cada
una) — acá no se repite el "por qué" en detalle, solo se linkea a dónde
está.

Columna **Estado**: es una sugerción de lectura, no un veredicto —
la idea es que la marques vos como ✅ conservar / ❌ sacar / 🔄 revisar.

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
| Validación cruzada (3 seeds) protege contra sobreajustar a un split, pero no corrige sesgo sistemático del proxy local | v2/v3 | 🔄 **hallazgo nuevo, sin resolver** | Al re-evaluar ALS original vs tuneado con la misma CV de 3 seeds, el número local *siguió* favoreciendo a la config tuneada (0.1006 vs 0.0944) pese a ser peor en Kaggle real. La CV arregla un problema (ruido de un split) pero no otro (sesgo compartido por todos los splits locales frente a la tarea real) -- posiblemente relacionado con la composición de población de `ejemplo.csv` ya detectada. Pendiente de investigar en serio. |
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
| Ranker reconstruido sobre el ALS original (revertido) | v3 | ❌ **hipótesis refutada** | Confirmado en Kaggle: 0.03815, **-1.3%** respecto a su propia base (ALS original solo, 0.03864) -- no la mejora esperada. El uplift del ranker no es estable: ayuda sobre una base débil (+7.1%) pero no sobre una fuerte (-1.3%). `als` sola sigue siendo la mejor entrega confirmada. Ver `bitacora.md`, sección "Ranker: el uplift no es estable". |
| Seguir tuneando LightGBM / agregando features al ranker | v3 | 🔄 **pausado a propósito** | No conviene invertir más en el ranker sin resolver primero el sesgo sistemático del proxy local (fila de arriba) -- cualquier mejora local corre el mismo riesgo de no sostenerse en Kaggle, como ya pasó dos veces (ALS+optuna y ranker sobre ALS original). |

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

**No usados todavía (siguen sobre la mesa):**
- La idea de EDA/bitácora de un **modelo content-based** con `autor` y
  `editorial` de libros para cold-start (usuarios con pocas
  interacciones, donde ALS tiene poca señal) — nunca se implementó.
- El 60.76% del catálogo sin metadata — se concluyó que "no afecta al
  modelo" porque son libros sin interacciones, pero no se revisó de
  nuevo con ALS (que sí podría, en teoría, recomendar libros de esa cola
  si tuvieran alguna interacción rara — vale la pena confirmar que no
  está pasando).
- `vive_en` (ubicación del lector) — nunca se usó como señal en ningún
  modelo, ni se investigó en el EDA más allá de la calidad de datos.
- `anio_edicion` de los libros (antigüedad de la edición) — tampoco se
  exploró como señal, ni en el EDA ni en ningún modelo.
- El género v1 (`libros.genero`) y la franja de nacimiento **no llegan a
  v2** (ver punto 3 de la tabla de arriba) — es la brecha más directa
  entre "lo que dijo el EDA que servía" y "lo que efectivamente usa el
  modelo más nuevo".
