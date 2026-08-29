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
| `frac_val=0.2` (20% de las interacciones de cada usuario a val, no un número fijo) | v0 | v0, v1, v2 | 🔄 **revisar — prioridad alta** | Funcionó bien mientras los modelos eran heurísticas de popularidad (v0: local sobreestima Kaggle +2.7%, v1: +44.8%), pero con ALS (señal usuario-item real) la sobreestimación saltó a **+573%** — un usuario con mucho historial termina con decenas de libros "relevantes" simultáneos en val, tarea mucho más fácil que la de Kaggle. Ver `bitacora.md`, sección v2 "Resultado". Alternativa propuesta ahí: leave-one-out literal (1 libro por usuario) o un tope fijo de libros en val por usuario. |
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
| Normalización de `libros.genero` (capitalización/espacios) antes de agrupar | v1 (a partir del EDA) | v1 | 🔄 revisar — **no llega a v2** | ALS no usa género en absoluto. Si se retoma la idea de combinar señales (ver sección 5), esta normalización habría que revivirla. |
| Franja de nacimiento (década, no edad real) como proxy demográfico | v1 (a partir del EDA) | v1 | 🔄 revisar — **no llega a v2** | Mismo caso que género: cubre ~70% de usuarios en v1, pero ALS no la usa. Sin uso actual fuera de v1. |
| Cadena de fallback secuencial género → franja → global, con backfill sin duplicar | v1 | v1 | 🔄 revisar | v2 (ALS) implementa un fallback mucho más simple (solo ALS → global, dos niveles, sin género/franja intermedios). Vale la pena decidir si conviene que ALS también pase por género/franja antes de caer a global, o si de verdad no aporta una vez que hay señal usuario-item real. |

## 4. Arquitectura de código (`submit.py` / modelos)

| Decisión | Versión | Sigue vigente en | Estado sugerido | Detalle |
|---|---|---|---|---|
| `MODELOS: dict[str, Callable[[usuarios, k], dict]]` — cada modelo es una función `(usuarios, k) -> {id_lector: [id_libro,...]}`, ya entrenada con todos los datos | v0 (ranking único) → extendido en v1 (ranking por usuario) | v0, v1, v2 | ✅ conservar | Extensión limpia: v0 devuelve el mismo ranking recortado por usuario, v1/v2 devuelven ranking ya personalizado. `--model` en la CLI queda automático (`choices=sorted(MODELOS)`) — no hubo que tocar el CLI para agregar `als`. |
| Cada `fit_*`/modelo recibe los datos como parámetro (nunca un DB global implícito), para poder reusarse igual con train (validación local) o con todos los datos (submission) | v0 | v0, v1, v2 | ✅ conservar | Es lo que permite que el mismo código sirva para evaluar y para generar la entrega, sin dos implementaciones paralelas. |

## 5. Modelo ALS (v2)

| Decisión | Versión | Estado sugerido | Detalle |
|---|---|---|---|
| Usar el rating explícito (1-10) como peso/confianza de `implicit.als`, en vez de binarizar "leyó/no leyó" | v2 | ✅ conservar (por ahora) | Directamente informado por el EDA: rating es 1-10 siempre positivo, con variación real (media 7.26, std 1.82) — no todo 10, así que binarizar tiraría señal. Alternativa descartada explícitamente: binarizar para usar el modelo implícito "puro". |
| Hiperparámetros `factors=128, regularization=0.1, iterations=20`, elegidos por sweep local (7 configs) | v2 | 🔄 **revisar** | Elegidos optimizando **NDCG local**, que según el punto 1 de esta tabla sobreestima fuertemente a Kaggle para modelos con señal usuario-item real. No hay garantía de que estos hiperparámetros sean los que mejor generalizan a Kaggle — el sweep necesitaría rehacerse una vez que el split local sea más representativo, o directamente comparar 2-3 configs vía submissions reales a Kaggle. |
| Fallback a popularidad global (v0) para usuarios sin fila en la matriz (cold start real) | v2 | ✅ conservar | No se disparó ni una vez en el split local (los 6,770 usuarios de val tuvieron fila), pero cubre el caso general sin costo. |
| No combinar ALS con género/franja (v1) ni con metadata de libro (`autor`, `editorial`) | v2 | 🔄 revisar — próximo paso ya anotado | Es la idea "próximo candidato" del final de la sección v2 de la bitácora: usar el score de ALS + popularidad segmentada como features de un ranker (LightGBM). Directamente relacionado con la pregunta de si género/franja "siguen teniendo sentido" — la respuesta probablemente sea "sí, pero como feature adicional, no como modelo standalone". |

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
