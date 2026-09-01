# Catálogo de features del ranker

Lista única y legible de las 32 features que usa `"ranker"` (`FEATURES`
en `src/recsys/models/ranker.py`), para revisar de un vistazo qué hay y
pensar qué falta o vale la pena mejorar. El detalle de *por qué* se
agregó cada una, los resultados de validación cruzada y de Kaggle están
en `experiments/decisiones.md` (secciones 6, 7 y 8) y
`experiments/bitacora.md` — acá no se repite eso, solo qué mide cada
feature y cómo se calcula.

Todas se calculan **solo con `train_candidatos`** (nunca con datos que
el ranker vea como etiqueta) excepto donde se aclara que son metadata
estática del libro (autor/editorial/año/resumen — no dependen del
split). Un candidato sin dato conocido queda con un valor sentinel, no
se imputa a ciegas (ver columna "Si falta").

> Mantener actualizado: cada vez que se agregue o saque una feature de
> `FEATURES`, actualizar esta tabla en el mismo cambio.

## Candidatos de ALS (filtrado colaborativo)

| Feature | Qué mide | Si falta |
|---|---|---|
| `score_als` | Score de `implicit.als` para este candidato — mayor score, más afinidad según el modelo colaborativo. | `0.0` |
| `rank_als` | Posición del candidato dentro del top-N que devuelve ALS (0 = primero). | `n_por_fuente` (justo afuera de la ventana) |
| `en_als` | 1/0 — ¿ALS propuso este candidato? | `0` |

## Candidatos de popularidad global

| Feature | Qué mide | Si falta |
|---|---|---|
| `score_popularidad` | Score bayesiano de popularidad global del libro (`fit_popularity`: shrinkage hacia el rating promedio global si tiene pocas interacciones). | `0.0` |
| `rank_popularidad` | Posición del libro en el ranking global de popularidad. | `n_por_fuente` |
| `en_popularidad` | 1/0 — ¿vino de esta fuente? | `0` |

## Candidatos de autores ya leídos (4ª fuente — ver `modelo_actual.md`, sección "Recomendación: ¿cambiar de paradigma?")

Para cada autor que el usuario ya leyó, hasta `n_por_autor` (20 por
default) libros sin leer de ese autor, rankeados por el score de
popularidad **global** (no se refittea un score bayesiano por autor —
la mayoría tiene muy pocas interacciones para un shrinkage propio
confiable). Motivada por medir que 28.6% de los libros objetivo son de
un autor ya leído, señal que antes solo existía como *feature*
(`en_autor_leido`/`n_libros_autor_leidos`, más abajo) y nunca proponía
candidatos nuevos por sí sola.

| Feature | Qué mide | Si falta |
|---|---|---|
| `score_autor_candidato` | Score de popularidad global del candidato (mismo score que `score_popularidad`) — reusado acá para rankear dentro del catálogo de cada autor. | `0.0` |
| `rank_autor_candidato` | Posición del candidato dentro del top-`n_por_autor` de *ese autor específico* (0 = el más popular sin leer). | `n_por_autor` (justo afuera de la ventana de esta fuente) |
| `en_autor_candidato` | 1/0 — ¿esta fuente propuso el candidato? Distinto de `en_autor_leido` (que mide historial del usuario con el autor, sin importar qué fuente trajo el candidato). | `0` |

## Candidatos por similitud de resumen (5ª fuente — ver `modelo_actual.md`, sección "Recomendación: ¿cambiar de paradigma?")

Para cada usuario, el top-`n_por_fuente` de libros de **todo el
catálogo con resumen** (~48.320 libros) más similares a su perfil de
lectura (TF-IDF, mismo perfil que arma `_calcular_perfil_texto` para
`sim_resumen_historial`, ver más abajo). Motivada por medir que los
libros objetivo que las otras 4 fuentes fallan en capturar son ~11x
menos populares (mediana de interacciones) que los que sí capturan —
la única de las 5 fuentes que no depende de cuánta gente más leyó un
libro, solo de su contenido. Se procesa en lotes de usuarios
(`TAMANO_LOTE_RESUMEN`) para no materializar un producto denso
usuarios×libros completo de una sola vez (ver docstring de esa
constante en `ranker.py`).

| Feature | Qué mide | Si falta |
|---|---|---|
| `score_resumen_candidato` | Similitud coseno TF-IDF entre el resumen del candidato y el perfil de lectura del usuario (mismo cálculo que `sim_resumen_historial`, pero buscando en todo el catálogo, no solo entre los candidatos que ya trajeron otras fuentes). | `0.0` |
| `rank_resumen_candidato` | Posición del candidato dentro del top-`n_por_fuente` por similitud (0 = más similar). | `n_por_fuente` (justo afuera de la ventana de esta fuente) |
| `en_resumen_candidato` | 1/0 — ¿esta fuente propuso el candidato? Distinto de `sim_resumen_historial` (que nunca propone candidatos nuevos por sí sola, solo puntúa los que ya llegaron de otra fuente). | `0` |

## Candidatos de popularidad por género preferido del usuario

| Feature | Qué mide | Si falta |
|---|---|---|
| `score_genero` | Score bayesiano de popularidad, calculado *solo* dentro del género literario preferido del usuario. | `0.0` |
| `rank_genero` | Posición del libro en ese ranking (acotado al género del usuario, no al macro-género — ver más abajo). | `n_por_fuente` |
| `en_genero` | 1/0 — ¿vino de esta fuente? | `0` |

## Volumen bruto (sin shrinkage)

| Feature | Qué mide |
|---|---|
| `n_interacciones_libro` | Cantidad total de interacciones que tiene el libro en `train_candidatos` — popularidad cruda, sin el shrinkage bayesiano de `score_popularidad`. |
| `n_interacciones_usuario` | Cantidad total de interacciones del usuario — actividad/tamaño de su historial. |

## Autor

| Feature | Qué mide | Si falta |
|---|---|---|
| `en_autor_leido` | 1/0 — ¿el usuario ya leyó algún libro de este autor? | `0` |
| `n_libros_autor_leidos` | Cuántos libros de este autor ya leyó el usuario. | `0` |

## Año de edición

| Feature | Qué mide | Si falta |
|---|---|---|
| `anio_edicion_dif` | Año de edición del candidato menos el promedio de lo que lee el usuario (positivo = más nuevo que lo habitual; negativo = más antiguo/clásico). | `0.0` (usuario sin año conocido, o candidato sin año) |

## Diversidad de género (historial)

| Feature | Qué mide |
|---|---|
| `n_generos_distintos_usuario` | Cantidad de géneros literarios distintos (normalizados) que leyó el usuario — generalista vs. especializado. No depende del candidato, es el mismo valor para todos sus candidatos. |

## Recencia

| Feature | Qué mide | Si falta |
|---|---|---|
| `dias_desde_ultima_interaccion_usuario` | Días desde la última interacción del usuario, relativo a la fecha más reciente de `train_candidatos`. | `SENTINEL_DIAS_DESCONOCIDO` (99999) |

## Co-lectura ítem-ítem

| Feature | Qué mide | Si falta |
|---|---|---|
| `score_coleido` | Cuántos usuarios leyeron *tanto* algún libro del historial del propio usuario *como* este candidato — señal tipo "quien leyó X también leyó Y", vía matriz dispersa ítem×ítem sobre la matriz binaria de ALS. | `0.0` (usuario sin fila ALS, o sin co-lectura) |

## Editorial (historial del usuario)

| Feature | Qué mide | Si falta |
|---|---|---|
| `en_editorial_leida` | 1/0 — ¿el usuario ya leyó algún libro de esta editorial? | `0` |
| `n_libros_editorial_leidos` | Cuántos libros de esta editorial ya leyó el usuario. | `0` |

## Resumen / texto

| Feature | Qué mide | Si falta |
|---|---|---|
| `sim_resumen_historial` | Similitud coseno (TF-IDF) entre el `resumen` del candidato y el perfil de lectura del usuario (centroide normalizado de los resúmenes de lo que leyó). | `0.0` (usuario sin perfil de texto, o candidato sin resumen) |

## Género macro (10 familias de dominio — ver `decisiones.md` sección 7)

| Feature | Qué mide | Si falta |
|---|---|---|
| `popularidad_genero_macro_candidato` | Score bayesiano de popularidad pooleado a nivel del macro-género del candidato (no del género granular, ni del preferido del usuario) — propiedad del libro: qué tan bien valorada está su familia temática en general. | `0.0` |
| `frecuencia_genero_macro_usuario` | Proporción (0–1) de las interacciones con género conocido del usuario que caen en el macro-género del candidato — señal graduada de afinidad, distinta de `n_generos_distintos_usuario` (que es diversidad, no afinidad a un género puntual). | `0.0` |

## Editorial (tamaño de catálogo — ver `decisiones.md` sección 8)

| Feature | Qué mide | Si falta |
|---|---|---|
| `n_libros_editorial_catalogo` | Cantidad de libros que tiene la editorial del candidato en *todo* `libros` (no solo los leídos por algún usuario) — señal de volumen/reconocimiento de la editorial, independiente del historial de cada usuario. | `0` |

## Señales cruzadas lector↔libro (ronda 2026-08-31 — ver `decisiones.md` sección 11)

`lectores.genero` es el género **declarado del lector** (Mujer/Hombre) —
OJO, no confundir con `libros.genero` (género literario), mismo nombre
de columna pero significado completamente distinto.

| Feature | Qué mide | Si falta |
|---|---|---|
| `popularidad_genero_lector_candidato` | Score bayesiano de popularidad del candidato, calculado *solo* con interacciones de lectores que declararon el mismo género que el usuario (Mujer/Hombre/desconocido). Mismo patrón que `popularidad_pais_candidato`/`popularidad_franja_candidato`: el segmento lo define el usuario. | `0.0` |
| `frecuencia_genero_macro_por_genero_lector` | Proporción (0–1) de las interacciones de la *cohorte* que declaró el mismo género que el usuario que caen en el macro-género del candidato — a diferencia de `frecuencia_genero_macro_usuario` (historial *individual*), acá es el patrón agregado del grupo. | `0.0` |
| `edad_lector_al_publicarse` | `anio_edicion` del candidato menos `nacimiento` del usuario — años de vida que tenía el lector cuando se publicó el libro (negativo = se publicó antes de que naciera). Cruza una propiedad del lector con una del libro directamente, a diferencia de `anio_edicion_dif` (que compara contra el promedio de lectura del usuario, no contra su nacimiento). | `0.0` (usuario sin `nacimiento` válido -- inválido o sentinel `1910`, ver `popularity_segmentada.NACIMIENTO_SENTINEL` -- o candidato sin `anio_edicion`) |

---

## Qué no está (candidatos para la próxima ronda)

Metadata disponible en el dataset que **todavía no se usa** en ninguna
feature del ranker (ver también `decisiones.md`, sección "EDA: qué se
usó y qué no"):

- Señales cruzadas lector↔libro basadas en `isbn`/`img_src` (ej. país/idioma
  implícito en el prefijo de grupo del ISBN vs. `vive_en` del lector) —
  evaluado y descartado de entrada sin implementar: 61% de los libros no
  tiene ISBN, y del resto, 94% es del grupo "84" (España) -- casi sin
  varianza, mismo problema de sesgo que `vive_en`/editorial pero peor.
- `vive_en` (ubicación del lector) — **se probó y se descartó** como
  `popularidad_pais_candidato` (popularidad segmentada por país del
  usuario, mismo patrón que macro-género): empeoró el ranker en 2 de 3
  seeds (ver `decisiones.md` sección 9 y `bitacora.md`). `pais_por_usuario`/
  `fit_popularity_por_pais` quedan en `popularity_segmentada.py`
  (testeadas, sin usar en el ranker) por si vale la pena retomar la idea
  con otro enfoque (otra granularidad, cruzarla con otra señal).
- Franja de nacimiento del lector — **se probó y se descartó** como
  `popularidad_franja_candidato` (popularidad segmentada por década de
  nacimiento, mismo patrón que país): mixto y negativo en 2 de 3 seeds
  (ver `decisiones.md` sección 10 y `bitacora.md`). De paso se corrigió
  un sentinel de datos (`nacimiento == 1910` es casi seguro un default
  de formulario, no gente real -- ver `popularity_segmentada.NACIMIENTO_SENTINEL`),
  que sí queda en el código porque mejora la calidad del dato para v1
  independientemente de esta feature puntual. `franja_nacimiento_por_usuario`/
  `fit_popularity_por_franja_nacimiento` quedan sin usar en el ranker.
- `vive_en` (ubicación del lector) — **se probó y se descartó** como
  `popularidad_pais_candidato` (popularidad segmentada por país del
  usuario, mismo patrón que macro-género): empeoró el ranker en 2 de 3
  seeds (ver `decisiones.md` sección 9 y `bitacora.md`). `pais_por_usuario`/
  `fit_popularity_por_pais` quedan en `popularity_segmentada.py`
  (testeadas, sin usar en el ranker) por si vale la pena retomar la idea
  con otro enfoque (otra granularidad, cruzarla con otra señal).
