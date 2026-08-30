# Catálogo de features del ranker

Lista única y legible de las 23 features que usa `"ranker"` (`FEATURES`
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

---

## Qué no está (candidatos para la próxima ronda)

Metadata disponible en el dataset que **todavía no se usa** en ninguna
feature del ranker (ver también `decisiones.md`, sección "EDA: qué se
usó y qué no"):

- `vive_en` (ubicación del lector) — nunca explorada como señal.
- Franja de nacimiento / década de nacimiento del lector — se probó en
  v1 (`popularity_segmentada.py`) pero no llegó al ranker.
- Metadata cruzada lector↔libro más allá de género/autor/editorial (ej.
  coincidencia de idioma, o señales de `isbn`/`img_src`, sin explorar).
