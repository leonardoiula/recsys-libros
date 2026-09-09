# comics (subproyecto del TP "recomendación en la web")

## Qué es esto

Análogo de `recsys-libros` pero de comics, con datos obtenidos por web scraping de
[comicbookroundup.com](https://comicbookroundup.com) en vez de un dataset ya armado.
Plan completo de diseño en `~/.claude/plans/synthetic-snuggling-marble.md` (histórico,
puede haber quedado desactualizado respecto de este archivo si algo cambió sobre la marcha).

Datos en `comics/data/raw/comics.db` (sqlite), 4 tablas:

- `comics(id_comic, titulo, serie, numero, editorial, anio_edicion, escritor, dibujante, precio_tapa, img_src, url)`
- `usuarios(id_usuario, nombre)`
- `interacciones(id_usuario, id_comic, fecha, rating, texto)` — reviews de usuarios reales, es el equivalente de `interacciones` en libros. Tiene una columna extra (`texto`) que libros no tiene: es barato de guardar y puede servir para features de contenido/sentiment más adelante.
- `critic_reviews(id_comic, outlet, reviewer, fecha, rating, texto, url_externa)` — reviews de crítica especializada, **no** son lectores con perfil de gustos, por eso están separadas de `interacciones` y no entran en el recomendador colaborativo.

`id_comic` es el path relativo de la URL, ej `"dc-comics/batman-(2025)/1"` — no un ID numérico arbitrario, así queda trazable a la fuente sin tabla auxiliar.

## Qué NO existe en el sitio (a diferencia del dataset de libros)

- No hay demografía de usuario (nada como `genero`/`vive_en`/`nacimiento`). Los `usuarios` son solo `id_usuario` + `nombre`.
- No hay género/categoría del cómic (tipo "Fantasía/Terror") ni sinopsis/resumen — comicbookroundup organiza por editorial, no por género. `libros.genero`/`libros.resumen` no tienen equivalente acá.

## Hallazgos del sitio (confirmados en vivo, no supuestos)

- `robots.txt` solo prohíbe `/signin`, `/search-results`, `/forgot-password` — scrapear el resto está permitido explícitamente. Sin `Crawl-delay`, igual respetamos rate limiting por buena práctica.
- HTML server-rendered, no hace falta Selenium/Playwright.
- Un issue que no existe **no da 404**: el sitio devuelve un 302 a la home. Si dejás que `requests` siga el redirect (default), terminás "parseando" la home como si fuera un comic real. `fetch.obtener_html` detecta esto (`PaginaNoEncontrada`) comparando la URL final contra la pedida.
- El catálogo se recorre por semana de lanzamiento: `/comic-books/release-dates/AAAA-MM-DD`, listando issues de todas las editoriales juntas. Restar 7 días da la semana anterior sin necesidad de parsear el link de navegación de esa página.
- Selectores de parsing verificados contra HTML real (ver `comics/tests/fixtures/`): metadata en `table.issue-info` (filas Writer/Artist/Cover Price), reviews en `<li>` con `div.list-review` (score) + `h3` + `p`; se distingue review de usuario vs. de crítica por la presencia de `div.user-review-details` (usuario) o su ausencia (crítica), no por el contenedor `<ul>` (ambos tipos pueden convivir sin un `<ul>` que los separe con id/clase propia).

## Comandos clave

- `uv run pytest comics/tests` corre los tests del subproyecto (no pegan a la red, usan fixtures de HTML guardado)
- `uv run python comics/scripts/scrape_comics.py --semanas N --desde AAAA-MM-DD` scrapea N semanas hacia atrás desde esa fecha. Reanudable: correrlo de nuevo salta los issues ya guardados (`db.existe_comic`).
- `uv run python comics/scripts/descargar_tapas.py` descarga a `comics/static/covers/` las imágenes de tapa (`comics.img_src`) de lo que ya esté en la BD, para el front Flask. Reanudable (salta las que ya existen en disco). Son imágenes con copyright de las editoriales -- uso académico no comercial, no redistribuir el dataset de imágenes.

## Estado actual del dataset (actualizar a mano después de cada corrida grande)

- Piloto de validación (3 semanas, 2026-09-02 a 2026-08-19): 158 comics, 86 usuarios, 352 interacciones, 0 errores. 60% de usuarios calificó ≥2 series distintas.
- Escalado (18 semanas, 2026-09-02 a 2026-05-06): 926 comics, 169 usuarios, 2958 interacciones, 0 errores. 60-68.6% de usuarios con ≥2 series distintas.
- Escalado 2 (44 semanas totales, hasta 2025-11-05): **2230 comics, 321 usuarios, 9141 interacciones, 0 errores.** 200/321 usuarios (62.3%) con ≥2 series distintas. Distribución de interacciones por usuario muy sesgada (mediana 5, máximo 418) -- mismo tipo de sesgo de actividad que ya se documentó y corrigió en el proyecto de libros (`pesos_por_actividad`/`evaluar_ndcg_ponderado_por_actividad` en `src/recsys/evaluation.py`); conviene revisar si hace falta el mismo tratamiento acá.
- **Medición de cobertura real del sitio** (`comics/scripts/contar_poblacion_total.py`, cuenta solo páginas de listado semanal, no cada issue): el archivo del sitio tiene **67.035 issues catalogados en 1265 semanas**, desde abril/mayo de 2002 hasta hoy.
- Escalado 3 (96 semanas totales, hasta 2024-11-06): **5029 comics, 645 usuarios, 30.001 interacciones, 0 errores en las 3 corridas grandes.** ≈7.5% del catálogo total del sitio (5029/67.035).
  - Densidad real de la matriz comics×usuarios: **0.925%** -- en línea con Netflix Prize (~1.2%), muy por debajo del ~5-10% que a veces se espera pensando en datasets chicos y curados como MovieLens 100k. La densidad **baja** a medida que se scrapea más historia (1.89% → 1.28% → 1.07% → 0.93% en las 4 mediciones de esta sesión) porque cada semana nueva suma ítems con pocas reviews -- es esperable y no es un problema, ALS/BPR están pensados para matrices así de sparse.
  - **27.3% de los comics no tienen ninguna review de usuario** (long tail real, guardados igual con su metadata) -- van a necesitar fallback a `ranking_global` en el recomendador, mismo patrón que usan los modelos de libros.
- Decisión explícita del usuario: seguir scrapeando más historia si se quiere, pero eso queda para más adelante ("después podríamos filtrar para entrenar si queremos") -- **ahora el foco pasa a portar el pipeline de recomendación** con el dataset actual.
- **Primer NDCG@20 real** (`comics/scripts/evaluar_popularity.py`, baseline de popularidad bayesiana, `C=n.mean()` sin sweep propio, 3 seeds), medido sobre el dataset de 44 semanas: **media=0.008325, desvío=0.000648**. Logueado en `comics/experiments/log.csv`. Es un primer número de referencia, no un techo -- falta al menos: sweep propio de `C` (el de libros no necesariamente generaliza, ver docstring de `models/popularity.py`), un modelo colaborativo (ALS/BPR), y re-correr esta evaluación sobre el dataset ya ampliado (más abajo).
- **Escalado 4** (a pedido del usuario, cobertura hasta 2021-12-29 inclusive -- cubre todo 2022 y más): +149 semanas, 9022 issues nuevos, solo 2 con error (timeouts transitorios de conexión, manejados por el catch-all de `crawl.py` sin abortar la corrida -- quedan pendientes de reintento si se quiere, correr el scraper de nuevo los reintenta solo a ellos). **Dataset final: 14.051 comics, 1.696 usuarios, 114.386 interacciones, 56.967 critic_reviews. Archivo `comics.db`: 63 MB.**
  - Curiosidad no problemática: hay 7 interacciones con fecha anterior a 2021 (la más vieja, 2016) -- son comics clásicos (ej. "Sandman #19", Vertigo, años 90) que aparecieron linkeados incidentalmente dentro de alguna página semanal scrapeada (el extractor de URLs matchea cualquier link con forma de página de issue en el HTML, no solo el grid principal), con reviews de usuarios reales de distintos años. No es un error de parsing, es cobertura extra involuntaria de clásicos.
  - Con 14.051 comics scrapeados, la cobertura del catálogo total del sitio (67.035 issues medidos el 2026-09-05) subió a **≈21%**.
  - Re-corrido `evaluar_popularity.py` sobre el dataset ampliado (10.555 comics con al menos 1 interacción, 1.696 usuarios, 114.386 interacciones): **NDCG@20 media=0.006987, desvío=0.001374** (bajó un poco respecto de 0.008325 con el dataset de 44 semanas -- esperable, no es una regresión: más ítems diluyen el ranking de popularidad, coherente con la caída de densidad que ya veníamos observando al escalar el scraping). Ambas filas quedan logueadas en `comics/experiments/log.csv` para comparar.

## Convenciones (heredadas de `recsys-libros`, ver ese `CLAUDE.md`)

- Split train/val: leave-one-out **temporal** por usuario (nunca aleatorio), igual que en libros — reusar `src/recsys/data.py::split_train_val` adaptado a estos nombres de columna.
- Nunca calcular estadísticas de popularidad usando datos de validación.
- Todo modelo nuevo loguea su NDCG@k local en `comics/experiments/log.csv` (mismas columnas que el de libros: `fecha,modelo,k,frac_val,seed,ndcg_local,ndcg_kaggle,notas` — acá `ndcg_kaggle` probablemente quede siempre vacío si este TP no tiene leaderboard externo).
- Nunca acumular todas las reviews scrapeadas en memoria antes de guardarlas — `db.guardar_pagina_issue` persiste issue por issue a propósito (lección ya aprendida en el proyecto de libros con el `ArrayMemoryError` del ranker).

## No hacer

- No commitear `comics/data/raw/comics.db` ni `comics/data/cache/` (ver `.gitignore`).
- No scrapear sin el rate limiting de `crawl.py` (`espera` entre requests) aunque el sitio no lo exija con `Crawl-delay`.
