\# recsys-libros



\## Qué es esto

Sistema de recomendación de libros para una competencia de Kaggle, evaluada con NDCG@k.

Datos en data/raw/data.db (sqlite), 3 tablas:

\- interacciones(id\_lector, id\_libro, fecha, rating)

\- lectores(id\_lector, nombre, genero, vive\_en, nacimiento)

\- libros(id\_libro, titulo, autor, genero, editorial, anio\_edicion, isbn, resumen, img\_src)



Formato de entrega: data/raw/ejemplo.csv muestra el formato exacto — columnas id\_lector,id\_libro,

k filas por usuario, EL ORDEN IMPORTA (es el ranking, primera fila = más recomendado).



\## Para retomar contexto

\- Empezá por `experiments/estado_del_arte.md`: modelo actual, cómo se valida, qué se probó y no funcionó, y el problema abierto. Es lo único que hace falta leer para retomar.

\- `experiments/legacy/` es el registro histórico completo (bitácora ronda por ronda, decisiones #1–26, análisis técnico), congelado al 2026-09-07. Se consulta bajo demanda, no se lee entero.

\- `experiments/log.csv` es el ledger: una fila por corrida, NDCG local vs Kaggle + nota.



\## Comandos clave

\- `uv run pytest` corre los tests

\- `uv run python -m src.recsys.submit --model popularity` genera un csv en outputs/submissions/



\## Convenciones

\- Split train/val: leave-one-out por usuario (nunca aleatorio global)

\- Nunca calcular estadísticas de popularidad usando datos de validación (data leakage)

\- Todo modelo nuevo loguea su NDCG@k local en experiments/log.csv antes de subir a Kaggle



\## No hacer

\- No commitear data/raw/data.db ni csvs pesados de outputs/

