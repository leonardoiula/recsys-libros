# Historia (congelada al 2026-09-07)

Estos 3 documentos son el **registro histórico completo** del razonamiento del proyecto,
de v0 (popularidad, 0.01024 en Kaggle) al récord **0.06316**. Se consultan **bajo
demanda** (para el porqué de una decisión puntual, o el detalle de un experimento) — no
se leen enteros para retomar contexto.

Para el estado actual del proyecto empezá por **`experiments/estado_del_arte.md`**.

| Archivo | Qué tiene |
|---|---|
| `bitacora.md` | Narrativa Objetivo / Implementación / Resultado / Reflexión de cada ronda, en orden cronológico. ~3.600 líneas. |
| `decisiones.md` | Tabla numerada **#1–26** de decisiones de diseño (qué se conserva ✅ / qué se descartó ❌ / qué revisar 🔄), + el bloque "Para retomar" con la investigación abierta sobre el límite del reranking (la forma de U). |
| `modelo_actual.md` | Descripción técnica más detallada del ranker de 2 etapas + los análisis dedicados "¿cambiar de paradigma?" y "¿incorporar agentes de IA al flujo?". |

Las referencias entre estos 3 archivos con nombre pelado (`` `decisiones.md` ``, etc.)
resuelven dentro de esta carpeta. Referencias desde código/otros docs usan
`experiments/legacy/...`.

Si el proyecto avanza a un récord mejor y `estado_del_arte.md` absorbe lo esencial de una
ronda posterior, ese material también se archiva acá.
