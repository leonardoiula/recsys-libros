# Guía didáctica: cómo se construyó el sitio web de comics_recsys

Este documento explica el **por qué** de cada pieza del webapp (`comics/webapp/`),
no solo el qué. Está pensado para acompañar el código real -- cada sección
apunta a archivos y líneas concretas, ya escritos y testeados
(`uv run pytest comics/tests`, 33 tests verdes al momento de escribir esto).
Si algo acá describe un comportamiento que el código ya no tiene, confiá en
el código: esta guía puede desactualizarse, el código es la fuente de verdad.

Idea general del sitio: un usuario se loguea (creando una cuenta nueva, o
reclamando la identidad de un usuario real que ya scrapeamos de
comicbookroundup.com), ve los comics que leyó y recibe recomendaciones. Todo
corre sobre Flask, pensado para desplegarse en el plan gratuito de
PythonAnywhere -- eso empuja varias decisiones de diseño hacia "liviano y sin
sorpresas de performance", como vas a ver más abajo.

## 1. Arquitectura: app factory, blueprints, y cómo llega un request

### App factory en vez de `app = Flask(__name__)` a nivel de módulo

El patrón más simple para "hola mundo" en Flask es:

```python
app = Flask(__name__)

@app.route("/")
def index(): ...
```

Acá en cambio (`comics/webapp/__init__.py:24`) hay una función
`create_app(**config_overrides)` que **construye y devuelve** la app. La
diferencia importa en cuanto necesitás más de una instancia de la app en el
mismo proceso Python -- que es exactamente lo que pasa al testear: cada test
de integración (`comics/tests/test_webapp_integration.py`) arma su propia
base de datos temporal y su propio cache de recomendaciones, y necesita una
app Flask apuntando a *esos* archivos, no a los de producción. Con
`create_app(DB_PATH=..., RECS_CACHE_PATH=...)` cada test consigue una app
limpia y aislada sin tocar variables de entorno globales.

Esto además es justo lo que un servidor WSGI (el protocolo que usa
PythonAnywhere para hablarle a tu app) espera: un objeto `app` ya armado que
pueda importar, no un script que hay que "correr".

### Blueprints: por qué separar `auth.py` de `dashboard.py`

Un blueprint es un grupo de rutas que se registra en la app (`app.register_blueprint(...)`,
ver `__init__.py:37-38`) en vez de definirse todas sobre el objeto `app`
directamente. Acá hay dos: `auth` (login/logout/registro,
`comics/webapp/auth.py`) y `dashboard` (la página principal,
`comics/webapp/dashboard.py`). La razón concreta, no genérica: sin
blueprints, todas las rutas y sus imports (repo, modelos, templates) viven en
un solo archivo que crece sin límite. Separado por área, cada archivo se
puede leer de punta a punta en un minuto.

### El viaje de un request: routing → vista → template

Cuando el navegador pide `GET /`, esto pasa en orden:

1. Flask matchea la URL contra las rutas registradas y encuentra
   `dashboard.index` (`comics/webapp/dashboard.py:16`, decorada con `@bp.route("/")`).
2. Antes de ejecutar la vista, corre el decorador `@login_required`
   (de Flask-Login) -- si no hay sesión válida, corta acá y redirige a
   `/login` (ver sección 2).
3. La función `index()` corre: pide la conexión a la base (`get_db()`),
   consulta lo que el usuario ya leyó (`repo.comics_leidos`) y lo que se le
   recomienda (`recomendador.recomendar` + `repo.comics_por_id`).
4. `render_template("dashboard.html", leidos=..., recomendados=...)` toma la
   plantilla Jinja2 (`comics/webapp/templates/dashboard.html`) y la rellena
   con esos datos, devolviendo el HTML final como respuesta.

Ese ciclo (routing → autenticación → vista → template) es el mismo para
cualquier ruta Flask, no solo esta.

## 2. Autenticación: passwords, cookies de sesión, y la carrera del doble-reclamo

### Por qué nunca se guarda un password en texto plano

`repo.crear_cuenta_nueva` (`comics/webapp/repo.py:48-57`) no guarda el
password que escribe el usuario -- guarda
`generate_password_hash(password)`. Un *hash* de password es el resultado de
una función que es fácil de calcular en un sentido (password → hash) pero
inviable de invertir (hash → password). `werkzeug.security` usa por default
un algoritmo pensado para esto (scrypt/pbkdf2 según versión), que además es
*lento a propósito* -- lento para vos calculando un hash una vez al loguearte,
prohibitivamente lento para alguien que robó la base y quiere probar millones
de passwords por segundo contra ella. Por eso, aunque alguien leyera
`comics.db` entero, no podría "leer" los passwords -- solo podría intentar
adivinarlos hash por hash, muy caro. `verificar_password`
(`repo.py:60-66`) nunca compara el password ingresado contra el guardado
directamente: usa `check_password_hash(hash_guardado, password_ingresado)`,
que recalcula el hash del intento y compara hashes.

### Qué es una cookie de sesión y cómo la usa Flask-Login

HTTP es *stateless* -- cada request es independiente, el servidor no "recuerda"
al navegador entre uno y otro por sí solo. El mecanismo estándar para simular
"estar logueado" es: el servidor manda una cookie con un token firmado
criptográficamente (usando `SECRET_KEY`, ver `comics/webapp/config.py:14`), el
navegador la reenvía automáticamente en cada request siguiente, y el servidor
la valida (la firma prueba que no fue alterada) para saber quién sos sin
volver a pedir el password.

Flask-Login automatiza esto. Tres piezas:

- `login_user(User(...))` (usado en `auth.py`, ej. línea del login exitoso)
  arma esa cookie firmada.
- El decorador `@login_required` (`dashboard.py:15`) chequea la cookie antes
  de correr la vista; si falta o es inválida, redirige a `login_manager.login_view`
  (configurado como `"auth.login"` en `comics/webapp/extensions.py:11`).
- `@login_manager.user_loader` (`comics/webapp/models_user.py:19`) es la
  función que Flask-Login llama en CADA request autenticado para reconstruir
  el objeto `User` a partir del id guardado en la cookie -- por eso no hace
  falta guardar el nombre/perfil completo en la cookie, solo el id, y se
  resuelve el resto contra la base en cada request (`repo.obtener_usuario`).

### La carrera del doble-reclamo

`reclamar_identidad` (`repo.py:32-45`) hace:

```sql
UPDATE usuarios SET password_hash = ? WHERE id_usuario = ? AND password_hash IS NULL
```

en vez de un `UPDATE` sin la condición `password_hash IS NULL`. La razón es
una condición de carrera real: si dos personas encuentran el mismo usuario
"Fulano" sin reclamar y las dos mandan el POST de reclamo casi al mismo
tiempo, solo UNA de las dos consultas va a matchear la condición (la que
llega primero ya puso un `password_hash`, así que la segunda consulta ya no
encuentra ninguna fila con `password_hash IS NULL` para ese id). SQLite
devuelve `cursor.rowcount` = filas afectadas; si es 0, sabemos que perdimos la
carrera (o el id no existe) y lanzamos `ValueError` en vez de pisar
silenciosamente el password de la primera persona. Esto se testea
directamente en `comics/tests/test_webapp_repo.py::test_reclamar_identidad_ya_reclamada_lanza_value_error`.

## 3. Capa de datos: SQL directo, conexión por request, y el JOIN de "ya leíste"

### Por qué SQL directo y no un ORM

Un ORM (Object-Relational Mapper, como SQLAlchemy) te deja escribir
`Usuario.query.filter_by(nombre="x")` en vez de SQL. Es útil cuando el schema
cambia seguido o el equipo prefiere pensar en objetos. Acá el resto del
proyecto (`comics_recsys/db.py`, `comics_recsys/data.py`) ya usa `sqlite3`
puro con SQL a mano -- consistencia con ese estilo, más una razón concreta:
el webapp es liviano a propósito (deploy en PythonAnywhere free), y un ORM es
una dependencia y una capa de indirección más para un sitio con ~6 consultas
totales. `comics/webapp/repo.py` sigue el mismo patrón: funciones que reciben
una conexión ya abierta (`conn: sqlite3.Connection`) y devuelven `dict`/`list[dict]`
(no objetos ORM).

### Conexión por request (`flask.g`)

`comics/webapp/db.py` tiene `get_db()`, que guarda la conexión en
`flask.g` (un objeto que Flask crea nuevo en cada request) y la reutiliza si
ya se pidió antes dentro del mismo request. `close_db()`, registrada con
`app.teardown_appcontext` (`db.py:26-27`), la cierra automáticamente al
terminar el request, se haya usado o no, haya habido excepción o no.

¿Por qué no una sola conexión global reusada entre requests? SQLite no está
pensado para que una misma conexión la usen varios threads/requests a la vez
sin coordinación explícita -- y PythonAnywhere free corre la app en un solo
proceso/worker igual, así que "una conexión nueva y corta por request" es
simple, correcto, y no cuesta nada de performance real acá (abrir una
conexión sqlite es barato).

### El JOIN de "ya leíste"

```sql
SELECT c.id_comic, c.titulo, c.serie, c.numero, c.img_src, i.rating, i.fecha
FROM interacciones i
JOIN comics c ON c.id_comic = i.id_comic
WHERE i.id_usuario = ?
ORDER BY i.fecha IS NULL, i.fecha DESC
```

(`repo.py:86-97`). `interacciones` tiene el rating y la fecha pero NO el
título/tapa del comic -- esos viven en `comics`. El `JOIN` junta ambas tablas
por `id_comic` para poder mostrar "Batman #1, tu rating: 9" en una sola fila,
en vez de hacer dos consultas separadas y unirlas a mano en Python.
`i.fecha IS NULL, i.fecha DESC` es un truco chico: en SQLite, `x IS NULL`
vale `0` (falso) o `1` (verdadero), así que ordenar por esa expresión primero
manda todas las fechas no-nulas antes que las nulas (0 < 1), y recién ahí
aplica el `DESC` por fecha real -- sin este truco, `ORDER BY fecha DESC` a
secas pondría los `NULL` primero en SQLite (se tratan como el valor "más
chico" posible).

### Paginación y orden sin ensuciar el SQL con input del usuario

Con miles de comics leídos por usuario (el caso real: 2999 para
`Psycamorean`), mostrar "Tu comiteca" entera en una sola página sería
lentísimo de renderizar y de navegar. `repo.comics_leidos` acepta `pagina` y
`por_pagina` (`repo.py`, usa `LIMIT ? OFFSET ?` -- `OFFSET` salta las
primeras `(pagina-1) * por_pagina` filas) y `dashboard.index`
(`dashboard.py`) calcula `total_paginas` con `contar_comics_leidos` +
`math.ceil`, y clampea la página pedida al rango válido (`min(max(pagina, 1),
total_paginas)`) -- así un link a `?pagina=9999` o `?pagina=0` no rompe nada,
simplemente cae en la última o la primera página real.

Para el criterio de orden (`?orden=fecha|rating`) hay un problema de
seguridad específico de SQL: los placeholders `?` de `sqlite3` sirven para
VALORES (`WHERE id = ?`), no para nombres de columna o cláusulas `ORDER BY`
-- no hay forma de parametrizar eso de forma segura. La solución es un
diccionario blanco (`repo.ORDENES`, mapea una clave corta a la cláusula SQL
real) y nunca dejar que el string que llega en la URL toque la consulta
directamente: `ORDENES.get(orden, ORDENES[ORDEN_DEFAULT])` -- si alguien
manda `?orden=algo; DROP TABLE usuarios--`, esa clave simplemente no está en
el diccionario, y se usa el default sin que ese string llegue nunca a tocar
SQL.

`editorial` en cambio NO es un criterio de orden, es un FILTRO
(`?editorial=Marvel`, un `<select>` en el template poblado con
`repo.editoriales_leidas` -- solo las editoriales que ESE usuario tiene
leídas, no una lista fija). Son dos ejes independientes a propósito: se
puede filtrar por editorial y a la vez elegir si esas filas se ven por fecha
o por rating. El mismo criterio de "nunca interpolar el valor crudo" aplica
acá por un motivo distinto: `editorial` si se interpola como VALOR de un
`WHERE c.editorial = ?` (no como nombre de columna), así que sí puede ir en
un placeholder normal -- pero igual se valida contra
`editoriales_leidas(conn, id_usuario)` antes de usarlo
(`dashboard.py::index`, `if editorial not in editoriales: editorial = None`)
para que un valor que no es una editorial real de ese usuario no rompa
silenciosamente el filtro ni la paginación (`contar_comics_leidos` y
`comics_leidos` tienen que estar de acuerdo en qué filtro aplican, o la
cuenta de páginas queda inconsistente con lo que se muestra).

### `pila_por_leer`: una tabla nueva, separada de `interacciones` a propósito

Cuando el usuario guarda una recomendación "para después" (botón `+ Pila`),
esa acción se guarda en una tabla NUEVA, `pila_por_leer`
(`comics_recsys/db.py`), no como una fila más de `interacciones` con rating
vacío. La razón es que `interacciones` alimenta directamente el
recomendador (`comics_calificados_por_usuario`, `fit_popularity`) -- si
"quiero leer esto" se guardara ahí como una interacción sin rating,
contaminaría esas estadísticas (o, peor, un rating `NULL` propagándose en
una suma ponderada como `NaN` rompe silenciosamente el ranking completo de
ese usuario, ver la sección de abajo). Separar la tabla mantiene
`interacciones` con un significado único y limpio: "esto es una lectura real
con una opinión real", scrapeada o puesta desde el sitio, da lo mismo.

Agregar una tabla nueva a una `comics.db` ya poblada, a diferencia de
agregar una COLUMNA (sección 1), no necesita un script de migración aparte:
`CREATE TABLE IF NOT EXISTS pila_por_leer (...)` dentro de
`comics_recsys.db.crear_esquema` simplemente crea la tabla si todavía no
existe, sin tocar las tablas que ya estaban -- correr `crear_esquema` de
nuevo contra la BD real (aunque ya tenga años de datos scrapeados) es
seguro e idempotente.

## 4. Motor de recomendación: matrices sparse, similitud coseno, y por qué precomputar

### El problema de memoria que fuerza todo el diseño

El dataset tiene ~14.000 comics. Si quisiéramos una matriz de similitud
"cada comic contra todos los demás" completa (densa), serían 14.000² ≈ 196
millones de números -- a 8 bytes cada uno (float64), ~1.5 GB solo para esa
matriz, en un plan gratuito con memoria muy limitada. Pero la matriz
usuario×comic real es **sparse** (dispersa): con 114.386 interacciones sobre
1.696 usuarios × 14.051 comics posibles, más del 99% de las celdas están
vacías (un usuario no calificó casi ningún comic). `scipy.sparse` guarda solo
los valores no-cero, así que las mismas operaciones matemáticas (sumas,
productos de matrices) cuestan proporcional a cuántos datos hay realmente,
no al tamaño total de la grilla.

### Qué es similitud coseno entre columnas

Pensá cada comic como un vector: una columna en la matriz usuario×comic,
donde cada posición es el rating que un usuario le dio (o 0/vacío si no lo
leyó). Dos comics son "similares" según este método si los usuarios que
calificaron a uno tienden a calificar parecido al otro -- geométricamente,
si sus vectores apuntan en direcciones parecidas. La similitud coseno mide
exactamente eso: el coseno del ángulo entre dos vectores, 1 si apuntan
igual, 0 si son ortogonales (nada en común), sin importar la magnitud (un
comic con 3000 reviews y uno con 5 pueden ser "igual de similares" a un
tercero si el patrón de gustos relativo es el mismo).

Matemáticamente, si normalizás cada columna a longitud 1 (dividir cada
valor por la norma L2 de su columna, `precomputar_item_item.py::normalizar_columnas`),
el producto de matrices `M_norm.T @ M_norm` te da DIRECTAMENTE todas las
similitudes coseno par a par, en un solo paso -- es la base de por qué el
precómputo usa multiplicación de matrices en vez de comparar comics de a
pares en un loop.

### Por qué offline y no en cada request

Calcular esa multiplicación de matrices (aunque sea sparse) para 14.000
comics no es instantáneo, y PythonAnywhere free cobra en **cuota de
CPU-segundos por día** -- recalcular esto en cada visita al dashboard
agotaría esa cuota rapidísimo, además de hacer la página lenta para el
usuario. La solución: `comics/scripts/precomputar_item_item.py` corre UNA
vez (localmente, o manualmente cuando el dataset cambie), calcula la
similitud completa, la recorta a los top-50 vecinos de cada comic (no hace
falta guardar 14.000 similitudes por comic si casi todas son irrelevantes o
cero), y guarda el resultado en un archivo (`.npz`, formato comprimido de
numpy). El webapp (`Recomendador.__init__`, `comics/webapp/recsys_runtime.py:26-33`)
carga ese archivo **una sola vez cuando arranca el proceso**, no en cada
request -- de ahí en más, recomendar para un usuario es solo indexar arrays
ya en memoria (rápido, sin CPU pesado).

En la corrida real sobre el dataset completo, precomputar tardó 3.4
segundos para 10.555 comics con al menos una review (`522.726` pares
vecino-vecino guardados) -- trivial una vez, prohibitivo si se repitiera en
cada uno de los miles de requests que recibiría el sitio.

### Cómo se combina item-item con el fallback de popularidad

`Recomendador.recomendar` (`recsys_runtime.py:35-64`):

1. Arma un "perfil" del usuario: qué comics leyó y con qué rating
   (o el `perfil_coldstart` que le pasen, ver más abajo).
2. Si el perfil está vacío (usuario nuevo, sin leer nada todavía) -> devuelve
   directo el ranking de popularidad (`fit_popularity`, reusado tal cual de
   `comics_recsys/models/popularity.py`, ya evaluado con NDCG@20 en
   `comics/experiments/log.csv`).
3. Si no está vacío -> para cada comic del perfil, suma (ponderado por el
   rating que le dio) la similitud de sus vecinos top-50 precomputados
   (`_agregar_vecinos`, `recsys_runtime.py:66-78`), arma un ranking de
   candidatos, y excluye lo que el usuario ya leyó.
4. Si esos candidatos no alcanzan para completar `k` (puede pasar: un comic
   con pocos vecinos, o un perfil chico), **rellena** con el ranking de
   popularidad, saltando lo que ya se recomendó o ya se leyó
   (`_completar_con_popularidad`, `recsys_runtime.py:80-87`).

Se verificó en vivo contra el dataset real (2.999 reviews de un usuario
reclamado, `Psycamorean`) que las recomendaciones personalizadas son
notoriamente distintas del fallback de popularidad puro que ve un usuario
nuevo -- no se solapa ni un título entre los primeros 5 de cada lista.

### Tope de issues por serie (`MAX_POR_SERIE = 2`)

Los issues de una misma serie son casi siempre los vecinos más parecidos
entre sí, porque los leen las mismas personas. Sin un tope, puntuar
*Ultimate Spider-Man #21* devolvía 10 de 10 recomendaciones de Ultimate
Spider-Man (#22, #23, #20, #19...). Con tope 2 quedan #22 y #23 (el número
siguiente de algo que te gustó sigue siendo una buena recomendación) y el
resto se abre a Ultimates, Ultimate Endgame, Ultimate Wolverine, Absolute
Batman. `_limitar_por_serie` recorre la lista ya ordenada por score y
solo saltea lo que excede el tope, no reordena nada. Aplica tanto a los
vecinos item-item como al relleno por popularidad.

### La interfaz pensada para el onboarding (ver sección 7)

El parámetro `perfil_coldstart: dict[str, float] | None` en `recomendar()`
existe por una razón concreta: cuando se construya la novela gráfica
point-and-click para usuarios nuevos, ese juego va a terminar armando "a qué
se parece el gusto de esta persona" sin que haya leído nada todavía. En vez
de forzar a esa fase 2 a escribir filas falsas en `interacciones` para que el
recomendador las "vea", puede llamar directamente a
`recomendar(id_usuario, k, perfil_coldstart={"algun-comic": 0.8, ...})` y el
resto del pipeline (agregación de vecinos, fallback a popularidad) funciona
igual, sin cambiar una línea de este archivo. Esto se prueba explícitamente
en `comics/tests/test_recsys_runtime.py::test_perfil_coldstart_no_toca_historial_real`.

### El otro lado del problema: reflejar un rating nuevo SIN esperar un restart

`perfil_coldstart` resuelve "recomendar con un perfil que no está en la
BD". Pero hay un problema simétrico: cuando el usuario marca un comic
recomendado como leído y lo puntúa (`dashboard.marcar_leido`,
`repo.marcar_como_leido` SÍ escribe esa fila real en `interacciones`), el
`Recomendador` no se entera solo -- `self._perfiles` y `self._ratings`
son estructuras que se calcularon UNA vez, al arrancar el proceso
(`Recomendador.__init__`, sección "Por qué offline y no en cada request"
de más arriba), no una consulta en vivo a la base. Sin ningún mecanismo
extra, esa foto quedaría vieja hasta el próximo restart del servidor -- el
usuario puntuaría un comic y seguiría viéndolo recomendado, lo cual además
sería una mala señal para el resto de la sesión.

La solución, `Recomendador.registrar_interaccion(id_usuario, id_comic,
rating)` (`recsys_runtime.py`), actualiza esas DOS estructuras en memoria
(agrega el id al `set` de `self._perfiles[id_usuario]`, agrega la entrada a
`self._ratings`) sin tocar el archivo `.npz` de similitud ni recalcular el
ranking de popularidad completo -- `dashboard.marcar_leido` la llama
inmediatamente después de guardar en la base
(`current_app.extensions["recomendador"].registrar_interaccion(...)`).
Por eso `self._ratings` es un `dict` común y no una `pandas.Series` (que sí
se usó en una versión anterior de este archivo): un `dict` se puede
actualizar en O(1) con una asignación común (`self._ratings[clave] =
valor`); mutar una `Series` de pandas punto a punto es más costoso y menos
directo. Este comportamiento se prueba en
`comics/tests/test_recsys_runtime.py::test_registrar_interaccion_se_refleja_sin_reconsultar_la_bd`
y de punta a punta (HTTP real) en
`comics/tests/test_webapp_integration.py::test_marcar_leido_lo_agrega_a_la_comiteca_y_lo_saca_de_la_pila`
-- se verificó también a mano contra el sitio corriendo con el dataset real:
marcar un comic recomendado como leído lo saca de "Te recomendamos" en el
siguiente refresh, sin reiniciar el proceso.

Lo que **no** se actualiza en caliente, a propósito, es
`self._ranking_popularidad` (el fallback) -- recalcular `fit_popularity`
sobre miles de usuarios por cada rating individual no vale el costo de CPU,
y una fila nueva cambia un ranking agregado tan poco que no se nota.
Coherente con la misma lógica de costo/beneficio que ya justifica precomputar
la similitud item-item offline.

## 5. Tests: por qué sqlite en memoria, y qué previene cada uno

Todos los tests nuevos siguen el patrón que ya usaba
`comics/tests/test_db.py`: una conexión sqlite `:memory:` (vive solo en RAM,
se descarta sola al cerrar la conexión) con el schema real
(`comics_recsys.db.crear_esquema`), en vez de un archivo en disco o mocks.
Ventaja concreta: es rápido (no toca disco), y prueba el SQL *real* que corre
en producción -- un mock de la capa de datos podría "pasar" aunque la query
esté mal escrita.

- `test_migracion_password_hash.py`: simula una `comics.db` **anterior** al
  cambio de schema (sin la columna) y prueba que `migrar()` la agrega sin
  perder datos existentes, y que correrla dos veces no rompe nada
  (idempotencia) -- esto previene un error real: sin el chequeo de
  `PRAGMA table_info` antes del `ALTER TABLE`, correr el script de más
  (por ejemplo, dos deploys seguidos) haría que SQLite tire un error de
  "columna duplicada" y aborte.
- `test_webapp_repo.py`: cubre cada función de `repo.py` por separado,
  incluyendo el caso de la carrera de doble-reclamo (sección 2) y que
  `comics_por_id` preserva el orden de entrada (si no lo hiciera, las
  recomendaciones se mostrarían en un orden random, no por score).
- `test_recsys_runtime.py`: arma una similitud item-item sintética a mano
  (4 comics con vecinos conocidos) para poder afirmar EXACTAMENTE qué
  debería recomendar el motor, sin depender del dataset real (que cambia).
  Prueba las 4 ramas de `recomendar()`: con historial, completando con
  popularidad, con `perfil_coldstart`, y con perfil vacío.
- `test_webapp_integration.py`: el único test que levanta la app Flask
  completa (`app.test_client()`) y prueba el cableado end-to-end -- registro
  → login → dashboard con 200 y contenido esperado; login con password mal
  no entra; pedir `/` sin sesión redirige a `/login`. Es el test que hubiera
  detectado, por ejemplo, un blueprint mal registrado o un `url_for` roto,
  que los tests unitarios de `repo.py` no pueden ver porque no arman la app.

## 6. Deployment a PythonAnywhere (plan free)

Sitio: **https://leonardoiula.pythonanywhere.com** (cuenta `leonardoiula`).

Qué NO viaja por git (gitignored, ver `comics/CLAUDE.md`): la BD y el cache
`item_item_top50.npz`. El `.npz` se precomputa localmente y se sube ya
calculado -- correrlo en el servidor gastaría la cuota de CPU del plan free
para algo que no necesita repetirse en cada deploy.

**BD de producción (`scripts/armar_db_produccion.py`):** no se sube la
`comics.db` local tal cual (~150 MB, excede el límite por archivo del upload
web). El script arma una copia en `comics/data/deploy/comics.db` (~40 MB) sin
`critic_reviews` ni `interacciones.texto` (el webapp no usa ninguno de los
dos) y sin nada generado por pruebas locales del webapp (cuentas `local-*`,
passwords de usuarios reclamados, pila, ratings rápidos) -- ver su docstring.

**Dependencias (`webapp/requirements.txt`):** solo las del webapp. NO usar
`uv export` del `pyproject.toml` raíz: arrastra lightgbm, implicit, pyarrow,
matplotlib, scikit-learn (modelos de libros), que no entran en el disco.

**Tapas (decisión provisoria, 2026-09-26):** el webapp NO sirve las tapas
desde `comics/static/covers/`, las muestra directo desde la URL original
(`comics.img_src`, hotlink a `images.comicbookroundup.com`). Motivo: el disco
del plan free no alcanza (las tapas completas rondarían ~480 MB). Se verificó
que el servidor de imágenes de ellos responde normalmente a pedidos con un
`Referer` de otro dominio. Si un comic no tiene `img_src`, o la URL externa
falla al cargar (`onerror` en el `<img>`), se muestra
`comics/static/sintapa.jpg` (recortado a la proporción 200x308 de las tapas
scrapeadas). Queda pendiente una alternativa mejor (no depender de un
servidor ajeno ni gastarle ancho de banda). `descargar_tapas.py` y
`comics_recsys/covers.py` se conservan para ese momento.

Al preparar esto apareció un bug del parser: solo reconocía tapas `.webp`, y
los issues anteriores a ~2022 las tienen en `.jpg` -- ~17 mil comics habían
quedado con `img_src` NULL. Corregido en `parse.py` (test de regresión con
fixture `house_of_x_2.html`) y rellenado con `scripts/rellenar_img_src.py`,
que re-parsea el HTML ya cacheado en `data/cache/` sin pedirle nada al sitio.

Pasos, en orden:

1. Local: `precomputar_item_item.py` y `armar_db_produccion.py`.
2. Server (consola Bash de PythonAnywhere):
   `git clone https://github.com/leonardoiula/recsys-libros.git` en el home.
3. Subir por la pestaña "Files" `comics/data/deploy/comics.db` a
   `~/recsys-libros/comics/data/raw/comics.db` y el `.npz` a
   `~/recsys-libros/comics/data/cache/` -- son las rutas default de
   `config.py`, así no hace falta setear `COMICS_DB_PATH` ni
   `COMICS_RECS_CACHE_PATH`.
4. Virtualenv: `python3.11 -m venv ~/.venvs/comics` e instalar
   `comics/webapp/requirements.txt`.
5. Pestaña "Web": crear la web app con "Manual configuration" (Python 3.11),
   apuntar "Virtualenv" a `/home/leonardoiula/.venvs/comics`, y reemplazar el
   WSGI configuration file por el contenido de `webapp/pythonanywhere_wsgi.py`
   con una `COMICS_SECRET_KEY` random. El secreto va en ese archivo (vive en
   `/var/www/`, fuera del clon) porque el repo es público.
6. "Static files": URL `/static/` -> `/home/leonardoiula/recsys-libros/comics/static`
   (así `sintapa.jpg` lo sirve PythonAnywhere directo, sin pasar por Flask).
7. Reload.

Actualizar el sitio después: `git pull` en el server + Reload; si cambió el
dataset, repetir el paso 1 y re-subir BD y `.npz`. Ojo: re-subir la BD pisa
las cuentas y ratings creados en producción.

Por qué las rutas de `config.py` se resuelven con `Path(__file__).resolve()`
y no con el directorio de trabajo actual (`Path(".")` o similar): el proceso
WSGI de PythonAnywhere no necesariamente arranca con el cwd en la raíz del
proyecto, así que cualquier ruta relativa al cwd podría apuntar a cualquier
lado según cómo lo arranque su infraestructura -- resolver contra
`__file__` es robusto sin importar desde dónde se invoque el proceso.

## 7. Onboarding: la ficción interactiva para usuarios nuevos

Una cuenta nueva no tiene historial, así que el recomendador solo puede
mostrarle lo más popular. El onboarding (`webapp/onboarding.py` +
`webapp/onboarding_datos.py`, plantilla `templates/onboarding.html`) junta
señales de gusto disfrazadas de ficción: el usuario "despierta" en una
rebelión contra la Entidad y tiene que reconstruir su memoria, el Códice.
El guion es `comics/1. El Despertar y la Bienvenida.txt` y la estética visual
sale de `comics/estetica.jpg` (el fondo es `static/onboarding/escena.jpg`, la
misma imagen comprimida).

### La bienvenida: el juego es opcional y se dice para qué sirve

Después de crear la cuenta (`auth.registro`) no se entra directo a la
ficción: primero aparece `/onboarding/bienvenida`, con la misma viñeta de
comic que el login y el registro. Ofrece dos caminos explícitos: "Jugar y
mejorar tus recomendaciones" (entra al onboarding) o "Ir directo al sitio"
(dashboard con lo más popular, que después sigue ofreciendo el juego
mientras no haya historial). Sin esta pantalla, un usuario recién
registrado caía en "Tranquilo, respira..." sin saber que era un juego ni
que podía salir.

### Las fases y qué dato saca cada una

| Fase | Pantalla | Qué se guarda |
|---|---|---|
| 0 | El despertar | nada (registra que empezó) |
| 1 | Universos: 12 editoriales, al menos 3 evaluadas | `onboarding_editoriales` (encanta / gusta / curiosidad / no me atrae) |
| 2 | Memoria reciente: lo de más impacto del último año, de las editoriales elegidas | "lo leí" + nota -> `interacciones`; "me da curiosidad" -> `onboarding_curiosidad` |
| 3 | Ecos: de hace 1 a 5 años | ídem |
| 4 | "Fundacionales": lo mejor puntuado (score bayesiano) de antes de hace 5 años | ídem |
| 5 | Anomalías: al azar, de editoriales NO elegidas | ídem |
| 6 | Sincronización | marca `onboarding_progreso.completado_en` |

La Fase 4 del guion hablaba de clásicos de la Golden Era, pero el dataset
arranca en 2018 y un clásico que no está en la matriz no tiene vecinos: no
aportaría a ninguna recomendación. Por eso se reinterpretó como "lo
fundacional de lo que sí tenemos" (House of X, Immortal Hulk, Doomsday
Clock...).

### Por qué "lo leí" va a `interacciones` y "me da curiosidad" no

Un "lo leí, 8/10" es un dato de lectura real, igual que el botón "Leído" del
dashboard, y reusa `repo.marcar_como_leido`. Sirve también para entrenar
offline. Una curiosidad o una preferencia por editorial NO son lecturas: si
fueran a `interacciones` ensuciarían el entrenamiento (sería como inventar
reviews). Por eso viven en tablas propias y entran al recomendador solo en
runtime, por la interfaz que se dejó preparada en la sección 4
(`perfil_coldstart`):

- `perfil_para_recomendar` arma `{id_comic: peso}` con los ratings reales +
  cada curiosidad con `PESO_CURIOSIDAD = 7.0` (misma escala 0-10). Es un valor
  de criterio: no hay cómo medir el NDCG del onboarding offline, porque no
  existen usuarios históricos que lo hayan hecho.
- Las editoriales solo reordenan el **relleno** por popularidad (primero las
  preferidas, al final las "no me atrae"). No tocan el score item-item:
  "me gusta Marvel" es una señal mucho más gruesa que un comic concreto.

### Decisiones de selección que no son obvias

- **Fecha de un comic = su primera review**: `anio_edicion` falta en ~36%, y
  como el scraping recorre semanas de lanzamiento, la primera review cae en
  la práctica en la semana en que salió. El "hoy" es la review más reciente
  del dataset, no la fecha del sistema: si el scraping queda viejo, "el
  último año" sigue teniendo datos.
- **Un issue por serie + editoriales intercaladas**: sin esto el carrusel del
  último año eran 4 números de Absolute Batman seguidos y era 100% DC.
- **Solo comics con tapa y con al menos una review**: lo que se reconoce es la
  tapa, y un comic sin reviews no tiene vecinos en la similitud.
- **Anomalías con semilla por usuario**: recargar la página no cambia la
  selección.

### Detalles de implementación

- Las tablas se crean al arrancar la app (`crear_tablas`, idempotente): un
  `git pull` + Reload en PythonAnywhere alcanza, sin migración a mano.
- `CatalogoOnboarding` calcula las estadísticas por comic una sola vez al
  arrancar, igual que el `Recomendador`: nada pesado por request.
- El progreso nunca retrocede (`MAX(fase, ...)`), así que el botón "atrás"
  del navegador no pierde avance. Quien se desconecta a mitad de camino ve
  en el dashboard una invitación a retomarlo mientras no tenga historial.
- Los ids de comic del form se validan contra el catálogo: un id inventado
  nunca llega a la BD.
- Cada carrusel (fases 2-5) exige al menos `MIN_INTERACCIONES_POR_FASE = 3`
  respuestas (nota o curiosidad), o todas si mostró menos de 3. Sin esto se
  podía atravesar el onboarding entero apretando "SINCRONIZAR" sin dar
  ninguna señal. Si faltan, la pantalla se vuelve a mostrar (render, no
  redirect) con lo ya cargado, para no hacer perder las notas escritas.
- El puntaje de "lo leí" son 5 estrellas clickeables (radios estilizados,
  sin JS obligatorio), cada una vale 2 puntos de la escala 0-10 de la BD. Se
  eligieron sobre un input numérico (flechitas de a 0,5: lento), un slider
  (siempre tiene valor, no distingue "no lo leí" de "le puse 5") o un
  desplegable (abrir-buscar-elegir por comic). Un JS mínimo permite
  desmarcar con un segundo click; la card respondida se "enciende". El botón
  "Leído" del dashboard usa las mismas estrellas (ahí, estrellas + click en
  "Leído": dos pasos a propósito, para que un click errado no marque un
  comic como leído).
- No se puede saltar hacia adelante escribiendo la URL (`/onboarding/6`):
  solo se entra a fases ya alcanzadas.
- Tests: `comics/tests/test_onboarding.py`.

## 8. Pantallas de entrada: login y registro como viñeta de comic

Login, registro y reclamar identidad extienden `templates/vineta_base.html`
(no `base.html`). Referencia visual: `comics/fondo.jpg`, un callejón
dibujado con una viñeta enmarcada al centro. El contenido va DENTRO de esa
viñeta:

- `.escena` se comporta como un `background-size: cover` pero es un
  elemento propio con la proporción de la imagen (1408x768). Así la viñeta
  se posiciona en % de la imagen (marco medido por píxeles: x 351-1073, y
  211-630) y queda calzada sobre el dibujo con cualquier tamaño de ventana.
- Los tamaños de letra dentro de la viñeta usan unidades de container query
  (`cqh`/`cqw`): escalan con la viñeta, no con la ventana.
- En pantallas angostas, verticales o muy bajas el dibujo entero no entra
  con la viñeta a un tamaño usable. El callejón queda de fondo oscurecido y
  la viñeta pasa a ser un recuadro con borde de tinta que muestra el
  recorte del marco (`static/login/vineta.jpg`, sacado de `fondo.jpg`).
- Cada página llena tres bloques: `lugar` (caja ámbar de arriba), `cuerpo`
  y `pie` (caja de abajo). Los mensajes flash salen como caja roja.
