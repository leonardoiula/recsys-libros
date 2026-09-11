"""
Precomputa similitud coseno item-item (comic-comic) sobre co-reviews y guarda
los top-K vecinos de cada comic en comics/data/cache/item_item_top50.npz.

Por qué offline y no en cada request: PythonAnywhere plan free tiene cuota de
CPU-segundos por día, y una matriz de similitud NxN densa con ~14k comics no
entra cómoda en memoria (14000^2 floats = ~1.5GB). Se calcula acá UNA vez,
sparse, y el webapp solo carga el resultado ya recortado a top-K al arrancar
(ver comics/webapp/recsys_runtime.py). Ver comics/docs/webapp/guia.md,
sección "Motor de recomendación".

Uso: uv run python comics/scripts/precomputar_item_item.py [--k-vecinos 50]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comics_recsys.data import load_interacciones

COMICS_DIR = Path(__file__).resolve().parents[1]
CACHE_PATH = COMICS_DIR / "data" / "cache" / "item_item_top50.npz"


def construir_matriz_usuario_comic(interacciones: pd.DataFrame) -> tuple[sp.csr_matrix, np.ndarray]:
    """Devuelve (matriz sparse usuario x comic con rating como valor, array de
    id_comic alineado a las columnas -- el índice de columna ES la posición
    en ese array)."""
    usuarios_cat = pd.Categorical(interacciones["id_usuario"])
    comics_cat = pd.Categorical(interacciones["id_comic"])

    matriz = sp.csr_matrix(
        (interacciones["rating"].to_numpy(dtype=np.float64), (usuarios_cat.codes, comics_cat.codes)),
        shape=(len(usuarios_cat.categories), len(comics_cat.categories)),
    )
    return matriz, np.array(comics_cat.categories, dtype=str)


def normalizar_columnas(matriz: sp.csr_matrix) -> sp.csr_matrix:
    """L2-normaliza cada columna (columna = comic) para que M.T @ M dé
    directamente similitud coseno."""
    normas = np.sqrt(matriz.power(2).sum(axis=0)).A1
    escala = np.divide(1.0, normas, out=np.zeros_like(normas), where=normas > 0)
    return matriz @ sp.diags(escala)


def top_k_vecinos(similitud: sp.csr_matrix, k_vecinos: int) -> sp.csr_matrix:
    """Recorta cada fila de `similitud` (ya sin diagonal) a sus k_vecinos más
    altos. `similitud` es simétrica -- cada comic queda con sus k vecinos
    propios, no necesariamente recíproco."""
    similitud = similitud.tocsr(copy=True)
    similitud.setdiag(0)
    similitud.eliminate_zeros()

    filas, columnas, datos = [], [], []
    for i in range(similitud.shape[0]):
        inicio, fin = similitud.indptr[i], similitud.indptr[i + 1]
        idx_col = similitud.indices[inicio:fin]
        valores = similitud.data[inicio:fin]
        if len(valores) > k_vecinos:
            top = np.argpartition(valores, -k_vecinos)[-k_vecinos:]
            idx_col, valores = idx_col[top], valores[top]
        filas.extend([i] * len(idx_col))
        columnas.extend(idx_col)
        datos.extend(valores)

    return sp.csr_matrix((datos, (filas, columnas)), shape=similitud.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-vecinos", type=int, default=50)
    args = parser.parse_args()

    t0 = time.time()
    interacciones = load_interacciones()
    matriz, comic_ids = construir_matriz_usuario_comic(interacciones)
    matriz_norm = normalizar_columnas(matriz)

    similitud = (matriz_norm.T @ matriz_norm).tocsr()
    similitud_top_k = top_k_vecinos(similitud, args.k_vecinos)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        data=similitud_top_k.data,
        indices=similitud_top_k.indices,
        indptr=similitud_top_k.indptr,
        shape=np.array(similitud_top_k.shape),
        comic_ids=comic_ids,
    )

    print(
        f"{len(comic_ids)} comics, {similitud_top_k.nnz} pares vecino-vecino "
        f"(top-{args.k_vecinos}), {time.time() - t0:.1f}s"
    )
    print(f"Guardado en {CACHE_PATH}")


if __name__ == "__main__":
    main()
