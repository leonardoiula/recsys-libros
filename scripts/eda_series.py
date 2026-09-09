"""EDA: ¿se puede derivar metadata de serie/saga desde los títulos, y
alcanza para mover la aguja?

Uso: uv run python scripts/eda_series.py

Resultado (2026-09-09): NO alcanza por COBERTURA (no por mecanismo).
- Patrón limpio "<título>. <serie> <N>": 2,0% de los libros con
  interacción (763 libros / 196 series de >=2 títulos).
- De los 8904 test targets: 3,6% en una serie detectable; 1,7% con el
  usuario ya leyendo esa serie; **0,8% donde el target es exactamente el
  siguiente (max+1)**. Techo de ganancia ~+0,002 local / ~+0,0005 Kaggle,
  dentro del ruido. Ver `experiments/estado_del_arte.md`.
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.data import split_train_val

DB = Path(__file__).resolve().parents[1] / "data" / "raw" / "data.db"
_PAT = re.compile(r"^(?P<main>.+?)\.\s*(?P<serie>.+?)\s+(?P<num>\d{1,3})\s*$")


def parse_serie(titulo: str) -> tuple[str, int] | None:
    m = _PAT.match(titulo or "")
    if not m:
        return None
    serie = m.group("serie").strip()
    if len(serie) < 4 or serie.isdigit():
        return None
    return serie.upper(), int(m.group("num"))


def main() -> None:
    con = sqlite3.connect(DB)
    libros = pd.read_sql("select id_libro, titulo from libros", con)
    inter = pd.read_sql("select id_libro, id_lector, fecha from interacciones", con)

    lc = libros[libros.id_libro.isin(set(inter.id_libro))].copy()
    lc["serie"] = lc.titulo.fillna("").str.strip().map(parse_serie)
    con_serie = lc.dropna(subset=["serie"])
    print(f"libros con interacción: {len(lc)}")
    print(f"con serie detectable (patrón '<título>. <serie> <N>'): {len(con_serie)}  ({len(con_serie)/len(lc):.1%})")
    n_series = con_serie.serie.map(lambda s: s[0]).nunique()
    print(f"series distintas: {n_series}")

    serie_de = {r.id_libro: r.serie for r in con_serie.itertuples()}
    tcf, test = split_train_val(inter, n_val=1, seed=42)
    hist = tcf.groupby("id_lector")["id_libro"].apply(list).to_dict()

    n = len(test)
    en_serie = leyendo = siguiente = 0
    for r in test.itertuples():
        if r.id_libro not in serie_de:
            continue
        en_serie += 1
        s_tgt, num_tgt = serie_de[r.id_libro]
        nums = [serie_de[b][1] for b in hist.get(r.id_lector, []) if b in serie_de and serie_de[b][0] == s_tgt]
        if nums:
            leyendo += 1
            if num_tgt == max(nums) + 1:
                siguiente += 1

    print(f"\ntest targets: {n}")
    print(f"  el objetivo está en una serie detectable:            {en_serie:5}  ({en_serie/n:.1%})")
    print(f"  ...y el usuario ya leyó un libro de esa serie:       {leyendo:5}  ({leyendo/n:.1%})")
    print(f"  ...y el objetivo es exactamente el siguiente (max+1):{siguiente:5}  ({siguiente/n:.1%})")


if __name__ == "__main__":
    main()
