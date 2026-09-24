"""Test PAREADO por usuario bajo el protocolo de CORTE TEMPORAL GLOBAL
(`scripts/evaluate_global_cutoff.py`), no el split por usuario de
`comparar_generadores_pareado.py`/`comparar_features_pareado.py`.

Uso: uv run python scripts/comparar_global_pareado.py

Por qué existe: el pareado de siempre (`comparar_generadores_pareado.py`) corrigió sus
dos bugs de config (`REFIT_PARA_TEST`/`nfa=500`) pero sigue siendo un split leave-one-out
**por usuario** -- todavía tiene la fuga temporal poblacional descripta en
`experiments/estado_del_arte.md` ("Validación con corte temporal global"): ALS/
popularidad/co-lectura ven señal de "futuro" respecto del punto de validación de un
usuario típico. Este script hace el mismo tipo de comparación A/B con el mismo aparato
estadístico (diferencia pareada, bootstrap, sigma), pero sobre
`evaluate_global_cutoff.construir_y_evaluar` -- las dos configs comparten la MISMA
ventana rodante de cortes de calendario, así que la comparación es tan limpia respecto
de fuga temporal como puede serlo con los datos de este proyecto.

Costo: cada `construir_y_evaluar()` completo tarda ~7-8 min (5 ventanas + refit) --
una comparación A/B son ~15 min, sin cache a disco (a diferencia de
`preparar_pipeline_cacheado`, no vale la pena cachear algo que se corre una vez por
idea). Editar `FUENTES_A`/`FUENTES_B` o `FEATURES_A`/`FEATURES_B` de abajo para
comparar otro par.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_global_cutoff import K, construir_y_evaluar
from recsys.models import ranker as R

N_BOOTSTRAP = 2000

# 2026-09-24: se usó para re-testear LMF (reimplementado temporalmente, wiring ya
# revertido de ranker.py) bajo este protocolo -- -1,09 sigma, P(mejora)=0,13, 36
# mejoran vs 52 empeoran -- NEGATIVO, a diferencia del +1,56 sigma "casi positivo"
# que daba el pareado por-usuario incluso ya corregido (refit+nfa=500). Confirma que
# el corte global (sin fuga temporal) es el instrumento que de verdad distingue este
# tipo de señal. Ver experiments/estado_del_arte.md y experiments/log.csv.
#
# 2026-09-24, segundo uso: dado que LMF (arriba) mostró que el pareado por-usuario
# sobreestima señal que explota patrones poblacionales, se re-examinan bajo el corte
# global las features adoptadas con señal LÍMITE en el protocolo viejo (nunca
# "positivo limpio en los 3 seeds", confirmadas por una sola submission de Kaggle con
# delta absoluto chico, varias del orden del SE~0.0065 de una submission) -- mismo
# perfil de riesgo que LMF. FEATURES_EXCLUIR_LIMITE junta las 7 de esas rondas:
# género macro + tamaño de editorial (2026-08-30, CV "no confirmado" por la regla
# estricta, confirmado igual en Kaggle +3.7-4.5%), señales cruzadas lector↔libro
# (2026-08-31, "casi positivo en los 3 seeds", Kaggle +0.5% solamente) y
# score_difusion_candidato (2026-09-10, pareado BORDERLINE 1.17σ, Kaggle +0.00071 ~
# del tamaño del SE de una submission). Resultado (fila "ranker" 0.045987,
# 2026-09-24, ver estado_del_arte.md): sacarlas EMPEORA bajo el corte global
# (+1.58 sigma a favor de conservarlas) -- quedan en producción, no se tocan.
FEATURES_EXCLUIR_LIMITE = [
    "popularidad_genero_macro_candidato",
    "frecuencia_genero_macro_usuario",
    "n_libros_editorial_catalogo",
    "popularidad_genero_lector_candidato",
    "frecuencia_genero_macro_por_genero_lector",
    "edad_lector_al_publicarse",
    "score_difusion_candidato",
]

# 2026-09-24, tercer uso: sanear los 2 hallazgos reales de la auditoría de catálogo
# (duplicados de libro + fragmentación de autor/editorial, ver "Sanear los tres
# hallazgos de la Parte I" en estado_del_arte.md) vs. la config de producción SIN
# CAMBIOS -- mismas 6 fuentes, mismas 40 features, mismos hiperparámetros. Aísla el
# efecto de limpiar el dato del de cualquier otro cambio (por eso FUENTES/FEATURES
# vuelven a None en vez de reusar la comparación de arriba). RESULTADO: NULO --
# NDCG 0.045987 (crudos) vs 0.045504 (saneados), n=811, +0.10 sigma, P(mejora)=0.546,
# CI [-0.0086,+0.0100] -- muy por debajo del umbral del proyecto. `sanear` queda
# disponible (opt-in, default False) pero sin efecto medible sobre esta arquitectura
# -- ver fila "ranker" 2026-09-24 en log.csv. Reseteado a False/False (sin
# experimento activo) tras esta corrida.
FUENTES_A = None
FUENTES_B = None
FEATURES_A = None  # None = R.FEATURES completo
FEATURES_B = None
SANEAR_A = False
SANEAR_B = False


def _reportar_pareado(valores_a: np.ndarray, valores_b: np.ndarray, nombre_a: str, nombre_b: str) -> None:
    diferencia = valores_a - valores_b
    n = len(diferencia)

    print(f"\nn usuarios de test: {n}")
    print(f"NDCG@{K} {nombre_a}: {valores_a.mean():.6f}")
    print(f"NDCG@{K} {nombre_b}: {valores_b.mean():.6f}")

    se_no_pareado = math.sqrt(valores_a.var(ddof=1) / n + valores_b.var(ddof=1) / n)
    sigma_no_pareado = diferencia.mean() / se_no_pareado if se_no_pareado else float("nan")
    print(f"  no pareado: dif medias {diferencia.mean():+.6f}   SE {se_no_pareado:.6f}   -> {sigma_no_pareado:.2f} sigma")

    se_pareado = diferencia.std(ddof=1) / math.sqrt(n)
    sigma_pareado = diferencia.mean() / se_pareado if se_pareado else float("nan")
    print(f"  PAREADO:    dif media {diferencia.mean():+.6f}   sd {diferencia.std(ddof=1):.6f}   "
          f"SE {se_pareado:.6f}   -> {sigma_pareado:.2f} sigma")
    print(f"  usuarios donde cambia el NDCG: {(diferencia != 0).mean():.4f}"
          f"  (mejora {(diferencia > 0).sum()}, empeora {(diferencia < 0).sum()})")

    rng = np.random.default_rng(0)
    bootstrap = np.array([rng.choice(diferencia, size=n, replace=True).mean() for _ in range(N_BOOTSTRAP)])
    print(f"  bootstrap 95% CI: [{np.percentile(bootstrap, 2.5):+.6f}, {np.percentile(bootstrap, 97.5):+.6f}]"
          f"   P(dif > 0) {(bootstrap > 0).mean():.4f}")
    if se_pareado:
        print(f"  ganancia de poder: SE no pareado / SE pareado = {se_no_pareado / se_pareado:.1f}x")


def main() -> None:
    nombre_a = "todas" if FUENTES_A is None else "+".join(sorted(FUENTES_A))
    nombre_b = "todas" if FUENTES_B is None else "+".join(sorted(FUENTES_B))
    print(
        f"=== corte global: fuentes_A=[{nombre_a}] sanear_A={SANEAR_A} vs "
        f"fuentes_B=[{nombre_b}] sanear_B={SANEAR_B} ==="
    )

    t0 = time.time()
    print("\n--- config A ---", flush=True)
    r_a = construir_y_evaluar(fuentes_activas=FUENTES_A, features=FEATURES_A, sanear=SANEAR_A)
    print("\n--- config B ---", flush=True)
    r_b = construir_y_evaluar(fuentes_activas=FUENTES_B, features=FEATURES_B, sanear=SANEAR_B)
    print(f"\nlas dos configs listas en {time.time()-t0:.0f}s", flush=True)

    print(f"\nranker.ndcg_pop A: {r_a['ndcg_pop']:.6f}  B: {r_b['ndcg_pop']:.6f}")
    print(f"ranker.ndcg_als A: {r_a['ndcg_als']:.6f}  B: {r_b['ndcg_als']:.6f}")

    usuarios_comunes = sorted(set(r_a["ndcg_por_usuario"]) & set(r_b["ndcg_por_usuario"]))
    valores_a = np.array([r_a["ndcg_por_usuario"][u] for u in usuarios_comunes])
    valores_b = np.array([r_b["ndcg_por_usuario"][u] for u in usuarios_comunes])
    _reportar_pareado(valores_a, valores_b, nombre_a, nombre_b)


if __name__ == "__main__":
    main()
