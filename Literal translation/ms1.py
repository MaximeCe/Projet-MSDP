# ms1.py
"""
Traduction Python de ms1.f (version moderne, modulaire).
But :
 - Reproduire la logique principale (lecture paramètres, lecture liste/cas,
   initialisation des tableaux, boucles de calcul, écriture sortie).
 - Fournir stubs/placeholders pour ms.par et ms.lis.
 - Préparer points d'extension pour les routines de ms2 (profils, interpolation, etc.)

Dépendances : numpy
Usage :
    python ms1.py          # exécution avec paramètres par défaut/test
    import ms1             # utiliser les fonctions depuis un autre module
"""

from __future__ import annotations
import sys
import math
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ms1")

# ---------------------------
# Defaults & constants
# ---------------------------
DEFAULT_PARAMS = {
    "nx": 101,                # number of x points
    "ny": 101,                # number of y points
    "x_min": 0.0,
    "x_max": 1.0,
    "y_min": 0.0,
    "y_max": 1.0,
    "temperature": 300.0,     # example physical parameter
    "output": "ms1_out.txt",
    # add other defaults that ms.par would usually contain
}

# ---------------------------
# I/O helpers (ms.par / ms.lis)
# ---------------------------


def read_par(path: Optional[str]) -> Dict[str, Any]:
    """
    Lire ms.par si présent, sinon retourner valeurs par défaut.
    Format attendu (simple) :
        key = value
    Lignes commentées par # ou c (optionnel).
    """
    params = DEFAULT_PARAMS.copy()
    if not path:
        logger.info(
            "Aucun ms.par fourni — utilisation des paramètres par défaut.")
        return params

    p = Path(path)
    if not p.exists():
        logger.warning(
            f"Fichier de paramètres {path} introuvable — utilisation des paramètres par défaut.")
        return params

    try:
        with p.open("r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or line.lower().startswith("c"):
                    continue
                # parser simple key = value
                if "=" in line:
                    key, val = [s.strip() for s in line.split("=", 1)]
                    # tenter conversion en nombre si possible
                    if val.replace(".", "", 1).replace("-", "", 1).isdigit():
                        if "." in val:
                            v = float(val)
                        else:
                            v = int(val)
                    else:
                        v = val
                    params[key] = v
    except Exception as e:
        logger.error(
            f"Erreur lecture {path}: {e}. Utilisation des paramètres par défaut.")

    logger.info(f"Paramètres lus ({len(params)}): {list(params.keys())}")
    return params


def read_lis(path: Optional[str]) -> List[Dict[str, Any]]:
    """
    Lire ms.lis si présent, sinon retourner une liste de cas par défaut (un cas).
    Format attendu (exemple, à adapter) :
        chaque ligne = un cas ou une référence à un fichier d'entrée
    Retourne une liste de dictionnaires représentant chaque cas.
    """
    default_case = {"case_id": 1, "description": "default case"}
    if not path:
        logger.info("Aucun ms.lis fourni — création d'un cas par défaut.")
        return [default_case]

    p = Path(path)
    if not p.exists():
        logger.warning(
            f"Fichier de liste {path} introuvable — création d'un cas par défaut.")
        return [default_case]

    cases = []
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as fh:
            for i, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line or line.lower().startswith("c") or line.startswith("#"):
                    continue
                # Simple parsing : on met la ligne brute dans la description
                cases.append({"case_id": i, "description": line})
    except Exception as e:
        logger.error(
            f"Erreur lecture {path}: {e}. Utilisation d'un cas par défaut.")
        return [default_case]

    if not cases:
        cases = [default_case]
    logger.info(f"{len(cases)} cas lus depuis {path}")
    return cases

# ---------------------------
# Numerical helpers / stubs for ms2 routines
# ---------------------------


def profile_function(x: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Stub pour une fonction profil de raie (ex: gaussien).
    Ceci représente l'équivalent d'une routine qui pourrait être traduite depuis ms2.f.
    """
    # gaussian profile
    return np.exp(-0.5 * ((x - center) / width) ** 2)


# ---------------------------
# Core computation (traduction logique ms1)
# ---------------------------
def initialize_grid(params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Initialise les grilles x, y et renvoie dx, dy.
    """
    nx = int(params.get("nx", DEFAULT_PARAMS["nx"]))
    ny = int(params.get("ny", DEFAULT_PARAMS["ny"]))
    x_min = float(params.get("x_min", DEFAULT_PARAMS["x_min"]))
    x_max = float(params.get("x_max", DEFAULT_PARAMS["x_max"]))
    y_min = float(params.get("y_min", DEFAULT_PARAMS["y_min"]))
    y_max = float(params.get("y_max", DEFAULT_PARAMS["y_max"]))

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    dx = (x_max - x_min) / max(1, nx - 1)
    dy = (y_max - y_min) / max(1, ny - 1)

    logger.info(
        f"Grid initialized: nx={nx}, ny={ny}, dx={dx:.4g}, dy={dy:.4g}")
    return x, y, dx, dy


def main_compute(params: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Version principale du calcul traduit de ms1.f.
    Retourne un dictionnaire contenant les résultats (tableaux, méta).
    """
    # 1) Initialisation grille
    x, y, dx, dy = initialize_grid(params)
    nx = x.size
    ny = y.size

    # 2) Initialiser champs (exemple: champ scalaire Z(x,y))
    Z = np.zeros((ny, nx), dtype=float)  # Fortran ordering often (y,x)

    # Exemple: remplir Z avec une fonction analytique basée sur paramètres (placeholder)
    # Ici on simule un calcul qui pourrait être fait par ms1: combinaisons, boucles, etc.
    # On laisse la place pour des calculs physiques réels (forces, intensités,...)
    for j in range(ny):
        for i in range(nx):
            xi = x[i]
            yj = y[j]
            # Exemple de formule : valeur dépendant de la distance à un centre et d'un paramètre
            cx = 0.5 * (params.get("x_min", 0.0) + params.get("x_max", 1.0))
            cy = 0.5 * (params.get("y_min", 0.0) + params.get("y_max", 1.0))
            r = math.hypot(xi - cx, yj - cy)
            # combinaison arbitraire (remplacer par la logique réelle)
            Z[j, i] = math.exp(- (r ** 2) / (2 * (0.1 ** 2))) * \
                (1.0 + 0.1 * math.sin(10 * xi) * math.cos(10 * yj))

    # 3) Exemple : convolution / profil (fait appel à routine ms2)
    # Construire un vecteur d'abscisses pour profil
    x_profile = np.linspace(-0.5, 0.5, 201)
    center = 0.0
    width = 0.05
    profile = profile_function(x_profile, center, width)

    # Convolver chaque ligne avec le profil (1D convolution le long x par exemple)
    Z_conv = np.zeros_like(Z)
    # note: scipy est optionnel ; utile si disponible
    from scipy.signal import fftconvolve

    try:
        for j in range(ny):
            # convolution 1D en x (mode same)
            Z_conv[j, :] = fftconvolve(Z[j, :], profile, mode="same")
        convolved = True
    except Exception as e:
        # si scipy absent, fallback naive (plus lent)
        logger.warning(
            "scipy.signal.fftconvolve indisponible : fallback en convolution naive.")
        kernel = profile / profile.sum()
        klen = kernel.size
        pad = klen // 2
        Z_pad = np.pad(Z, ((0, 0), (pad, pad)), mode="edge")
        for j in range(ny):
            for i in range(nx):
                Z_conv[j, i] = (Z_pad[j, i:i + klen] * kernel).sum()
        convolved = False

    # 4) Calculs supplémentaires (ex: intégration, normalisation)
    total_signal = Z_conv.sum()
    if total_signal != 0.0:
        Z_norm = Z_conv / total_signal
    else:
        Z_norm = Z_conv

    # 5) Résultats
    results = {
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "Z": Z,
        "Z_conv": Z_conv,
        "Z_norm": Z_norm,
        "profile": profile,
        "convolved_with_fft": convolved,
        "meta": {
            "case": case,
            "params": params,
            "total_signal": float(total_signal),
        },
    }

    logger.info(
        f"Calcul terminé pour case {case.get('case_id')}. total_signal={total_signal:.6g}")
    return results

# ---------------------------
# Output
# ---------------------------


def write_output(results: Dict[str, Any], path: Optional[str] = None) -> None:
    """
    Écrit les résultats principaux dans un fichier texte (format simple).
    """
    out_path = Path(path or results.get("meta", {}).get(
        "params", {}).get("output", "ms1_out.txt"))
    try:
        with out_path.open("w", encoding="utf-8") as fh:
            meta = results.get("meta", {})
            fh.write("# ms1.py output\n")
            fh.write(f"# case: {meta.get('case')}\n")
            fh.write(f"# params: {meta.get('params')}\n")
            fh.write(f"# total_signal: {meta.get('total_signal')}\n")
            fh.write("\n# x y Z_norm_row_major\n")
            x = results["x"]
            y = results["y"]
            Z_norm = results["Z_norm"]
            # écriture simple : chaque ligne = x y z
            for j in range(y.size):
                for i in range(x.size):
                    fh.write(f"{x[i]:.8e} {y[j]:.8e} {Z_norm[j, i]:.8e}\n")
        logger.info(f"Résultats écrits dans {out_path}")
    except Exception as e:
        logger.error(f"Impossible d'écrire {out_path}: {e}")


# ---------------------------
# CLI / main
# ---------------------------
def main(args: Optional[List[str]] = None) -> int:
    """
    Point d'entrée principal.
    Usage CLI : python ms1.py [ms.par path] [ms.lis path]
    """
    args = list(args or sys.argv[1:])
    par_path = args[0] if len(args) >= 1 else None
    lis_path = args[1] if len(args) >= 2 else None

    params = read_par(par_path)
    cases = read_lis(lis_path)

    # itérer sur les cas
    for case in cases:
        results = main_compute(params, case)
        write_output(results, params.get("output"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
