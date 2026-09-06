#!/usr/bin/env python3
"""
Vérification de la réussite du pipeline Fortran MSDP.

Ce script contrôle que les RÉSULTATS du pipeline sont bons, et pas seulement
que l'algorithme s'est exécuté jusqu'au bout. Il analyse le fichier
``ACDF2.lis`` (en-têtes de coins des 9 canaux) produit par ms2.f et s'assure
que la géométrie détectée est physiquement cohérente.

Critères vérifiés :
  1. Présence et format du fichier : 9 lignes (9 canaux) × 8 valeurs numériques.
  2. Aucune valeur nulle (0.00) et aucun NaN — un 0 signifie un canal non détecté.
  3. Ordre des canaux : les centres X croissent strictement de canal en canal.
  4. Régularité des centres X : l'écart inter-canaux est stable (tolérance).
  5. Largeur de canal cohérente : écart-type / moyenne < tolérance (pas de canal
     replié ou dégénéré).
  6. Bornes physiques : chaque coordonnée est dans l'image CCD permutée
     (X ∈ [0,im], Y ∈ [0,jm]).

Sorties : rapport lisible + code de retour (0 = succès, 1 = échec).

Usage :
    python3 check_fortran_pipeline.py [ACDF2.lis [ms_par]]
      - ACDF2.lis : fichier à contrôler (défaut: data/output/ACDF2.lis)
      - ms_par    : optionnel, ms.par/ms.yml pour l'afficher en tête de rapport
"""

import sys
import math
from pathlib import Path

# ------- Configuration -------
DEFAULT_ACDF2 = Path("data/output/ACDF2.lis")
DEFAULT_MS_PAR = Path("data/output/ms_par_run_001.par")  # dernier ms.par disponible
NM_EXPECTED_DEF = 9     # nombre de canaux par défaut (si le paramètre est absent)
COL_PER_ROW = 8          # A_X, C_X, D_X, F_X, A_Y, C_Y, D_Y, F_Y
# Bornes physiques (image permutée du pipeline) — défauts
IM_PIX_DEF = 1536
JM_PIX_DEF = 1536
MAX_ABS = 2500.0         # valeur absolue maximale raisonnable
# Tolérances de régularité
CTR_X_TOL_FRAC = 0.25    # tolérance relative de l'écart-type des pas de centres X
WIDTH_TOL_FRAC = 0.35    # tolérance relative de l'écart-type des largeurs de canal


class Config:
    """Paramètres de validation, chargés depuis ms.par (ou valeurs par défaut)."""
    def __init__(self, nm, im_pix, jm_pix, ms_par_path=None):
        self.nm = nm
        self.im_pix = im_pix
        self.jm_pix = jm_pix
        self.ms_par_path = ms_par_path


def load_config_fortran(ms_par_path: Path | None) -> Config:
    """Charge nm, im, jm depuis un fichier ms.par (format a8, i8)."""
    nm, im, jm = NM_EXPECTED_DEF, IM_PIX_DEF, JM_PIX_DEF
    if ms_par_path and Path(ms_par_path).exists():
        try:
            with open(ms_par_path) as f:
                for line in f:
                    name = line[:8].strip()
                    try:
                        val = int(line[8:19].strip())
                    except ValueError:
                        continue
                    if name == "nm":
                        nm = val
                    elif name == "im":
                        im = val
                    elif name == "jm":
                        jm = val
        except Exception:
            pass  # silencieux : on garde les défauts
    return Config(nm, im, jm, ms_par_path)


def parse_acdf2(path: Path) -> list[list[float]]:
    """Lit le fichier et retourne une liste de 9 listes de 8 floats."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except ValueError:
                    raise ValueError(f"valeur non numérique « {p} » dans {path}")
            rows.append(vals)
    return rows


def stddev(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def run_checks(path: Path, config: Config | None = None) -> tuple[bool, list[str]]:
    """Effectue tous les contrôles. Retourne (ok, liste de messages)."""
    if config is None:
        config = load_config_fortran(DEFAULT_MS_PAR)
    nm_exp = config.nm
    im_pix, jm_pix = config.im_pix, config.jm_pix
    ok = True
    msgs = []
    add = msgs.append

    # -- 1. Format ------------------------------------------------
    if not path.exists():
        return False, [f"❌ ACDF2.lis introuvable : {path}"]
    try:
        rows = parse_acdf2(path)
    except Exception as e:
        return False, [f"❌ Lecture impossible de {path} : {e}"]

    if len(rows) != nm_exp:
        ok = False
        add(f"❌ Nombre de canaux : {len(rows)} (attendu {nm_exp})")
    else:
        add(f"✔  {nm_exp} canaux")

    for i, r in enumerate(rows):
        if len(r) != COL_PER_ROW:
            ok = False
            add(f"❌ Canal {i+1}: {len(r)} valeurs (attendu {COL_PER_ROW})")

    # -- 2. Valeurs nulles / NaN / bornes -------------------------
    bad_zero = bad_nan = bad_bound = 0
    for i, r in enumerate(rows[:nm_exp]):
        if len(r) != COL_PER_ROW:
            continue
        x = r[0:4]   # A_X, C_X, D_X, F_X
        y = r[4:8]   # A_Y, C_Y, D_Y, F_Y
        for v in x + y:
            if math.isnan(v) or math.isinf(v):
                bad_nan += 1
            elif abs(v) < 1e-6:
                bad_zero += 1
            elif abs(v) > MAX_ABS:
                bad_bound += 1
        # Bornes physiques
        for v in x:
            if not (0 <= v <= im_pix):
                bad_bound += 1
        for v in y:
            if not (0 <= v <= jm_pix):
                bad_bound += 1
    if bad_zero:
        ok = False
        add(f"❌ {bad_zero} valeur(s) nulle(s) (canal non détecté ou énantiomère)")
    if bad_nan:
        ok = False
        add(f"❌ {bad_nan} valeur(s) NaN/infini (overflow intersec)")
    if bad_bound:
        ok = False
        add(f"❌ {bad_bound} valeur(s) hors bornes physiques")

    # -- 3. Centres X croissants et réguliers ---------------------
    if len(rows) >= nm_exp and all(len(r) == COL_PER_ROW for r in rows[:nm_exp]):
        centers = [(r[0] + r[1] + r[2] + r[3]) / 4.0 for r in rows[:nm_exp]]
        # Croissance stricte
        increases = [centers[i+1] > centers[i] for i in range(len(centers)-1)]
        if not all(increases):
            n_bad = sum(1 for inc in increases if not inc)
            ok = False
            add(f"❌ Centres X non croissants ({n_bad} décroissance(s))")
        else:
            add("✔  Centres X strictement croissants")
        # Régularité du pas inter-canaux
        steps = [centers[i+1] - centers[i] for i in range(len(centers)-1)]
        if steps:
            mean_step = sum(steps) / len(steps)
            if mean_step > 0:
                rel = stddev(steps) / mean_step
                if rel > CTR_X_TOL_FRAC:
                    ok = False
                    add(f"❌ Pas inter-canaux irrégulier (rel.stdev={rel:.2%})")
                else:
                    add(f"✔  Pas inter-canaux régulier ({mean_step:.1f} px/ canal, σ={rel:.2%})")
            else:
                ok = False
                add("❌ Pas inter-canaux moyen non positif")

        # -- 4. Largeur de canal cohérente ------------------------
        # Largeur = droite - gauche  (D-A et F-C)
        widths = []
        for r in rows[:nm_exp]:
            w1 = r[2] - r[0]   # D_X - A_X
            w2 = r[3] - r[1]   # F_X - C_X
            if w1 > 0:
                widths.append(w1)
            if w2 > 0:
                widths.append(w2)
        if widths:
            mean_w = sum(widths) / len(widths)
            if mean_w > 0:
                rel = stddev(widths) / mean_w
                if rel > WIDTH_TOL_FRAC:
                    ok = False
                    add(f"❌ Largeurs de canaux incohérentes (rel.stdev={rel:.2%})")
                else:
                    add(f"✔  Largeurs cohérentes ({mean_w:.1f} px, σ={rel:.2%})")
            else:
                ok = False
                add("❌ Largeur de canal moyenne non positive")

    # -- Bilan ------------------------------------------------
    add("")
    add("SUCCÈS : la géométrie du pipeline Fortran est cohérente." if ok
        else "ÉCHEC : des anomalies ont été détectées dans la géométrie.")
    return ok, msgs


def main() -> int:
    acdf2_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ACDF2
    ms_par = sys.argv[2] if len(sys.argv) > 2 else str(DEFAULT_MS_PAR)

    config = load_config_fortran(ms_par)

    print("=" * 60)
    print("  Vérification du pipeline Fortran MSDP")
    print(f"  Fichier : {acdf2_path}")
    if ms_par:
        print(f"  Paramètres : {ms_par}")
    print(f"  Config : nm={config.nm}, im={config.im_pix}, jm={config.jm_pix}")
    print("=" * 60)

    ok, msgs = run_checks(acdf2_path, config)
    for m in msgs:
        print(f"  {m}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())