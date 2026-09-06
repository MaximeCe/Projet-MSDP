#!/usr/bin/env python3
"""
auto_params.py — Estimation automatique des paramètres MSDP depuis un flat,
et écriture dans ms.par (Fortran) et ms.yml (Python).

Basé sur le prototype validé (docs/prototype-nm-estimation.md) : détection des
canaux par maxima locaux sur coupes 1D à Y fixe + vote multi-lignes.

Ce module est la brique "A-auto" de la spec multi-instrument. Il détermine :
  - im, jm      : dimensions image (header FITS)
  - nm          : nombre de canaux (vote sur les pics)
  - ja1..3      : positions des 3 coupes de détection (réparties dans la zone utile)
  - interc      : espacement inter-canaux (médiane des pics)
  - mingrad     : seuil de gradient (percentile), réglé par la boucle de correction
  - xdel/jtriple: valeurs par défaut (25, 1)

Usage:
    from auto_params import estimate_params, write_ms_par, write_ms_yml
    cfg = estimate_params("flat.fit", "dark.fit")
    cfg.write(ms_par_out="ms.par.auto", ms_yml_out="ms.yml.auto")
"""
from pathlib import Path
import numpy as np
from astropy.io import fits

# bornes des tableaux Fortran (newgeom) — doivent rester ≤ 4096
MAX_DIM = 4096


# ---------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------
def load_flat(path):
    """Retourne l'image flat en float (shape as-Is from FITS header)."""
    with fits.open(path) as h:
        data = np.asarray(h[0].data, dtype=np.float64)
        header = h[0].header
    return data, header


def smooth(x, w):
    if w <= 1:
        return x.copy()
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


# ---------------------------------------------------------------------------
# Détection du nombre de canaux (nm)
# ---------------------------------------------------------------------------
def count_peaks(cut, lissage=31, distance=15, prom_frac=0.30):
    """Nombre de maxima locaux sur une coupe, après lissage."""
    try:
        from scipy.signal import find_peaks
    except ImportError:
        # repli : max local simple — retourne la LISTE des pics (API identique
        # à find_peaks), PAS leur nombre
        s = smooth(cut, lissage)
        peaks = []
        for i in range(1, len(s) - 1):
            if s[i] > s[i-1] and s[i] >= s[i+1]:
                if not peaks or i - peaks[-1] >= distance:
                    peaks.append(i)
        return peaks
    s = smooth(cut, lissage)
    prom = prom_frac * (np.max(s) - np.min(s))
    peaks, _ = find_peaks(s, distance=distance, prominence=prom)
    return list(peaks)


def estimate_nm(flat, y_frac=(0.15, 0.85), n_lines=15, lissages=(21, 31, 41)):
    """
    Estime nm par vote multi-lignes (méthode robuste du prototype).
    Retourne (nm, confiance_en_%, liste des pics d'une coupe de référence).
    """
    ny, nx = flat.shape
    y0, y1 = int(ny * y_frac[0]), int(ny * y_frac[1])
    lines = list(range(y0, y1, max(1, (y1 - y0) // n_lines)))
    from collections import Counter
    votes = Counter()
    ref_peaks = None
    for w in lissages:
        for Y in lines:
            p = count_peaks(flat[Y, :], lissage=w)
            if len(p) >= 2:
                votes[len(p)] += 1
                if ref_peaks is None:
                    ref_peaks = p
    if not votes:
        raise ValueError("Impossible d'estimer nm : aucun canal détecté sur le flat.")
    nm, nb_votes = votes.most_common(1)[0]
    total = sum(votes.values())
    conf = 100.0 * nb_votes / total
    return nm, conf, ref_peaks


def estimate_ja(flat, nm, y_frac=(0.15, 0.5, 0.85)):
    """3 coupes de détection, équiréparties sur la hauteur J du CCD.
    Fractions standard [0.15, 0.5, 0.85] de jm (proches des ja Meudon
    [151, 501, 851] pour jm=1024). Retourne [ja1, ja2, ja3] (1-based)."""
    ny = flat.shape[0]
    vals = [max(2, min(int(ny * f), ny - 2)) for f in (0.15, 0.5, 0.85)]
    return vals


def estimate_ja_bords(flat, nm=None, margin=0.10):
    """Positionne les 3 coupes ja à partir de la bande utile des canaux.

    Méthode de l'utilisateur : on moyenne le flat le long de i pour obtenir un
    profil 1D `profil(j) = mean_i(flat)`. L'entrée et la sortie de la zone des
    canaux correspondent aux deux plus grands pics (signe opposé) de la dérivée
    d(profil)/dj. On place :
        ja1 = entrée + 10% de la bande
        ja3 = sortie - 10% de la bande
        ja2 = milieu de la bande

    Garde-fou : si les pics ne sont pas assez nets (signal faible / bande trop
    étroite), on retombe sur l'équirépartition ([0.15,0.5,0.85]·jm).
    """
    ny = flat.shape[0]
    profile = flat.mean(axis=1)                       # (ny,)
    deriv = np.gradient(profile)
    smooth = np.convolve(deriv, np.ones(7) / 7, mode="same")

    j_entry = int(np.argmax(smooth))                  # dérivée max (entrée)
    j_exit = int(np.argmin(smooth))                   # dérivée min (sortie)
    band_width = abs(j_exit - j_entry)
    pr_range = float(np.max(profile) - np.min(profile))
    edge = max(abs(smooth[j_entry]), abs(smooth[j_exit]))

    # Garde-fou : bande trop étroite ou pic pas assez marqué par rapport à l'amplitude
    if band_width < 0.10 * ny or pr_range <= 0 or edge < 0.02 * pr_range:
        return estimate_ja(flat, nm)

    lo, hi = min(j_entry, j_exit), max(j_entry, j_exit)
    width = hi - lo
    ja1 = int(lo + margin * width)
    ja3 = int(hi - margin * width)
    ja2 = (lo + hi) // 2
    return [max(2, min(ny - 2, ja1)),
            max(2, min(ny - 2, ja2)),
            max(2, min(ny - 2, ja3))]


def _edge_regularity(flat, ja, nm):
    """Score de régularité des bords détectés à une coupe ja.
    Idéal : ~nm bords bien espacés. Retourne (nb_bords, écart-type/espacemoyen).
    Plus faible écart-type = plus régulier."""
    Y = ja - 1 if ja > 0 else 0
    Y = int(max(0, min(flat.shape[0] - 1, Y)))
    peaks = count_peaks(flat[Y, :], lissage=21)
    if len(peaks) < 2:
        return 10 ** 9  # très mauvais
    sp = np.diff(peaks)
    med = float(np.median(sp))
    if med <= 0:
        return 10 ** 9
    # proche de nm-1 intervalles, espacement régulier
    n_penalty = abs(len(peaks) - nm)
    reg = float(np.std(sp) / med) if len(sp) > 1 else 0.0
    return n_penalty * 3.0 + reg


def estimate_ja_best(flat, nm):
    """Essaie plusieurs jeux de ja (équirépartition ± décalages) et garde celui
    donnant la meilleure régularité de bords. Retourne [ja1, ja2, ja3]."""
    ny = flat.shape[0]
    base = [max(2, min(int(ny * f), ny - 2)) for f in (0.15, 0.5, 0.85)]
    candidates = [base]
    # variations ±5% de la hauteur, pour roder autour de l'équirépartition
    step = max(2, int(ny * 0.05))
    for d1 in (-step, 0, step):
        for d3 in (-step, 0, step):
            cand = [max(2, min(ny - 2, base[0] + d1)),
                    base[1],
                    max(2, min(ny - 2, base[2] + d3))]
            candidates.append(cand)
    best = base
    best_score = 10 ** 9
    for cand in candidates:
        s = sum(_edge_regularity(flat, j, nm) for j in cand) / len(cand)
        if s < best_score:
            best_score, best = s, cand
    return best


def estimate_interc(ref_peaks):
    """Espacement inter-canaux = médiane des écarts entre pics."""
    if not ref_peaks or len(ref_peaks) < 2:
        return None
    sp = np.diff(ref_peaks)
    med = float(np.median(sp))
    # filtrer les outliers (écarts > 1.5x la médiane) et moyenne
    clean = sp[np.abs(sp - med) < 1.5 * med]
    return float(np.mean(clean)) if len(clean) else med


# ---------------------------------------------------------------------------
# Sélection de nm par cohérence (9 ou 18) + balayage de mingrad
# ---------------------------------------------------------------------------
def estimate_nm_best(flat, ja, candidates=(9, 18)):
    """Choisit nm parmi les candidats (9, 18) selon la COHÉRENCE des bords.

    On utilise le score de régularité d'edges (`_edge_regularity`) évalué aux
    3 coupes ja : le bon nm est celui qui rend les bords aussi proches que
    possible d'un motif ~nm régulier. Ce n'est PAS une formule d'espacement
    (interc ≈ im/nm est ambigu à cause des faux pics internes) — c'est le
    juge de cohérence réel, le même que le pipeline valide.
    """
    best, best_score = None, 10 ** 9
    for nm in candidates:
        s = sum(_edge_regularity(flat, j, nm) for j in ja) / len(ja)
        if s < best_score:
            best_score, best = s, nm
    return best if best is not None else 9


def _edge_regularity_mingrad(flat, ja, nm, mingrad):
    """[OBSOLÈTE — à ne pas utiliser pour estimer mingrad]

    Ce proxy a été abandonné : il normalise le gradient sur 0-100 et sélectionne
    un seuil (~55) qui ne correspond PAS au mingrad réel de newgeom (le bon est
    18). mingrad est un paramètre INTERNE du code Fortran (échelle de
    normalisation propre à l'implémentation), non dérivable du flat seul. La
    boucle de correction du pipeline est le vrai sélecteur.

    Conserver pour référence/diagnostic uniquement.

    Compte les bords longs (maxima locaux de |dI/di| > mingrad, échelle 0-100)
    sur chaque coupe ja, et pénalise l'écart au nombre attendu (nm+1 frontières)
    ainsi que l'irrégularité de leur espacement. Plus bas = mieux.
    """
    expected = nm + 1
    total = 0.0
    for jj in ja:
        Y = int(max(0, min(flat.shape[0] - 1, jj - 1)))
        cut = flat[Y, :]
        grad = np.abs(np.gradient(cut))
        gmax = float(grad.max())
        if gmax <= 0:
            total += 10 ** 9
            continue
        norm = 100.0 * grad / gmax
        edges = [i for i in range(1, len(norm) - 1)
                 if norm[i] >= norm[i - 1] and norm[i] > norm[i + 1]
                 and norm[i] >= mingrad]
        if len(edges) < 2:
            total += 10 ** 9
            continue
        sp = np.diff(edges)
        med = float(np.median(sp))
        if med <= 0:
            total += 10 ** 9
            continue
        n_penalty = abs(len(edges) - expected)
        reg = float(np.std(sp) / med) if len(sp) > 1 else 0.0
        total += n_penalty * 3.0 + reg
    return total


def estimate_mingrad(flat, ja, nm, lo=5, hi=95, step=5):
    """[OBSOLÈTE — voir _edge_regularity_mingrad]

    Abandonnée : le proxy de cohérence sur le flat produit un seuil (~55) qui
    ne correspond pas au mingrad de newgeom (18). Ne pas utiliser pour
    estimer mingrad ; garder mingrad=18 + boucle de correction du pipeline.
    Conservée pour référence/diagnostic uniquement.

    Sélectionne mingrad par balayage : le seuil qui rend la géométrie
    cohérente (nb de bords ≈ nm+1, espacement régulier) au lieu d'une valeur
    fixe. Retourne le mingrad optimal (int).

    ⚠️ La grille doit monter JUSQU'À ~95 : la courbe de score décroît puis
    remonte (un minimum local existe là où ~nm+1 bords émergent). Un balayage
    trop étroit (ex. 5-40) ne voit que le côté descendant et retombe
    faussement sur la borne haute.
    """
    best_mg, best_score = None, 10 ** 9
    for mg in range(lo, hi + 1, step):
        s = _edge_regularity_mingrad(flat, ja, nm, mg)
        if s < best_score:
            best_score, best_mg = s, mg
    return best_mg if best_mg is not None else 18


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------
class Params:
    """Paramètres estimés, prêts à écrire dans les configs."""
    def __init__(self, im, jm, nm, ja, interc, mingrad=18, xdel=25, jtriple=1,
                 metadata=None):
        self.im, self.jm, self.nm = im, jm, nm
        self.ja = ja
        self.interc = interc
        self.mingrad = mingrad
        self.xdel, self.jtriple = xdel, jtriple
        self.metadata = metadata or {}

    def write_ms_par(self, path):
        """Écrit un ms.par dans le format a8,i8 (compatible par1 Fortran)."""
        # charge le ms.par de base comme modèle
        template = Path(__file__).resolve().parent.parent.parent / "src" / "fortran" / "ms.par"
        lines = []
        if template.exists():
            lines = open(template).read().split("\n")
        # surcharge les champs estimés
        import re as _re
        def set_val(name, val):
            nonlocal lines
            comment = ""
            for l in lines:
                if l[:8].strip() == name:
                    # commentaire = tout après la valeur numérique (2+ espaces)
                    m = _re.search(r"\d+\s{2,}(.*)", l)
                    comment = m.group(1).rstrip() if m else ""
                    break
            # format a8 + valeur i8. NB: le nom doit être ALIGNÉ À DROITE sur 8 colonnes
            # (par1 cherche '     ja1' = 5 espaces + nom), comme dans ms.par original.
            newline = f"{name:>8}{val:>8}   {comment}".rstrip()
            applied = False
            for i, l in enumerate(lines):
                if l[:8].strip() == name:
                    lines[i] = newline
                    applied = True
                    break
            if not applied:
                lines.append(newline)
        set_val("im", self.im)
        set_val("jm", self.jm)
        set_val("nm", self.nm)
        set_val("ja1", self.ja[0])
        set_val("ja2", self.ja[1])
        set_val("ja3", self.ja[2])
        if self.interc:
            set_val("interc", int(round(self.interc)))
        set_val("mingrad", self.mingrad)
        set_val("xdel", self.xdel)
        set_val("jtriple", self.jtriple)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def write_ms_yml(self, path):
        """Écrit un ms.yml (YAML) avec les champs estimés."""
        import yaml
        template = Path(__file__).resolve().parent.parent.parent / "src" / "python" / "ms.yml"
        params = {}
        if template.exists():
            params = yaml.safe_load(open(template)) or {}
        params["im"] = self.im
        params["jm"] = self.jm
        params["nm"] = self.nm
        params["ja"] = self.ja
        if self.interc:
            params["interc"] = int(round(self.interc))
        params["mingrad"] = self.mingrad
        params["xdel"] = self.xdel
        params["jtriple"] = self.jtriple
        with open(path, "w") as f:
            yaml.safe_dump(params, f, sort_keys=False, allow_unicode=True)
        return path


def estimate_params(flat_path, dark_path=None, mingrad=18):
    """
    Estime tous les paramètres depuis un flat field.
    Retourne un objet Params.
    """
    flat, header = load_flat(flat_path)
    if flat.ndim == 3:
        flat = flat[0]
    if flat.ndim != 2:
        raise ValueError(f"Flat doit être 2D, got {flat.ndim}D ({flat.shape})")
    im, jm = flat.shape[1], flat.shape[0]   # (col, row) = (im, jm)

    # dimensions bornées à 4096 (max des tableaux Fortran)
    if im > MAX_DIM or jm > MAX_DIM:
        raise ValueError(f"Dimensions {im}x{jm} > MAX_DIM {MAX_DIM}")

    # ja : détection de la bande utile par la dérivée du profil moyen (méthode
    # utilisateur : entrée/sortie de la zone canaux, +10%/−10%, milieu),
    # avec garde-fou vers l'équirépartition si les pics ne sont pas nets.
    ja = estimate_ja_bords(flat)

    # nm : vote du prototype (diagnostic) PUIS sélection finale par cohérence
    # parmi {9, 18} — le cas où les bords sont les plus réguliers gagne.
    nm, conf, ref_peaks = estimate_nm(flat)
    nm_sel = estimate_nm_best(flat, ja, candidates=(9, 18))

    # mingrad : VALEUR PAR DÉFAUT (18). ⚠️ Estimation "proxy" abandonnée :
    # _edge_regularity_mingrad normalisait le gradient sur 0-100 et produisait
    # un seuil faux (~55) au lieu du 18 validé, car mingrad est un paramètre
    # INTERNE de newgeom (échelle de normalisation propre à l'implémentation
    # Fortran), non dérivable du flat seul. Le 18 est la valeur empirique
    # validée ; la boucle de correction du pipeline est le vrai sélecteur.
    mingrad_sel = mingrad

    interc = estimate_interc(ref_peaks)

    metadata = {
        "flat": str(flat_path),
        "dark": str(dark_path) if dark_path else None,
        "nm_conf_percent": round(conf, 1),
        "nm_vote": nm, "nm_sel": nm_sel,
        "im": im, "jm": jm,
    }
    return Params(im, jm, nm_sel, ja, interc, mingrad=mingrad_sel,
                  metadata=metadata)


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python3 auto_params.py <flat.fit> [mingrad]")
        sys.exit(1)
    flat = sys.argv[1]
    mg = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    cfg = estimate_params(flat, mingrad=mg)
    print(json.dumps({
        "im": cfg.im, "jm": cfg.jm, "nm": cfg.nm,
        "ja": cfg.ja, "interc": cfg.interc, "mingrad": cfg.mingrad,
        "metadata": cfg.metadata,
    }, indent=2))
    print(f"\nÉcriture : 'ms.par.auto' + 'ms.yml.auto' dans data/output/")
    outdir = Path("data/output"); outdir.mkdir(parents=True, exist_ok=True)
    cfg.write_ms_par(str(outdir / "ms.par.auto"))
    cfg.write_ms_yml(str(outdir / "ms.yml.auto"))
    print("OK")