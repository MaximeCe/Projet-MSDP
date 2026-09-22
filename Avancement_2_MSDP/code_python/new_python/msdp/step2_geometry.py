"""Étape 2 — Géométrie des canaux MSDP (portage de ``ms2.f`` `newgeom`).

Reproduit la détection de géométrie des canaux : à partir de l'image moyenne
(``flat − dark``) **permutée**, on détecte l'emplacement des inter-canaux sur
3 coupes horizontales puis les bords verticaux, et on en déduit les 6 points
ABC DEF + les coins A..F par intersection de droites.

⚠️ Ce module est porté depuis **`src/fortran/new/ms2.f`** — qui est en plusieurs
points DIFFÉRENT du port historique ``src/python/ms2.py``. Les différences
prises en compte ici (fidélité au ``new/``) :
- `ja(2)` est **calculé** ``(ja1+ja3)/2`` (non lu dans ms.par) ;
- les seuils de gradient viennent de ``mgx``/``mgy`` (=12/12), PAS de
  ``mingrad``/``interp`` (qui n'existent pas dans ms.par) ;
- ``interp`` est commenté dans le code → ``smax`` est **toujours** appelé ;
- les coordonnées portent un ``+0.5`` (``xx = i + eps - 1 + 0.5``) ;
- les bords verticaux k,l,m,n utilisent la **normalisation globale**
  ``zmax/zgmax`` (celle de la coupe centrale), pas une valeur locale ;
- les points mémoire sont stockés aux lignes **l=7..10** (0-based 6..9), et
  les étiquettes B/E sont aux lignes 12/15 (0-based), C/D/F aux 13/14/16.

Les indices de points (0-based, lignes) :
    0..5   : a b c d e f   (bords horizontaux, 3 coupes × gauche/droite)
    6..9   : k l m n       (bords verticaux)
    10..15 : A B C D E F   (coins extrapolés)
La correspondance exacte des étiquettes : suivant le code ``new/``,
``xx(11)=A, xx(12)=B, xx(13)=C, xx(14)=D, xx(15)=E, xx(16)=F`` (0-based ul
indices 10..15).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from msdp.config import Config

# dimensions canoniques du CCD (ms2.f newgeom hardcode im=1536, jm=1024, nm=9;
# on les lit néanmoins depuis la config pour être paramétrable).
_DEF_IM, _DEF_JM, _DEF_NM = 1536, 1024, 9


def _read_averaged_python(path: str | Path, isp: int, jsp: int) -> np.ndarray:
    """Lit un binaire moyen produit par l'étape Python 1 (sans marqueurs).

    Format consigne : entête 512 int32, puis ``isp×jsp`` int16 en ordre
    colonne-major (F). Renvoie l'image ``(isp, jsp)`` (esp = X post-permutation).

    Parameters
    ----------
    path : str | Path
        Chemin du binaire moyen.
    isp, jsp : int
        Dimensions (X, Y) du champ permuté.

    Returns
    -------
    np.ndarray
        Image ``(isp, jsp)`` dtype int16.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    header_n = 512 * 4
    n_val = isp * jsp
    data = np.frombuffer(raw[header_n:header_n + n_val * 2], dtype="<i2")
    return data.reshape((isp, jsp), order="F")


def _load_meanflat(dark: str | Path, flat: str | Path,
                   isp: int, jsp: int) -> np.ndarray:
    """Charge dark & flat moyens et calcule le ``flat − dark`` (commes ms1/ms2).

    Le fichier moyen produit par ms1.f est **permuté** ``(jsp, isp)`` =
    ``(1024, 1536)`` (X CCD transmute en Y). ``newgeom`` (ms2.f) travaille
    ensuite sur ``meanflat(im=1536, jm=1024)`` dont les indices sont
    ``(i=1..1536, j=1..1024)`` — la géométrie est donc calculée sur la
    **transposée** du fichier moyen (fidèle au ``SRECT``/port validé ms2.py,
    qui fait ``if shape != (im,jm): data = data.T``).

    La soustraction ``lec(i)=lec(i)-lecx(i)`` avec clamp ``lec<0 → 1`` se fait
    ligne à ligne pendant la lecture (idc=1).

    Parameters
    ----------
    dark, flat : str | Path
        Binaires moyens (dark, flat), permutés ``(1024, 1536)``.
    isp, jsp : int
        Dimensions du fichier moyen permuté (X, Y) — isp=js_ccd=1024,
        jsp=is_ccd=1536.

    Returns
    -------
    np.ndarray
        ``meanflat (im,jm) = (1536, 1024)`` = transposée de (flat−dark),
        dtype int16, valeurs ≥ 1.
    """
    d = _read_averaged_python(dark, isp, jsp).astype(np.int32)   # (isp,jsp) permuté
    f = _read_averaged_python(flat, isp, jsp).astype(np.int32)
    mf = f - d
    np.clip(mf, 1, None, out=mf)          # lec<0 → 1
    mf = mf.T                              # (jsp, isp) -> (im, jm) pour newgeom
    return mf.astype(np.int16)


@dataclass
class GeometryResult:
    """Résultat de la détection de géométrie.

    Attributes
    ----------
    xx : np.ndarray
        Coordonnées X des points, ``(20, nm)`` (0-based : a..F).
    yy : np.ndarray
        Coordonnées Y des points, ``(20, nm)``.
    distorsions : np.ndarray
        ``(2, nm)`` — écarts de distortion (diagnostic, non utilisé plus loin).
    val_qm : float
        Moyenne quadratique des distorsions (px).
    ja : list[int]
        Positions (1-based) des 3 coupes horizontales ``[ja1, ja2, ja3]``.
    """

    xx: np.ndarray
    yy: np.ndarray
    distorsions: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    val_qm: float = 0.0
    ja: list[int] = field(default_factory=list)
    # données de tracé fidèle au Fortran (newgeom) :
    aa: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    bb: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    cc: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    zc: np.ndarray = field(default_factory=lambda: np.zeros(0))   # intensité centrale normalisée
    zgc: np.ndarray = field(default_factory=lambda: np.zeros(0))  # gradient central normalisé
    mgx: int = 0            # seuil gradient horizontal (mgx)
    mgy: int = 0            # seuil gradient vertical (mgy)
    xdel: float = 25.0      # décalage vertical (xdel)
    zmax: float = 1.0       # normalisation zc
    zgmax: float = 1.0      # normalisation zgc

    def acdf2(self) -> np.ndarray:
        """Renvoie le tableau ACDF2 (nm × 8) : A,C,D,F en X puis en Y."""
        rows = []
        for n in range(self.xx.shape[1]):
            rows.append([
                self.xx[10, n], self.xx[12, n], self.xx[13, n], self.xx[15, n],
                self.yy[10, n], self.yy[12, n], self.yy[13, n], self.yy[15, n],
            ])
        return np.array(rows)

    def write_acdf2(self, path: str | Path = "ACDF2.lis") -> None:
        """Écrit ACDF2.lis (format 8f8.2 par ligne, mm canaux)."""
        with open(path, "w") as fh:
            for row in self.acdf2():
                fh.write("".join(f"{v:8.2f}" for v in row) + "\n")

    def to_xr_yr(self) -> tuple[np.ndarray, np.ndarray]:
        """Convertit (xx, yy) → (xr, yr) au format attendu par `channels`.

        newgeom (ms2.f new/) termine en écrivant, pour n=1..nm, kg=1..2,
        l=1..3 : ``lp = l + 10 + 3*(kg-1)`` puis :
            xr(n,l,kg) = yy(lp, n)     yr(n,l,kg) = xx(lp, n)
        (xr et yr sont donc TRANSPOSÉS par rapport à xx/yy.)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(xr, yr)`` chacun de shape ``(nm, 3, 2)``.
        """
        nm = self.xx.shape[1]
        xr = np.zeros((nm, 3, 2))
        yr = np.zeros((nm, 3, 2))
        for n in range(nm):
            for kg in range(2):
                for l in range(3):
                    lp = l + 10 + 3 * kg      # 0-based (= 1-based l+10+3*(kg-1))-1
                    xr[n, l, kg] = self.yy[lp, n]
                    yr[n, l, kg] = self.xx[lp, n]
        return xr, yr


# --------------------------------------------------------------------------- #
# Détection de géométrie (newgeom)
# --------------------------------------------------------------------------- #
def smax_eps(z: np.ndarray, i: int) -> float:
    """Interpolation parabolique du max de gradient (port de ``ms2.f`` `SMAX`).

    ``z`` est normalisé (déjà divisé par le max). L'offset sous-pixel renvoyé
    n'est PAS borné (fidèle au Fortran, qui ne clappe pas).

    Parameters
    ----------
    z : np.ndarray
        Signal (gradient normalisé).
    i : int
        Index (0-based) de l'approximation du maximum.

    Returns
    -------
    float
        Décalage sous-pixel ``eps``.
    """
    if i <= 0 or i >= len(z) - 1:
        return 0.5
    b = (z[i + 1] - z[i - 1]) / 2.0
    a = (z[i + 1] + z[i - 1]) / 2.0 - z[i]
    if a == 0.0:
        return 0.5
    return -b / (2.0 * a)


def _smax_parab(z: np.ndarray, i: int) -> tuple[float, float, float, float]:
    """Coefs de la parabole d'interpolation du gradient (ms2.f `SMAX` l.403-407).

    ``bpar=(z[i+1]-z[i-1])/2``, ``apar=(z[i+1]+z[i-1])/2-z[i]``,
    ``cpar=z[i]``, ``eps=-bpar/(2*apar)`` (si ``apar≠0``).

    Returns
    -------
    (apar, bpar, cpar, eps)
    """
    if i <= 0 or i >= len(z) - 1:
        a = 0.0; b = 0.0
        return a, b, z[i], 0.5
    b = (z[i + 1] - z[i - 1]) / 2.0
    a = (z[i + 1] + z[i - 1]) / 2.0 - z[i]
    if a == 0.0:
        return a, b, z[i], 0.5
    eps = -b / (2.0 * a)
    return a, b, z[i], eps


def intersect_lines(
    x1: float, y1: float, x2: float, y2: float,
    x3: float, y3: float, x4: float, y4: float,
) -> tuple[float, float]:
    """Intersection de deux droites (port de ``ms2.f`` `intersec`).

    Droite 1 (quasi-horiz.) ``x = a*y + b`` passant par (x1,y1),(x2,y2).
    Droite 2 (quasi-vert.)  ``y = c*x + d`` passant par (x3,y3),(x4,y4).

    Le Fortran ``new/`` écrit ``xres=(a*d+b)/(1.-ac)`` avec ``ac`` VARIABLE
    IMPLICITE non déclarée (≈0) — on utilise ici le dénominateur
    mathématiquement correct ``1 - a*c`` (documenté et validé, écart ≈ 0 car
    ``a*c≈0`` — voir skill `python-port-fixes` Bug 3/note.

    Returns
    -------
    tuple[float, float]
        ``(xres, yres)``.
    """
    if abs(y2 - y1) < 1e-10 or abs(x3 - x4) < 1e-10:
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    a = (x2 - x1) / (y2 - y1)
    b = x1 - a * y1
    c = (y3 - y4) / (x3 - x4)
    d = y3 - c * x3
    denom = 1.0 - a * c
    if abs(denom) < 1e-10:
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    xres = (a * d + b) / denom
    yres = c * xres + d
    return xres, yres


def detect_geometry(meanflat: np.ndarray, config: Config) -> GeometryResult:
    """Détection de la géométrie des canaux (`newgeom` de ms2.f new/).

    Parameters
    ----------
    meanflat : np.ndarray
        Image ``flat − dark`` permutée ``(isp, jsp)``.
    config : Config
        Doit contenir au moins ``is``, ``js``, ``nm``, ``ja1``, ``ja3``,
        ``mgx``, ``mgy`` (+ ``xdel`` par défaut 25).

    Returns
    -------
    GeometryResult
        Points de référence a..F pour chaque canal + ACDF2.
    """
    im = int(config.ccd_x or _DEF_IM)
    jm = int(config.ccd_y or _DEF_JM)
    nm = int(config.nm or _DEF_NM)

    ja1 = int(config.get("ja1", 151))
    ja3 = int(config.get("ja3", 851))
    ja = [ja1, (ja1 + ja3) // 2, ja3]       # ja2 calculé
    jc = ja[1]                              # coupe centrale (1-based)
    jc_idx = jc - 1

    laddx = int(config.get("laddx", 0))
    laddy = int(config.get("laddy", 0))
    mgx = int(config.get("mgx", 12))
    mgy = int(config.get("mgy", 12))
    zgx = float(mgx)                        # seuil gradient horizontal
    zgy = float(mgy)                        # seuil gradient vertical
    xdel = float(config.get("xdel", 25))

    sig = [1.0, -1.0]

    xx = np.zeros((20, nm))
    yy = np.zeros((20, nm))

    # ---------------------------------------------------------------
    # Normalisation globale (coupe centrale jc), réutilisée partout.
    # ---------------------------------------------------------------
    zc = meanflat[:, jc_idx].astype(np.float64)
    zmax = float(np.max(zc))
    zgc = np.zeros(im)
    if im > 1:
        zgc[:im - 1] = zc[1:] - zc[:-1]
    zgmax = float(np.max(np.abs(zgc[:im - 1])))
    zc_n = 100.0 * zc / zmax
    zgc_n = 100.0 * zgc / zgmax

    # coefs de parabole (ms2.f smax) pour le tracé fidèle de geo1/geo2/geo3
    aa = np.zeros((20, nm))
    bb = np.zeros((20, nm))
    cc = np.zeros((20, nm))

    i1, i2 = 5, im - 4                     # 1-based (ms2.f)
    i1_idx, i2_idx = i1 - 1, i2 - 1

    # ---------------------------------------------------------------
    # 3 coupes horizontales -> bords a..f (rows 0..5)
    # ---------------------------------------------------------------
    for nj in range(3):
        jj = ja[nj]                         # 1-based
        jj_idx = jj - 1
        # profil z(i) sur i1..i2 (option laddx = moyenne de lignes)
        if laddx == 0:
            z = meanflat[i1_idx:i2_idx + 1, jj_idx].astype(np.float64)
        else:
            z = np.zeros(i2_idx - i1_idx + 1)
            for d in range(-laddx, laddx + 1):
                z += meanflat[i1_idx:i2_idx + 1, (jj_idx + d) % jm].astype(np.float64)
            z /= float(2 * laddx + 1)
        zg = np.zeros_like(z)
        zg[:-1] = z[1:] - z[:-1]
        zg[-1] = zg[-2]                     # ms2.f : zg(i2)=zg(i2-1)

        z = 100.0 * z / zmax
        zg = 100.0 * zg / zgmax

        # détection des bords — automates goto-10 de ms2.f (l.552-598) :
        #   idx parcourt z ; si max signé trouvé, eps (smax) et
        #   xx(l,n)=i+eps-1+0.5 ; puis bascule is (1->2 est SAME n, 2->1
        #   incrémente n et, si n>nm, passe à la coupe suivante).
        #   i(1-based) = i1+idx ; numpy idx, isgn : 1=+ sig, 2=- sig.
        n = 1
        isgn = 1
        idx = 0                          # numpy index (i_F = i1 + idx)
        while n <= nm:
            idx += 1
            if idx >= len(zg) - 1:       # i out of [i1..i2] -> sécurité
                break
            if isgn == 1:
                l = nj
                s = 1.0
            else:
                l = nj + 3
                s = -1.0
            piv2 = s * zg[idx]
            if piv2 < zgx:
                continue
            piv1 = s * zg[idx - 1]
            piv3 = s * zg[idx + 1]
            if piv2 < piv1 or piv2 < piv3:
                continue
            apar, bpar, cpar, eps = _smax_parab(zg, idx)
            # xx = i + eps - 1 + 0.5 ; i = i1 + idx  (fidèle new/, signe +0.5)
            xx[l, n - 1] = (i1_idx + idx) + eps + 0.5
            yy[l, n - 1] = jj_idx
            aa[l, n - 1] = apar; bb[l, n - 1] = bpar; cc[l, n - 1] = cpar
            if isgn == 1:
                isgn = 2                 # même canal, bord opposé
            else:
                n += 1
                isgn = 1
            idx += 1                     # ms2.f : i=i+1 après détection

    # ms2.f new/: xx(15)=xx(5) (E), xx(12)=xx(2) (B) — en 0-based :
    # E=xx[14]=xx[4], B=xx[11]=xx[1] (ATTENTION : 1-based≠0-based).
    for n in range(nm):
        xx[14, n] = xx[4, n]; yy[14, n] = yy[4, n]   # E = e
        xx[11, n] = xx[1, n]; yy[11, n] = yy[1, n]   # B = b

    # distortion (diagnostic) ms2.f new/ : distort(1)=xx(2)-(xx(1)+xx(3))/2
    #                                  distort(2)=xx(5)-(xx(4)+xx(6))/2
    # (0-based : xx[1],xx[0],xx[2] et xx[4],xx[3],xx[5])
    distort = np.zeros((2, nm))
    for n in range(nm):
        distort[0, n] = xx[1, n] - (xx[0, n] + xx[2, n]) / 2.0
        distort[1, n] = xx[4, n] - (xx[3, n] + xx[5, n]) / 2.0
    valqm = float(np.sqrt(
        float(np.sum(distort[0] ** 2 + distort[1] ** 2)) / (2.0 * nm)))

    # ---------------------------------------------------------------
    # Bords verticaux k,l,m,n (rows 1-based 7..10 -> 0-based 6..9)
    # Le Fortran new/ indexe les bords horizontaux ainsi (1-based) :
    #   k: xx(1)  (=a),  l: xx(3)  (=c),  m: xx(4)  (=d),  n: xx(6)  (=f)
    # soit en 0-based : k=xx[0], l=xx[2], m=xx[3], n=xx[5].
    # ii = int(coord + 1 +/- xdel) sert de COLONNE à échantillonner ;
    # la valeur stockée est xx(l) = ii - 1 (= coord +/- xdel).
    # jj (bornes de la tranche) : k,m partent de 1 (0-based 0), l,n partent
    # de yy(ref)+1 (1-based) => 0-based yy[ref] ; tous finissent (k,m) à
    # yy(ref)+1-1=yy[ref] (inclus) et (l,n) à jm.
    # ---------------------------------------------------------------
    _V = {                  # l(1-based) : (ref_h_idx, +delta_col, jj_start_idx)
        7:  (0, +1, 0),      # k : ref=a, col=int(a+1+xdel), jj de 0
        8:  (2, +1, None),   # l : ref=c, col=int(c+1+xdel), jj de yy[c]
        9:  (3, -1, 0),      # m : ref=d, col=int(d+1-xdel), jj de 0
        10: (5, -1, None),   # n : ref=f, col=int(f+1-xdel), jj de yy[f]
    }
    for n in range(nm):
        for l in (7, 8, 9, 10):
            r = (l - 7) + 6                # 0-based row : l=7->6, 8->7, 9->8, 10->9
            h, sign_dx, _ = _V[l]
            ii = int(xx[h, n] + 1 + sign_dx * xdel)   # Fortran ii (1-based)
            if l in (7, 9):                # k, m : jj de 1 (0-based 0) à yy(ref)+1-1
                jj1_idx = 0
                jj2_stop = int(yy[h, n]) + 1          # slice up to yy(ref)+1 exclusive
                is_pos = True
            else:                          # l, n : jj de yy(ref)+1 à jm
                jj1_idx = int(yy[h, n])
                jj2_stop = jm
                is_pos = False
            if ii - 1 < 0 or ii > im or jj2_stop <= jj1_idx or len(meanflat) == 0:
                continue
            # colonne 0-based = ii-1 (le Fortran indexe meanflat(ii,jj) 1-based)
            z = meanflat[ii - 1, jj1_idx:jj2_stop].astype(np.float64)
            if laddy != 0:
                z2 = np.zeros_like(z)
                for d in range(-laddy, laddy + 1):
                    z2 += meanflat[(ii - 1 + d) % im, jj1_idx:jj2_stop].astype(np.float64)
                z = z2 / float(2 * laddy + 1)

            zg = np.zeros_like(z)
            # commes ms2.f : zg(jj)=(z(jj+1)-z(jj))*sig(is) ; is==1 -> +, is==2 -> -
            if len(zg) > 0:
                zg[:-1] = (z[1:] - z[:-1]) * (1.0 if is_pos else -1.0)
                zg[-1] = zg[-2]

            # normalisation GLOBALE (ms2.f new/: utilise zmax/zgmax de la coupe)
            z = 100.0 * z / zmax
            zg = 100.0 * zg / zgmax

            found = False
            for jj_k in range(1, len(zg) - 1):
                piv2 = zg[jj_k]
                if piv2 < zgy:
                    continue
                piv1 = zg[jj_k - 1]
                piv3 = zg[jj_k + 1]
                if piv2 < piv1 or piv2 < piv3:
                    continue
                eps = smax_eps(zg, jj_k)
                xx[r, n] = ii - 1          # ms2.f : xx(l,n)=ii-1
                yy[r, n] = jj1_idx + jj_k + eps + 0.5
                apar, bpar, cpar, _ = _smax_parab(zg, jj_k)
                # ms2.f l.709-711 : coefs stockés à la ligne l (1-based) :
                # l=7->r=6, 8->7, 9->8, 10->9  (même dramage que xx/yy)
                aa[r, n] = apar; bb[r, n] = bpar; cc[r, n] = cpar
                found = True
                break                       # ms2.f : goto 45 après détection
            if not found:
                # ne pas laisser de zéro (repris par les intersections)
                pass

    # ---------------------------------------------------------------
    # Coins A,B,C,D,E,F par intersection (rows 11..16, 0-based 10..15)
    # Correspondance verticaux (0-based) : k=6, l=7, m=8, n=9
    # Correspondance horizontaux (0-based) : a=0(+ja1) b=1(+ja2) c=2(+ja3)
    #                                        d=3(-ja1) e=4(-ja2) f=5(-ja3)
    # ---------------------------------------------------------------
    for n in range(nm):
        # A : intersection (b,a) & (k,m)   -> 0-based (b=1,a=0, k=6,m=8)
        xres, yres = intersect_lines(
            xx[1, n], yy[1, n], xx[0, n], yy[0, n],
            xx[6, n], yy[6, n], xx[8, n], yy[8, n])
        xx[10, n], yy[10, n] = xres, yres

        # B = b  (0-based 11)
        xx[11, n], yy[11, n] = xx[1, n], yy[1, n]

        # C : intersection (b,c) & (l,n)  (0-based c=2, l=7, n=9)
        xres, yres = intersect_lines(
            xx[1, n], yy[1, n], xx[2, n], yy[2, n],
            xx[7, n], yy[7, n], xx[9, n], yy[9, n])
        xx[12, n], yy[12, n] = xres, yres

        # D : intersection (e,d) & (m,k)  (0-based e=4, d=3, m=8, k=6)
        xres, yres = intersect_lines(
            xx[4, n], yy[4, n], xx[3, n], yy[3, n],
            xx[8, n], yy[8, n], xx[6, n], yy[6, n])
        xx[13, n], yy[13, n] = xres, yres

        # E = e  (0-based 14)
        xx[14, n], yy[14, n] = xx[4, n], yy[4, n]

        # F : intersection (e,f) & (n,l)  (0-based f=5, n=9, l=7)
        xres, yres = intersect_lines(
            xx[4, n], yy[4, n], xx[5, n], yy[5, n],
            xx[9, n], yy[9, n], xx[7, n], yy[7, n])
        xx[15, n], yy[15, n] = xres, yres
    return GeometryResult(xx=xx, yy=yy, distorsions=distort,
                          val_qm=valqm, ja=ja,
                          aa=aa, bb=bb, cc=cc,
                          zc=zc_n, zgc=zgc_n, mgx=mgx, mgy=mgy,
                          xdel=xdel, zmax=zmax, zgmax=zgmax)


def compute_geometry(dark: str | Path, flat: str | Path,
                     config: Config) -> GeometryResult:
    """Enchaîne lecture moyen + flat−dark + détection (API de haut niveau).

    Le fichier moyen est permuté ``(1024, 1536)`` ; ``_load_meanflat`` renvoie
    la transposée ``(1536, 1024)`` attendue par ``newgeom``.
    """
    isp, jsp = config.ccd_y, config.ccd_x   # fichier moyen permuté (1024, 1536)
    mf = _load_meanflat(dark, flat, isp, jsp)
    return detect_geometry(mf, config)