"""Module graphique — équivalents Matplotlib des figures PGPLOT du pipeline MSDP.

Toutes les figures :
- **redressent** les images en coordonnées de champ ``(X, Y)`` en arc-secondes
  (au lieu des indices crus ``i, j``), via ``milsec`` (pixel en milli-arcsec) ;
- sont en **noir et blanc** (colormap ``gray``), conformes aux exigences.

Sorties (commes run_pipeline.sh / ms2.f) :
- geo1   : courbe d'intensité centrale + gradient (coupe).
- geo2   : polygones de pavage des 9 canaux (coins A..F, ordre Fortran).
- geo3   : détail 1er canal : contour + verticaux k,l,m,n + lettres.
- geo4   : distorsion / quadrillage d'un canal redressé.
- calib  : profils de raie calibrés par canal.
- cal    : maps d'intensité normalisée par canal.
- obs / obsD1 / obsD2 : observations (brute / non calibrée / calibrée).
- ivprof1..3 : profils I/V (degrés 3, 4, mixte).

Chaque fonction isole la visualisation du calcul : entrées = arrays de calcul,
sortie = fichier PDF/PNG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non-interactif (commes PGPLOT)
import matplotlib.pyplot as plt

__all__ = [
    "plot_geo1", "plot_geo2", "plot_geo3", "plot_geo4",
    "plot_calib", "plot_calmap",
    "plot_obs", "plot_obsD1", "plot_obsD2",
    "plot_ivprof1", "plot_ivprof2", "plot_ivprof3",
]


# --------------------------------------------------------------------------- #
# Helpers communs
# --------------------------------------------------------------------------- #
def _extent(iim: int, jjm: int, milsec: int) -> tuple[float, float, float, float]:
    """Bounds (X0, X1, Y0, Y1) en arc-sec pour une grille (iim, jjm)."""
    return 0.0, iim * milsec / 1000.0, 0.0, jjm * milsec / 1000.0


def _axes2d(ax, iim: int, jjm: int, milsec: int) -> None:
    """Étiquette les axes d'une image 2D redressée en X/Y (arcsec)."""
    ax.set_xlabel("X (arcsec)"); ax.set_ylabel("Y (arcsec)")


def _mosaic(images: np.ndarray, milsec: int, path: str | Path,
            title: str, cmap: str = "gray", vmin=None, vmax=None,
            aspect: str = "equal", ncols: int = 3,
            pixels: bool = False) -> None:
    """Affiche les ``nm`` images (iim, jjm, nm) en mosaïque N&B 1:1.

    Fidèle à plotcal2 (ms2.f l.2900-2990 : des(i,j)=im(j,i), imshow sans .T,
    échelle pixel 1:1) : chaque canal est un rectangle portrait (j × i).
    """
    iim, jjm, nm = images.shape
    nrows = int(np.ceil(nm / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 11),
                             squeeze=False)
    axes = np.atleast_1d(axes).ravel()
    for n in range(nm):
        ax = axes[n]
        ax.imshow(images[:, :, n], origin="lower", cmap=cmap,
                  aspect=aspect, extent=(0, jjm - 1, 0, iim - 1),
                  vmin=vmin, vmax=vmax)
        if pixels:
            ax.set_xlabel("j (px)"); ax.set_ylabel("i (px)")
        else:
            _axes2d(ax, iim, jjm, milsec)
        ax.set_title(f"canal {n+1}", fontsize=9)
    for k in range(nm, len(axes)):
        axes[k].axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
# Géométrie — plots fidèles au Fortran (ms2.f newgeom)
# --------------------------------------------------------------------------- #
def plot_geo1(g: "object", im: int, jm: int,
              path: str | Path,
              title: str = "geo1 — coupe intensité + gradient + pavage") -> None:
    """Reproduit geo1.ps ms2.f (l.930-1046) : 3 panneaux empilés.

    1. Intensité centrale ``zc`` (normalisée 0-100) sur X=0..im-1.
    2. Gradient ``zgc`` (normalisé ±100) + seuils ±mgx.
    3. Pavage des 9 canaux (bords ABCDEF) + coupes horizontales ja1/2/3.
    (ACDF2.lis est écrit séparément — voir pipeline.)
    """
    zc, zgc, mgx = g.zc, g.zgc, g.mgx
    x = np.arange(im)
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 10),
                                     sharex=True, gridspec_kw={"hspace": 0.25})
    # (1) intensité
    a1.plot(x, zc, color="k", lw=0.8)
    a1.set_ylim(0, 100); a1.set_title(title)
    a1.set_ylabel("Intensité (norm.)")
    # (2) gradient + seuils
    a2.plot(x, zgc, color="k", lw=0.8)
    a2.axhline(mgx, color="gray", ls="--", lw=0.8)
    a2.axhline(-mgx, color="gray", ls="--", lw=0.8)
    a2.set_ylim(-100, 100)
    a2.set_ylabel("Gradient (norm.)")
    a2.text(120, mgx + 15, "mgx"); a2.text(30, -mgx - 20, "-mgx")
    # (3) pavage des canaux (ms2.f l.1018-1036 : A-B-C-F-E-D-A)
    for n in range(g.xx.shape[1]):
        o = [10, 11, 12, 15, 14, 13, 10]
        xs = [g.xx[k, n] for k in o]; ys = [g.yy[k, n] for k in o]
        a3.plot(xs, ys, color="k", lw=0.8)
    for nd in g.ja:                         # coupes horizontales ja(nd)-1
        a3.axhline(nd - 1, color="gray", ls="--", lw=0.8)
    a3.set_xlim(0, im); a3.set_ylim(0, jm)
    a3.invert_yaxis()
    a3.set_xlabel("X (pixels)"); a3.set_ylabel("Y (pixels)")
    fig.savefig(path); plt.close(fig)


def plot_geo2(g: "object", im: int, path: str | Path,
              title: str = "geo2 — zoom 1er canal : intensité, gradient, parabole") -> None:
    """Reproduit geo2.ps ms2.f (l.1050-1181) : zoom sur le 1er canal.

    2 panneaux (intensité puis gradient) centrés sur ``xmax=xx(2,1)`` ± 6,
    + seuils ±mgx + parabole d'interpolation du maximum (aa,bb,cc du point b).

    Note : le Fortran centre la fenêtre sur le pic du 1er canal (b) ; l'abscisse
    x1/x2 = xmax±6 suit le pic détecté (pas une coupe fixe).
    """
    zc, zgc, mgx = g.zc, g.zgc, g.mgx
    xmax = float(g.xx[1, 0])               # xx(2,1) — pic du 1er canal
    kmax = int(xmax)
    x1, x2 = kmax - 6 + 0.5, kmax + 6 + 0.5
    x1, x2 = max(x1, 0), min(x2, im - 1)
    x = np.arange(int(x1), int(x2) + 1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6),
                                 gridspec_kw={"hspace": 0.4})
    a1.plot(x, zc[x], color="k", lw=1)
    a1.axvline(xmax, color="k", lw=0.8)
    a1.set_ylim(0, 100)
    a1.set_title(title); a1.set_ylabel("Intensité")
    a2.plot(x, zgc[x], color="k", lw=1)
    a2.axvline(xmax, color="k", lw=0.8)
    a2.axhline(mgx, color="gray", ls="--", lw=0.8)
    a2.axhline(-mgx, color="gray", ls="--", lw=0.8)
    a2.set_ylim(-100, 100)
    # parabole d'interpolation du max (ms2.f l.1153-1177 : aa,bb,cc du pt b)
    apar, bpar, cpar = g.aa[1, 0], g.bb[1, 0], g.cc[1, 0]
    if apar != 0 or bpar != 0:
        xp = kmax + 0.5 + 0.1 * (np.arange(21) - 11)
        yp = cpar + (0.1 * (np.arange(21) - 11)) * (
            bpar + (0.1 * (np.arange(21) - 11)) * apar)
        a2.plot(xp, yp, color="k", lw=2)
        a2.text(xmax + 0.1, -80, "B")       # étiquette comme Fortran
    a2.set_xlabel("X (pixels)"); a2.set_ylabel("Gradient")
    fig.savefig(path); plt.close(fig)


def plot_geo3(xx: np.ndarray, yy: np.ndarray, xdel: float, jm: int,
              path: str | Path, title: str = "geo3 — détail 1er canal") -> None:
    """Contour + verticaux k,l,m,n + étiquettes du 1er canal (geo3.ps l.1195-1354).

    Reproduit fidèlement : 3 coupes horizontales Y1,Y2,Y3 (ja), les 6 points
    abcdef du canal 1, les 4 verticaux k,l,m,n (à xx(ref)±xdel), les coins
    A..F (liaison A-B-C-F-E-D-A) et les étiquettes.
    """
    n = 0
    x1, x2 = 0.0, 350.0
    fig, ax = plt.subplots(figsize=(9, 6))
    # 3 lignes horizontales Y1,Y2,Y3 (l.1229-1235)
    for jj in (yy[0, n], yy[1, n], yy[2, n]):
        ax.plot([x1, x2], [jj, jj], color="gray", lw=0.8, ls="--")
    # les 6 points abcdef (l.1239-1242)
    for nn in range(6):
        ax.plot([xx[nn, n]], [yy[nn, n]], "s", ms=4, color="k")
    # verticaux k,l,m,n (l.1248-1272) : xx(1,3,4,6)±xdel
    for h, sgn in ((0, +1), (2, +1), (3, -1), (5, -1)):
        xv = xx[h, n] + sgn * xdel
        if h in (0, 3):
            ax.plot([xv, xv], [0, yy[h, n]], color="k", lw=0.8, ls="--")
        else:
            ax.plot([xv, xv], [yy[h, n], jm], color="k", lw=0.8, ls="--")
    # points klmn (l.1274-1278)
    for r in range(6, 10):
        ax.plot([xx[r, n]], [yy[r, n]], "s", ms=4, color="k")
    # coins ABCDEF + liaison A-B-C-F-E-D-A (l.1283-1299)
    order = [10, 11, 12, 15, 14, 13, 10]
    ax.plot([xx[k, n] for k in order], [yy[k, n] for k in order],
            "-o", ms=3, color="k", lw=1.5)
    # étiquettes a..F (l.1303-1354, positions approximées)
    labels = "abcdefklmnABCDEF"
    for i, ch in enumerate(labels):
        dx = -30 if i in (0, 1, 6, 9, 10, 11) else 10
        dy = 10 if i not in (2, 7, 9, 12) else -30
        ax.text(xx[i, n] + dx, yy[i, n] + dy, ch, fontsize=8)
    ax.set_xlim(x1, x2); ax.set_ylim(jm, 0)
    ax.set_xlabel("X (pixels)"); ax.set_ylabel("Y (pixels)")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_geo4(xx: np.ndarray, yy: np.ndarray, nm: int,
              path: str | Path,
              title: str = "geo4 — distorsions AC/DF/AD/CF par canal") -> None:
    """Reproduit plotgeo4 ms2.f (l.1383-1584) : 8 graphes de distorsions.

    Deux colonnes (X à gauche, Y à droite) × 4 panneaux (AC, DF, AD, CF),
    chaque panneau traçant la distance |pointA-pointB| en fonction du canal n
    (fenêtre centrée sur le canal nc=5 ±5).
    """
    nc = 5
    d = {
        "AC": (abs(xx[10] - xx[12]), abs(yy[10] - yy[12])),
        "DF": (abs(xx[13] - xx[15]), abs(yy[13] - yy[15])),
        "AD": (abs(xx[10] - xx[13]), abs(yy[10] - yy[13])),
        "CF": (abs(xx[12] - xx[15]), abs(yy[12] - yy[15])),
    }
    fig, axes = plt.subplots(4, 2, figsize=(8, 9), sharex=True)
    xv = np.arange(1, nm + 1)
    for i, (name, (dx, dy)) in enumerate(d.items()):
        for col, vals, colname in ((0, dx, "X"), (1, dy, "Y")):
            ax = axes[i, col]
            ax.plot(xv, vals, "o-", ms=3, color="k", lw=1)
            ax.set_title(f"{colname} — {name}", fontsize=9)
            ax.set_xlim(0, nm + 1)
            yc = vals[nc - 1]
            ax.set_ylim(yc - 5, yc + 5)
            if i == 3:
                ax.set_xlabel("canal n")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
# Calibration & observations — mosaïques redressées N&B
# --------------------------------------------------------------------------- #
def plot_calib(cal: np.ndarray, imd: int, jmd: int, nm: int,
               x1cal: int, x2cal: int, jtr: int, jt1: int, jt2: int,
               path: str | Path,
               yln: np.ndarray | None = None,
               center: np.ndarray | None = None,
               nc: int = 5,
               il1: int = 1, il2: int = 0,
               pronoc: np.ndarray | None = None,
               profmm: np.ndarray | None = None,
               title: str = "calib — panneaux Fortran (grille 2×3)") -> None:
    """Reproduit calib.ps ms2.f : 6 panneaux (2×3).

    Ordre Fortran (chacun via pgadvance) :
      1-2. A Line center n=nc, n=nc+1  (plot_line l.2231-2284 : yln vs i=il1..il2
           + center vs i=1..imd)
      3.   B Wavelength range used for mean profile (l.2315-2377)
      4.   B Mean profile (profmean l.2637-2727 : pronoc + profmm)
      5-6. Calibration X=x1cal, X=x2cal (l.1970-2012 : 9 canaux superposés)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    xl, yl = float(jmd) - 1.0, float(imd) - 1.0
    ic = (1 + imd) // 2
    y = np.arange(1, imd + 1) - 1.0          # 0-based (y=i-1, i=1..imd)
    if il2 <= il1:
        il2 = imd

    def _axes_XY(ax):
        ax.set_xlim(0, xl); ax.set_ylim(0, yl)
        ax.set_xlabel("X"); ax.set_ylabel("Y")

    # --- 1-2. A Line center n=nc, n=nc+1 ---
    for k in range(2):
        ax = axes[k // 2, k % 2]
        n1 = nc - 1 + k                    # 0-based (nc, nc+1 en 1-based)
        ax.set_title(f"A  Line center  n = {nc + k}")
        if yln is not None and center is not None:
            # yln(i,n) pour i=il1..il2 ; center(i,n) pour i=1..imd (l.2249-2272)
            ax.plot(yln[il1 - 1:il2, n1], y[il1 - 1:il2], color="k", lw=1)
            ax.plot(center[:, n1], y, color="k", lw=1, ls="--")
        _axes_XY(ax)

    # --- 3. B Wavelength range ---
    ax = axes[1, 0]
    ax.set_title("B  Wavelength range used for mean profile")
    if center is not None:
        n0 = nc - 1                        # 0-based (Fortran n0=nc)
        cline = center[:, n0] - center[ic - 1, n0]
        for n in (1, 2):
            xc = xl / 2.0 + (2 * n - 3) * 0.5 * float(jtr)
            ax.plot(xc + cline, np.arange(1, imd + 1) - 1, color="k", lw=1)
        trj2 = float(jtr) / 2.0
        ax.axvline(xl / 2 - trj2, color="k", lw=1, ls="--")
        ax.axvline(xl / 2 + trj2, color="k", lw=1, ls="--")
        ax.text(xl / 2 - 10, yl / 2 + 50, f"Ts = {jtr}", fontsize=8)
        ax.text(xl / 2 - trj2 - 9, yl / 2, "W1", fontsize=7)
        ax.text(xl / 2 + trj2 + 2, yl / 2, "W2", fontsize=7)
        ax.text(32, yl / 5, "L1", fontsize=7)
        ax.text(88, yl / 5, "L2", fontsize=7)
    _axes_XY(ax)

    # --- 4. B Mean profile ---
    ax = axes[1, 1]
    ax.set_title("B  Mean profile")
    if profmm is not None:
        km = len(profmm)
        xjtr = float(jtr)
        xl1 = -float(jmd - jt2) / xjtr
        xl2 = float(nm) + float(jt1 - 1) / xjtr
        xk = np.array([xl1 + float(k - 1) / xjtr for k in range(1, km + 1)])
        if pronoc is not None:
            ax.plot(xk, pronoc, color="k", lw=0.8)
        ax.plot(xk, profmm, color="k", lw=1.5)
        ax.set_xlim(xl1, xl2)
        ax.set_xlabel("channels"); ax.set_ylabel("Intensity")
    else:
        _axes_XY(ax)

    # --- 5-6. Calibration X=x1cal, x2cal ---
    for k, (ax, icd) in enumerate(((axes[0, 2], x1cal), (axes[1, 2], x2cal))):
        ax.set_title(f"C  Calibration  X={icd}")
        x = np.array([n_ - 1 + (np.arange(jmd)) / (float(jmd) - 1.0)
                      for n_ in range(1, nm + 1)])
        for n in range(nm):
            ax.plot(x[n], cal[icd, :, n], color="k", lw=1)
        ax.set_xlim(0, nm); ax.set_ylim(0, 2)
        ax.set_xlabel("Channels"); ax.set_ylabel("I")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path); plt.close(fig)


def plot_calmap(cal: np.ndarray, path: str | Path,
                calbl: float = 0.8, calwh: float = 1.2,
                title: str = "cal — images d'intensité normalisée") -> None:
    """Reproduit cal.ps ms2.f (l.2022-2056 + plotcal2 l.2900-2990).

    9 panneaux (1 par canal), chacun une image ``cal(i, j, n)`` permutée
    (des(i,j)=cal(j,i) comme plotcal2) en **niveaux de gris** bornés
    [calbl, calwh], en **échelle pixel 1:1** (aspect==ratio) pour que les
    canaux apparaissent rectangulaires (portrait 123×885) comme le Fortran.
    """
    iim, jjm, nm = cal.shape
    ncols = 3
    nrows = int(np.ceil(nm / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 9),
                             squeeze=False)
    for n in range(nm):
        r, c = divmod(n, ncols)
        ax = axes[r, c]
        ax.imshow(cal[:, :, n], origin="lower", cmap="gray",
                  vmin=calbl, vmax=calwh, aspect="equal",
                  extent=(0, jjm - 1, 0, iim - 1))
        ax.set_title(f"canal {n+1}", fontsize=8)
    for k in range(nm, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path); plt.close(fig)


def plot_obs(obs: np.ndarray, milsec: int, path: str | Path,
             vmin=None, vmax=None,
             title: str = "obs — observation par canal") -> None:
    """Observation (plotcal2 nplot=2), N&B 1:1, pixels. Bornes iobsbl/iobswh."""
    _mosaic(obs, milsec, path, title, cmap="gray", vmin=vmin, vmax=vmax,
            pixels=True)


def plot_obsD1(obs: np.ndarray, milsec: int, path: str | Path,
               vmin=None, vmax=None,
               title: str = "obsD1 — non calibrée") -> None:
    """Observation non calibrée, N&B 1:1, pixels. Bornes iobsDbl/iobsDwh."""
    _mosaic(obs, milsec, path, title, cmap="gray", vmin=vmin, vmax=vmax,
            pixels=True)


def plot_obsD2(obs: np.ndarray, milsec: int, path: str | Path,
               vmin=None, vmax=None,
               title: str = "obsD2 — calibrée") -> None:
    """Observation calibrée, N&B 1:1, pixels. Bornes iobsDbl/iobsDwh."""
    _mosaic(obs, milsec, path, title, cmap="gray", vmin=vmin, vmax=vmax,
            pixels=True)


# --------------------------------------------------------------------------- #
# Profils I/V
# --------------------------------------------------------------------------- #
def _ivprof(profnf: np.ndarray, jjs: Sequence[int], milsec: int,
            path: str | Path, title: str) -> None:
    """Profils I/V pour 3 coupes de référence (X jj), en N&B."""
    nfm = profnf.shape[1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes = np.atleast_1d(axes)
    for k, ax in enumerate(axes):
        jj = jjs[k] if k < len(jjs) else jjs[0]
        jj_idx = min(jj - 1, profnf.shape[0] - 1)
        if profnf.ndim == 3:
            prof = profnf[jj_idx, :, 0]
        else:
            prof = profnf[jj_idx, :]
        xs = np.arange(nfm) * milsec / 1000.0
        ax.plot(xs, prof, color="k", lw=1.5)
        ax.set_title(f"X = {jj}", fontsize=9)
        ax.set_xlabel("λ (arcsec)"); ax.set_ylabel("I")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path); plt.close(fig)


def plot_ivprof1(profnf: np.ndarray, milsec: int, path: str | Path,
                 title: str = "ivprof1 — I/V degré 3") -> None:
    _ivprof(profnf, (21, 61, 101), milsec, path, title)


def plot_ivprof2(profnf: np.ndarray, milsec: int, path: str | Path,
                 title: str = "ivprof2 — I/V degré 4") -> None:
    _ivprof(profnf[:, :, 1], (21, 61, 101), milsec, path, title)


def plot_ivprof3(profnf: np.ndarray, milsec: int, path: str | Path,
                 title: str = "ivprof3 — I/V mixte") -> None:
    _ivprof(profnf[:, :, 2], (21, 61, 101), milsec, path, title)