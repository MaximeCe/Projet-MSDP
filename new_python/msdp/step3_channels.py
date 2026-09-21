"""Étape 3a — Extraction des canaux MSDP (portage de ms2.f `channels`/`COEFF`/`pix`).

À partir de la géométrie (points xr/yr de pavage ABCD de chaque canal) et de
l'**image brute** ``ima`` (= flat − dark, NON transposée — l'entrée de `geom`),
on reconstruit chaque canal sur une grille régulière ``(iim, jjm)`` :

1. ``COEFF`` calcule les coefficients d'une transformation bilinéaire
   (+ termes de distorsion quadratiques) reliant la grille canal à la grille ima ;
2. ``pix`` inverse cette transformation : coordonnée canal ``(xx,yy)`` →
   coordonnée ima ``(i+di, j+dj)`` ;
3. ``channels`` pile les canaux dans ``cymx(i,j,n)`` par interpolation bilinéaire
   sur les 4 pixels ima voisins.

La sortie ``cymx(2000,200,24)`` est alimentée à la calibration (step3_calib).

Rappels new/ :
- ``ima = meanflat.T`` (meanflat(i,j) = ima(j,i)).
- ``iim = int(li/milsec + 1.5)``, ``jjm = int(lj/milsec + 1.5)`` ; Meudon
  li=442000, lj=61000, milsec=500 → (885, 123).
- ``pix`` : xi = c1 + xx*c2 + yy*c3 + xx*yy*c4 + xx²*(c5 + yy*c6).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def channels_dims(li: int, lj: int, milsec: int) -> Tuple[int, int]:
    """Dimensions de la grille de canaux (``iim``, ``jjm``)."""
    return int(li / milsec + 1.5), int(lj / milsec + 1.5)


def coeff_from_ref(xr: np.ndarray, yr: np.ndarray, n: int,
                   im: int, jm: int, kdistor: int = 1) -> np.ndarray:
    """Coefficients de transformation d'un canal (ms2.f `COEFF`).

    ``xr(n,l,kg)`` / ``yr(n,l,kg)`` : coordonnées (X,Y) des points de référence
    du canal ``n`` (kg=1 gauche, kg=2 droite). ``xl=im-1``, ``yl=jm-1``.

    Returns
    -------
    np.ndarray
        ``coef`` shape ``(6, 2)`` ; l'axe 1 (colonne) = X puis Y.
    """
    xl = im - 1
    yl = jm - 1
    coef = np.zeros((6, 2))
    for k in (0, 1):  # k=0 -> axe X, k=1 -> axe Y
        # Indices 1-based ms2.f : x1=xr(n,1,1) ... x6=xr(n,2,2)  -> 0-based ci-dessous
        #   x1=xr(.,0,0)  x2=xr(.,2,0)  x3=xr(.,0,1)  x4=xr(.,2,1)
        #   x5=xr(.,1,0)  x6=xr(.,1,1)   ; y_* même indices mais dans yr.
        ref_x = (xr[n, 0, 0], xr[n, 2, 0], xr[n, 0, 1], xr[n, 2, 1],
                 xr[n, 1, 0], xr[n, 1, 1])
        ref_y = (yr[n, 0, 0], yr[n, 2, 0], yr[n, 0, 1], yr[n, 2, 1],
                 yr[n, 1, 0], yr[n, 1, 1])
        b1, b2, b3, b4, b5, b6 = ref_x if k == 0 else ref_y
        coef[0, k] = b1
        coef[2, k] = (b3 - b1) / yl
        if kdistor == 0:
            coef[1, k] = (b2 - b1) / xl
            coef[3, k] = (b1 + b4 - b2 - b3) / (xl * yl)
        else:
            coef[1, k] = (-3.0 * b1 - b2 + 4.0 * b5) / xl
            coef[3, k] = (3.0 * b1 + b2 - 3.0 * b3 - b4 - 4.0 * b5 + 4.0 * b6) / (xl * yl)
            coef[4, k] = 2.0 * (b1 + b2 - 2.0 * b5) / (xl * xl)
            coef[5, k] = 2.0 * (-b1 - b2 + b3 + b4 + 2.0 * b5 - 2.0 * b6) / (xl * xl * yl)
    return coef


def pix(x: float, y: float, coef: np.ndarray) -> Tuple[int, int, float, float]:
    """Coordonnée ima pour une coordonnée canal (ms2.f `pix`).

    Returns
    -------
    (i, j, di, dj) : partie entière (i,j) + fractionnaire (di,dj) de la
    coordonnée ima «complète».
    """
    xi = (coef[0, 0] + x * coef[1, 0] + y * coef[2, 0] + x * y * coef[3, 0]
          + (x * x) * (coef[4, 0] + y * coef[5, 0]))
    yj = (coef[0, 1] + x * coef[1, 1] + y * coef[2, 1] + x * y * coef[3, 1]
          + (x * x) * (coef[4, 1] + y * coef[5, 1]))
    i = int(xi)
    j = int(yj)
    return i, j, xi - i, yj - j


def extract_channels(ima: np.ndarray, xr: np.ndarray, yr: np.ndarray,
                     li: int, lj: int, milsec: int, nm: int) -> np.ndarray:
    """Extrait les canaux (ms2.f `channels`) → ``cymx(iim, jjm, nm)``.

    Parameters
    ----------
    ima : np.ndarray
        Image brute ``(im, jm)`` (flat−dark, NON transposée ; ima[j,i]).
    xr, yr : np.ndarray
        Points de référence ``(nm, 3, 2)`` sortie de newgeom.
    li, lj, milsec, nm : int
        Grandeurs du champ + taille pixel de sortie + nb de canaux.

    Returns
    -------
    np.ndarray
        ``cymx (iim, jjm, nm)`` — intensité échantillonnée de chaque canal.
    """
    iim, jjm = channels_dims(li, lj, milsec)
    cymx = np.zeros((iim, jjm, nm))

    for n in range(nm):
        coef = coeff_from_ref(xr, yr, n, iim, jjm, kdistor=1)
        for jj in range(jjm):
            yy = float(jj)              # ms2.f : xx=ii-1., yy=jj-1.
            for ii in range(iim):
                xx = float(ii)
                i, j, di, dj = pix(xx, yy, coef)
                # NB cosi : le Fortran indexe ima(i,j) en 1-based (i,j venant de
                # PIX sont des entiers ≥1). En numpy (0-based) il faut donc
                # décaler de -1 : ima[i-1, j-1]. (Validé : paraît ME~4.6 sur la
                # ligne de référence du log Fortran.)
                ia = max(i - 1, 0)
                ja = max(j - 1, 0)
                # bilinéaire sur ima (ms2.f : cymx = (ima(i,j)*(1-di)+...)*...
                ib = min(ia + 1, ima.shape[0] - 1)
                jb = min(ja + 1, ima.shape[1] - 1)
                v00 = float(ima[ia, ja]); v10 = float(ima[ib, ja])
                v01 = float(ima[ia, jb]); v11 = float(ima[ib, jb])
                cymx[ii, jj, n] = ((v00 * (1 - di) + v10 * di) * (1 - dj)
                                   + (v01 * (1 - di) + v11 * di) * dj)
    return cymx