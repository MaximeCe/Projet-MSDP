"""Étape 3b — Calibration des canaux MSDP (portage de ``ms2.f`` `calib` & cie).

À partir des canaux extraits ``cymx(i,j,n)``, le contrôleur ``calib`` :

1. **cliss** : version lissée de cymx (moyenne glissante sur
   ``(2*iliss+1)×(2*jliss+1)`` autour de chaque pixel, zone ``il1..il2``).
2. **transpec** : translation en longueur d'onde entre canaux (Ts) estimée à
   partir de la géométrie (``xr/yr``).
3. **linecurv / plot_line** : position du centre de raie ``center(i,n)`` et de
   sa pente ``pte(n)`` (moindres carrés DPMCAR sur la courbe lissée).
4. **profmean** : profil moyen ``profmm`` (moyenne des canaux alignés en j).
5. **cal(i,j,n)** : normalisation d'intensité ``cymx(i,jp,n)/profmm(kp)`` —
   la carte de calibration photométrique.

La visualisation (calib.ps, cal.ps) est isolée dans ``plotting.py``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Moindres carrés (DPMCAR, ms1.f) — petit util polynomial
# --------------------------------------------------------------------------- #
def dpcar(x: np.ndarray, y: np.ndarray, p: np.ndarray, nt: int) -> np.ndarray:
    """Polynôme NT termes par moindres carrés (port de ``ms1.f`` DPMCAR).

    ``x`` est mis à l'échelle par ``/1000``. Retourne les ``nt`` coefficients
    (coef(1) = ordre 0 ; les suivants sont divisés par 1000``**(i-1)``).

    Parameters
    ----------
    x, y : np.ndarray
        Données (taille ND).
    p : np.ndarray
        Poids (taille ND).
    nt : int
        Nombre de termes du polynôme (≤10).

    Returns
    -------
    np.ndarray
        Coefficients, shape ``(nt,)``.
    """
    nt = min(int(nt), 10)
    ntp = nt + 1
    nd = len(x)
    c = np.zeros((ntp + 1, ntp))       # on indexe C(m,k) ; m 1..nt+1
    f = np.zeros(nt)
    for l in range(nd):
        f[0] = 1.0
        for i in range(1, nt):
            f[i] = f[i - 1] * x[l] / 1000.0
        for m in range(nt):
            for k in range(m, nt):
                c[k, m] += f[k] * f[m] * p[l]
            c[nt, m] += f[m] * p[l] * y[l]
    # symétriser (hors diagonale supérieure) — en fait C(M,K)=C(K,M)
    for m in range(nt + 1):
        for k in range(m, nt + 1):
            if k >= m:
                c[m, k] = c[k, m]
    coef = np.zeros(nt)
    # élimination de Gauss (décompose le système normal)
    cc = c.copy()
    for i in range(nt):
        piv = cc[i, i]
        if piv == 0:
            continue
        for m in range(nt):
            cc[m + 1, i] /= piv
        for k in range(nt):
            if i == k:
                continue
            piv2 = cc[i, k]
            for m in range(nt):
                cc[m + 1, k] -= cc[m + 1, i] * piv2
    coef[0] = cc[nt, 0]
    for i in range(1, nt):
        coef[i] = cc[nt, i] / (1000.0 ** i)
    return coef


def parafit(y: np.ndarray, i1: int, i2: int, l: int, z: np.ndarray) -> np.ndarray:
    """Lissage parabolique de Savitzky (port de ms2.f `parafit`).

    ``z[i]`` lissé sur une fenêtre ``2L+1`` ; cas L=0 (copie), L>=10000
    (moyenne), sinon parabole d'ajustement locale.

    Parameters
    ----------
    y : np.ndarray (indexable)
    i1, i2, l : int
    z : np.ndarray (réceptacle)

    Returns
    -------
    np.ndarray
        ``z`` (modifié en place).
    """
    if l == 0:
        z[i1:i2 + 1] = y[i1:i2 + 1]
        return z
    if l >= 10000:
        z[i1:i2 + 1] = y[i1:i2 + 1].mean()
        return z
    n = np.arange(1, l + 1)
    sx2 = 2 * np.sum(n ** 2)
    sx4 = 2 * np.sum(n ** 4)
    d = (2 * l + 1) * sx4 - sx2 ** 2
    ia = i1 + l
    ib = i2 - l
    if ib < ia:
        return z
    for i in range(ia, ib + 1):
        il1, il2 = i - l, i + l
        di = np.arange(il1 - i, il2 - i + 1)
        sy = float(np.sum(y[il1:il2 + 1]))
        syx = float(np.sum(y[il1:il2 + 1] * di))
        syx2 = float(np.sum(y[il1:il2 + 1] * di * di))
        a = (sy * sx4 - syx2 * sx2) / d
        z[i] = a
        if i == ia or i == ib:
            b = syx / sx2
            c = ((2 * l + 1) * syx2 - sy * sx2) / d
        if i == ia:
            for ip in range(i1, ia):
                zd = ip - ia
                z[ip] = a + zd * (b + zd * c)
        if i == ib:
            for ip in range(ib, i2 + 1):
                zd = ip - ib
                z[ip] = a + zd * (b + zd * c)
    return z


# --------------------------------------------------------------------------- #
# Lissage cliss
# --------------------------------------------------------------------------- #
def smooth_cliss(cymx: np.ndarray, margline: int, iliss: int, jliss: int
                 ) -> np.ndarray:
    """Version lissée de cymx (ms2.f `calib`).

    ``cliss = cymx`` hors de la zone, puis moyenne glissante dans
    ``il1..il2 × jl1..jl2`` (sur ``(2*iliss+1)×(2*jliss+1)``).

    Parameters
    ----------
    cymx : np.ndarray ``(iim, jjm, nm)``
    margline, iliss, jliss : int

    Returns
    -------
    np.ndarray
        ``cliss`` même shape.
    """
    imd = cymx.shape[0]
    jmd = cymx.shape[1]
    nm = cymx.shape[2]
    cliss = cymx.copy()
    # bornes 0-based d'après ms2.f (bornes Fortran 1-based : i1=1+margline,
    # i2=imd-margline, il1=i1+iliss, il2=i2-iliss ; de même en j).
    il1 = margline + iliss                    # 0-based (Fortran i1-1 + iliss)
    il2 = imd - margline - iliss - 1          # 0-based (Fortran i2-1 - iliss)
    jl1 = margline + jliss
    jl2 = jmd - margline - jliss - 1
    win_i = 2 * iliss + 1
    win_j = 2 * jliss + 1
    den = win_i * win_j
    for n in range(nm):
        sub = cymx[:, :, n]
        for j in range(jl1, jl2 + 1):
            for i in range(il1, il2 + 1):
                piv = float(np.sum(sub[i - iliss:i + iliss + 1,
                                       j - jliss:j + jliss + 1]))
                cliss[i, j, n] = piv / den
    return cliss


# --------------------------------------------------------------------------- #
# transpec (ms2.f)
# --------------------------------------------------------------------------- #
def transpec(xr: np.ndarray, yr: np.ndarray, ntrans: int, jmd: int,
             mupris: int, mustep: int) -> float:
    """Translation Ts entre canaux (ms2.f `transpec`).

    ``trj = (jmd-1)*mustep*vecjb / (mupris*chajb)`` où chajb/vecjb sont des
    combinaisons des coordonnées yr du canal de référence.

    Parameters
    ----------
    xr, yr : np.ndarray ``(nm,3,2)`` ; ntrans : canal de référence (1-based)
    jmd : dim Y canaux ; mupris, mustep : int

    Returns
    -------
    float
        ``trj``.
    """
    nm = yr.shape[0]
    nc = ntrans - 1            # 0-based
    chajb = 0.5 * (yr[nc, 0, 1] - yr[nc, 0, 0] + yr[nc, 2, 1] - yr[nc, 2, 0])
    nm1 = nc + 1 if nc + 1 < nm else nc
    nm2 = nc - 1 if nc - 1 >= 0 else nc
    vecjb = 0.125 * (yr[nm1, 0, 0] + yr[nm1, 2, 0] + yr[nm1, 0, 1] + yr[nm1, 2, 1]
                     - yr[nm2, 0, 0] - yr[nm2, 2, 0] - yr[nm2, 0, 1] - yr[nm2, 2, 1])
    if chajb == 0:
        return 0.0
    return float(jmd - 1) * mustep * vecjb / (float(mupris) * chajb)


# --------------------------------------------------------------------------- #
# trans (ms2.f) — meilleure translation jtr par moindres
# --------------------------------------------------------------------------- #
def trans_best_jtr(cliss: np.ndarray, jtrprox: int) -> Tuple[int, float]:
    """Recherche la translation jtr optimale (ms2.f `trans`).

    Essaie ``jtrans1=jtrprox-10 .. jtrprox+10``, calcule pour chacun un
    facteur de cohérence entre canaux et retient celui qui minimise le
    max|co-1|.

    Parameters
    ----------
    cliss : np.ndarray ``(imd, jmd, nm)`` (lissé)
    jtrprox : int

    Returns
    -------
    (jtrcalc, devcalc)
    """
    imd, jmd, nm = cliss.shape
    ic = (1 + imd) // 2 - 1
    jc = 1 + jmd // 2
    nc = (1 + nm) // 2 - 1
    jt1 = jtrprox - 10
    jt2 = jtrprox + 10
    best_jtr, best_dev = jtrprox, None
    for jt in range(jt1, jt2 + 1):
        jt1c = jc - jt // 2
        jt2c = jt1c + jt
        jk1 = jmd - (jt1c - 1)
        jk2 = jmd - (jt2c - 1)
        if jk1 < 0 or jk2 >= jmd:
            continue
        co = np.ones(nm)
        cl1 = np.array([cliss[ic, jk1, n] for n in range(nm)])
        cl2 = np.array([cliss[ic, jk2, n] for n in range(nm)])
        for n in range(1, nm):
            co[n] = co[n - 1] * cl2[n - 1] / cl1[n]
        co = co / co[nc]
        devm = float(np.max(np.abs(co - 1.0)))
        if best_dev is None or devm < best_dev:
            best_dev = devm
            best_jtr = jt
    return best_jtr, (best_dev if best_dev is not None else 1.0)


# --------------------------------------------------------------------------- #
# linecurv (ms2.f) — centre de raie + pente
# --------------------------------------------------------------------------- #
def linecurv_centers(cymx: np.ndarray, cliss: np.ndarray, n: int,
                     il1: int, il2: int, jl1: int, jl2: int,
                     jparab: int) -> Tuple[np.ndarray, float]:
    """Centres de raie ``center(:,n)`` et pente ``pte(n)`` (ms2.f `linecurv`).

    Returns
    -------
    center : np.ndarray len(imd)
    pte : float
    """
    yln, center, pte = linecurv_curve(cymx, cliss, n, il1, il2, jl1, jl2, jparab)
    return center, pte


def linecurv_curve(cymx: np.ndarray, cliss: np.ndarray, n: int,
                   il1: int, il2: int, jl1: int, jl2: int,
                   jparab: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Centre de raie complet (ms2.f `linecurv`, branche leastsq=1).

    Returns
    -------
    yln : np.ndarray len(imd+1)      positions de raie brutes (1-based)
    center : np.ndarray len(imd)     fit linéaire
    pte : float                      pente coef² (slope line/i)
    """
    imd = cliss.shape[0]
    jmd = cliss.shape[1]
    il1c, il2c = il1, il2
    jl1c = jl1 + jparab
    jl2c = jl2 - jparab
    if il2c > imd:
        il2c = imd
    if jl2c > jmd:
        jl2c = jmd
    yln = np.zeros(imd + 1)          # 1-based (yln(i), i=1..imd)
    for i in range(il1c, il2c + 1):
        seg = cliss[i - 1, jl1c - 1:jl2c, n]           # j 1-based jl1c..jl2c
        piv = float(np.min(seg))
        jinds = np.where(seg == piv)[0]
        japp_i = jl1c + (int(jinds[-1]) if len(jinds) else 0)   # 1-based approx
        ndpar = 1 + 2 * jparab
        xs = np.zeros(ndpar); ys = np.zeros(ndpar)
        pds = np.ones(ndpar)
        for nd in range(ndpar):
            japp = japp_i - jparab + nd                 # 1-based
            xs[nd] = japp
            if 0 < japp <= jmd:
                ys[nd] = cliss[i - 1, japp - 1, n]
        coef = dpcar(xs, ys, pds, 3)
        if coef[2] != 0:
            yln[i] = -coef[1] / (2.0 * coef[2])
        else:
            yln[i] = japp_i
    ipar = il2c - il1c + 1
    xs = np.arange(il1c, il2c + 1, dtype=float)
    ys = yln[il1c:il2c + 1]
    pds = np.ones(ipar)
    coef = dpcar(xs, ys, pds, 2)
    center = coef[0] + np.arange(1, imd + 1) * coef[1]
    return yln, center, coef[1]
def _build_profmm(cliss: np.ndarray, jt1: int, jt2: int, jtr: int, nm: int,
                  nr: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Profil moyen ``profmm`` et composantes (ms2.f `profmean`).

    Calcule, pour la ligne centrale ``ic`` de ``cliss`` :
    - ``profm(j,n) = cliss(ic, j, n)`` ;
    - le facteur de normalisation ``coeff(n)`` (ms2.f l.2612-2622) ;
    - le profil non calibré ``pronoc`` et le profil moyen ``profmm`` par
      concaténation des canaux alignés (segments j1..j2, ms2.f l.2651-2694).

    Returns
    -------
    (pronoc, profmm, coeff, km) — pronoc/profmm de longueur ``km``.
    """
    imd = cliss.shape[0]
    jmd = cliss.shape[1]
    ic = imd // 2
    profm = cliss[ic, :, :]                     # (jmd, nm)
    # ms2.f : profm est indexé 1..jmd (1-based) ; on convertit en 0-based ici.
    # jt1, jt2 sont fournis 1-based.  profm(j0,n) = cliss(ic, j, n).
    # coeff : ms2.f l.2612-2622 (profm(jt2,n)/profm(jt1,n-1)), normalisé coeff0
    coeff = np.ones(nm)
    for n in range(1, nm):
        coeff[n] = coeff[n - 1] * profm[jt2 - 1, n] / profm[jt1 - 1, n - 1]
    coeff = coeff / coeff[nr - 1] if nr else coeff / coeff[0]  # /coeff0=coeff(nr)

    pronoc = [0.0] * (8 * jmd)
    profmm = [0.0] * (8 * jmd)
    # canal 1 : l.2652-2664  j1=jmd, j= jmd..jt1
    k1, k2 = 1, jmd - jt1 + 1
    k = k1
    for j in range(jmd, jt1 - 1, -1):           # j1..j2 = jmd..jt1 (1-based)
        pronoc[k - 1] = profm[j - 1, 0]
        profmm[k - 1] = pronoc[k - 1] / coeff[0]
        k += 1
    k3 = k2
    # canaux intérieurs : l.2666-2678  j1=jt2, j= jt2..jt1, k2=k1+(jt2-jt1)
    for n in range(1, nm - 1):
        k1 = k3
        k2 = k1 + (jt2 - jt1)
        k = k1
        for j in range(jt2, jt1 - 1, -1):
            pronoc[k - 1] = profm[j - 1, n]
            profmm[k - 1] = pronoc[k - 1] / coeff[n]
            k += 1
        k3 = k2
    # dernier canal : l.2680-2693  j1=jt2, j= jt2..1, k2=k1+(jt2-1)
    n = nm - 1
    k1 = k3
    k2 = k1 + (jt2 - 1)
    k = k1
    for j in range(jt2, 0, -1):
        pronoc[k - 1] = profm[j - 1, n]
        profmm[k - 1] = pronoc[k - 1] / coeff[n]
        k += 1
    km = k2
    return pronoc[:km], profmm[:km], coeff, km


def profmean(cliss: np.ndarray, center: np.ndarray, pte: np.ndarray,
             jt1: int, jt2: int, jtr: int, nm: int,
             nr: int = 0) -> Tuple[np.ndarray, int]:
    """Profil moyen ``profmm`` (ms2.f `profmean`), API simplifiée.

    Returns
    -------
    profmm : np.ndarray (km)
    km : int
    """
    pronoc, profmm, coeff, km = _build_profmm(cliss, jt1, jt2, jtr, nm, nr)
    return profmm, km


# --------------------------------------------------------------------------- #
# Calibration finale (ms2.f calib) — cal(i,j,n)
# --------------------------------------------------------------------------- #
def calibrate(cymx: np.ndarray, cliss: np.ndarray, profmm: np.ndarray,
              center: np.ndarray, pte: np.ndarray,
              jtr: int, jt1: int, jt2: int, km: int) -> np.ndarray:
    """Normalisation d'intensité (ms2.f `calib`).

    ``cal(i,j,n) = cymx(i, jp, n) / profmm(kp)`` avec
    ``jp = j + jdel``, ``jdel = pte(ncurv)*(i - ic)``,
    ``k = 1 + jmd - jp + (n-1)*jtr`` clampé à ``[1, km]``.

    Parameters
    ----------
    cymx, cliss : np.ndarray
    profmm : np.ndarray (km), center/pte pour la pente
    jtr, jt1, jt2, km : int

    Returns
    -------
    cal : np.ndarray (iim, jjm, nm)
    """
    imd, jmd, nm = cymx.shape
    ic = (1 + imd) // 2
    # ms2.f calib : jdel = pte(ncurv)*(i-ic) ; ncurv est le canal de référence.
    # pte peut être transmis scalaire (float) ou array ; on normalise en float.
    pte_n = float(pte) if np.ndim(pte) == 0 else float(pte[0])
    cal = np.empty_like(cymx)
    for n in range(nm):
        for i in range(imd):
            jdel = pte_n * (i + 1 - ic)
            for j in range(jmd):
                jp = j + 1 + jdel          # 1-based j + jdel que ms2.f indexe
                k = 1 + jmd - jp + n * jtr
                kp = int(k)
                if kp < 1:
                    kp = 1
                if kp > km:
                    kp = km
                jpi = int(min(max(jp - 1, 0), jmd - 1))
                val = cymx[i, jpi, n]
                if val < 1.0:
                    val = 1.0
                cal[i, j, n] = val / max(profmm[kp - 1], 1.0)
    return cal