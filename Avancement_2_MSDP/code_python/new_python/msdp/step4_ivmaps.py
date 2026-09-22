"""Étape 4b — Profils I/V + vitesses (portage de ms3.f `ivmap3`/`intvel3`
et ms4.f `ivmap4`/`intvel4`).

À partir du cube ``sobstot(i,j,n,nfb)`` (observations calibrées) et de la carte
de calibration ``cal``, on :
- **ivmap3** (degré 3) : pour chaque ligne ``j``, interpolation DPMCAR deg4 sur
  4 canaux → ``profnf(iic,j,nf,1)`` (81 nouveaux lambdas entre canaux) ;
- **ivmap4** (degré 4) : interpolation deg5 sur 5 canaux → ``profnf(.,.,2)``,
  et ``profnf(.,.,3)`` = profil mixte degré 3 ;
- **intvel3/intvel4** : mesure du **bissecteur de vitesse** (droite passant par
  le demi-point des différences) → coordonnées ``(xv2, yv2)`` écrites dans
  ``miv.lis`` (km/s).

La visualisation (ivprof1/2/3.ps) est isolée dans ``plotting.py``. Les vitesses
ne sont calculées que si ``lbdvel* > 0``.

⚠️ Piège documenté : ``intvel3``/``intvel4`` (lbdvel>0) ont causé des segfaults
historiques ; ms3.f a été corrigé (nla→nla1). On garde le calcul sans émulation
du bug.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from msdp.step3_calib import dpcar


# --------------------------------------------------------------------------- #
# Bissecteur de vitesse (intvel3 — ms3.f)
# --------------------------------------------------------------------------- #
def intvel3(profil: np.ndarray, lbdvel: int, nlm: int) -> Tuple[float, float]:
    """Bissecteur de vitesse sur un profil (port de ms3.f `intvel3`).

    Reproduit ms3.f : copie ``pro(nl)=yy(nl)`` (indices 1-based), puis cherche
    le 1er ``nl`` où ``diff(nla1)*diff(nlb1) <= 0`` (changement de signe entre
    2 segments de largeur ``lbdvel``), puis calcule le point d'intersection.

    Parameters
    ----------
    profil : np.ndarray
        Profil de raie (valeurs arbitraires).
    lbdvel : int
        Largeur (en pas) de la mesure (>0).
    nlm : int
        Longueur utile du profil.

    Returns
    -------
    (xv2, yv2) : coordonnées du point bissecteur (NaN si introuvable).
    """
    # tableau 1-based comme le Fortran
    pro = np.zeros(nlm + 2)
    n = min(nlm, len(profil))
    pro[1:n + 1] = np.asarray(profil)[:n]
    nlmb = nlm - lbdvel
    if nlmb < 1:
        return float("nan"), float("nan")
    diff_arr = np.zeros(nlm + lbdvel + 2)

    nla1 = nlb1 = None
    for nl in range(1, nlmb + 1):
        nla2 = nl + 1
        nlb1v = nl + lbdvel
        nlb2 = nl + 1 + lbdvel
        diff_arr[nl] = pro[nlb1v] - pro[nl]
        diff_arr[nlb1v] = pro[nlb2] - pro[nla2]
        if diff_arr[nl] * diff_arr[nlb1v] <= 0.0:
            nla1 = nl
            nlb1 = nlb1v
            break
    if nla1 is None:
        return float("nan"), float("nan")

    da = diff_arr[nla1]
    db = diff_arr[nlb1]
    if (db - da) == 0.0:
        return float("nan"), float("nan")
    dn = -da / (db - da)
    xl = nla1 + dn + lbdvel / 2.0 - 1.0
    yl = pro[nla1] + dn * (pro[nlb1] - pro[nla1])
    return xl, yl


# --------------------------------------------------------------------------- #
# ivmap3 — interpolation degré 3 (ms3.f)
# --------------------------------------------------------------------------- #
def ivmap3(sobstot: np.ndarray, cal: np.ndarray, iim: int, jjm: int,
           nm: int, lbdvels: Tuple[int, int, int]) -> Tuple[np.ndarray, list]:
    """Interpolation degré 3 des profils (ms3.f `ivmap3`).

    Parameters
    ----------
    sobstot : np.ndarray ``(iim, jjm, nm, nfb)`` (nfb1=1)
    cal : np.ndarray ``(iim, jjm, nm)`` (référence ; non modifiée ici)
    iim, jjm, nm : int
    lbdvels : (lbdvel1, lbdvel2, lbdvel3)

    Returns
    -------
    (profnf, velocités)
        - profnf : ``(250?, )`` réindexé voir note (on retourne le cube j×nf).
        - liste de (lbdvel, xv2, yv2) pour les vitesses>0.
    """
    iic = (iim + 1) // 2
    # profnf(iic, j, nf, ideg) — on dimensionne pour nf=1..( (nm-1)*10+1 )
    nfm = (nm - 1) * 10 + 1
    profnf = np.zeros((jjm, nfm, 3))
    p = np.ones(4)
    vels3 = []

    for j in range(jjm):
        for n in range(2, nm - 1):  # Fortran n=2..nm-2
            xs, ys = np.zeros(4), np.zeros(4)
            for k in range(4):
                # Fortran : np=k+n-2 (1-based canal), x=10*(np-1)=10*(k+n-3)
                np_idx = k + n - 2            # 0-based canal échantillonné (= canal np_fort-1)
                # np_fort=k_fort+n-2 ; k_fort(1..4)=k+1 => np_fort=k+n-1
                xs[k] = 10.0 * (k + n - 2)   # =10*(np_fort-1)
                ys[k] = sobstot[iic - 1, j, np_idx, 0]
            coef = dpcar(xs, ys, p, 4)
            # bornes d'interpolation entre canaux (valeurs d'abscisse x)
            # Fortran : n==2 ; n>2 & n<nm-2 ; n==nm-2 (l'ordre compte)
            if n == 2:
                nd1, nd2 = 0, 20
            elif n == nm - 2:
                nd1, nd2 = (nm - 3) * 10, (nm - 3) * 10 + 20
            else:
                nd1, nd2 = 10 * (n - 1), 10 * (n - 1) + 10
            # nf(1-based)=nd+1 ; en 0-based profnf[:,nd] (nd=0..80)
            for nd in range(nd1, min(nd2, nfm - 1) + 1):
                x = float(nd)
                if 0 <= nd < nfm:
                    profnf[j, nd, 0] = (coef[0] + x * (coef[1] + x *
                                        (coef[2] + x * coef[3])))
    # vitesses sur 3 profils de référence (jj=21,61,101) si lbdvel>0
    for jj in (21, 61, 101):
        jj_idx = jj - 1
        if jj_idx >= jjm:
            continue
        yy = profnf[jj_idx, :, 0]
        for lv in lbdvels:
            if lv and lv < len(yy):
                xv, yv = intvel3(yy, lv, nfm)
                if not np.isnan(xv):
                    vels3.append((lv, xv, yv))
    return profnf, vels3


# --------------------------------------------------------------------------- #
# ivmap4 — interpolation degré 4 (ms4.f)
# --------------------------------------------------------------------------- #
def ivmap4(sobstot: np.ndarray, cal: np.ndarray, iim: int, jjm: int,
           nm: int, profnf3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolation degré 4 (ms4.f `ivmap4`).

    Recalcule les profils sur 5 canaux (DPMCAR deg5) et produit :
    - ``profnf(.,.,2)`` : degré 4,
    - ``profnf(.,.,3)`` : mixte (début = copie du degré 3, complété).

    Parameters
    ----------
    sobstot : ``(iim, jjm, nm, nfb)``
    profnf3 : résultat degré 3 (``(jjm, nfm, 3)``)

    Returns
    -------
    (profnf, pro5, ...) — profnf ``(jjm, nfm, 3)`` complété,
    centres ``pro`` et densité.
    """
    iic = (iim + 1) // 2
    nfm = (nm - 1) * 10 + 1
    profnf = np.array(profnf3, copy=True)   # .3 = deg3 au départ (mix poi)
    ii = iic - 1
    # bornes d'itération : j=21,61,101 ; n0=2,3,4
    pro = np.zeros((nm, nfm))
    for (jj_idx, n0) in zip([20, 60, 100], [2, 3, 4]):
        if jj_idx >= jjm:
            continue
        ntm = 5
        xs, ys = np.zeros(ntm), np.zeros(ntm)
        p = np.ones(ntm)
        for k in range(ntm):
            ny = k + n0 - 1              # canal (1-based) : n0..n0+4
            xs[k] = float(k + 1)
            ys[k] = sobstot[ii, jj_idx, ny - 1, 0]
        coef = dpcar(xs, ys, p, ntm)
        nff1 = (n0 - 1) * 10 + 1          # 11,21,31
        nff2 = nff1 + 40                  # 51,61,71
        nff1i, nff2i = nff1 - 1, nff2    # 0-based
        for nff in range(nff1i, min(nff2i, nfm)):
            x = 1.0 + float((nff + 1) - nff1) / 10.0
            val = (coef[0] + x * (coef[1] + x * (coef[2] + x *
                   (coef[3] + x * coef[4]))))
            pro[n0 - 1, nff] = val
            profnf[jj_idx, nff, 1] = val   # deg 4
            profnf[jj_idx, nff, 2] = val   # mix (début copie deg3 surplombée)
    return profnf, pro


# --------------------------------------------------------------------------- #
# miv.lis — assemblage des vitesses
# --------------------------------------------------------------------------- #
def format_miv(vels: list) -> str:
    """Formate les vitesses en lignes miv.lis (commes ms3/ms4)."""
    lines = []
    for lv, xv, yv in vels:
        lines.append(f"       3      {lv}   {xv:.2f}  {yv:.2f}")
    return "\n".join(lines)