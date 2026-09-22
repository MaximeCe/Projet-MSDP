"""Étape 4a — Observations solaires (portage de ms2.f `solarobs`/`mdark`/`calobs`).

Pour chaque observation ``*b1.fit`` (nfb1..nfb2), on :
1. lit l'image brute (FITS, permutation iswap/ipermu) → ``ima`` (1024×1536),
2. aux retire le **dark moyen** (fichier binaire ``x..._00000``) → ``ima − imadark``,
3. extrait les canaux via ``channels`` (réutilise step3_channels) → ``sobs``,
4. applique la **correction de calibration** : ``sobs = sobs / cal`` (``calobs``),
5. empile dans ``sobstot(i,j,n,nf)`` (toutes les obs) et garde ``sobsplot`` (nfplot).

Le cube ``sobstot`` alimente ensuite ivmap3/ivmap4 (ms3/ms4) => étape suivante.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from msdp.config import Config
from msdp import step1_average as s1
from msdp.step2_geometry import _read_averaged_python
from msdp.step3_channels import channels_dims, extract_channels


def read_observation(path: str | Path, config: Config) -> np.ndarray:
    """Lit une observation FITS brute → image permutée ``(1024, 1536)``.

    Reproduit ``solarobs`` (iswap=1, ipermu=1) :  même transformation que
    l'étape 1 — astropy lit en natif (pas de re-byteswap), puis permutation
    ``_permute`` (= ``meanflat.T``) → layout ``(js, is)`` attendu par
    ``extract_channels``.

    Parameters
    ----------
    path : str | Path
        Fichier ``*b1.fit`` de l'observation.
    config : Config
      (``is``/``js`` pour la permutation).

    Returns
    -------
    np.ndarray
        Image permutée ``(1024, 1536)``.
    """
    data = s1._read_fits_native(path)
    return s1._permute(data, config.ccd_x, config.ccd_y)


def read_dark_averaged(path: str | Path, config: Config) -> np.ndarray:
    """Lit le dark moyen (binaire ``x..._00000``) → ``(1024, 1536)``.

    Même lecteur que l'étape 2 ; le layout retourné est ``(1024, 1536)``
    (= meanflat.T), directement soustractible à `ima`.
    """
    return _read_averaged_python(path, config.ccd_y, config.ccd_x)


def subtract_dark(ima: np.ndarray, imadark: np.ndarray) -> np.ndarray:
    """Soustraction du dark (ms2.f : ima = ima − imadark, sans clamp ici)."""
    return ima.astype(np.int32) - imadark.astype(np.int32)


def calobs(cal: np.ndarray, sobs: np.ndarray) -> np.ndarray:
    """Correction de calibration (ms2.f `calobs`) : ``sobs = sobs / cal``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(cal != 0, sobs.astype(np.float64) / cal, 0.0)
    return out


def compute_obsD(sobs: np.ndarray, cal: np.ndarray,
                 den: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Construit sobsD1 (non calibré) et sobsD2 (calibré) + normalise par ligne.

    Reproduit ms2.f calib l.2092-2105 et la normalisation l.2126-2140 :
    - ``sobsD2 = sobs`` (sobs est déjà calibré par calobs) ;
    - ``sobsD1 = sobs × den`` avec ``den = cal`` (si ``cal=0`` → ``den=0.5``) :
      on "dé-calibre" pour revenir à l'observation brute ;
    - pour chaque canal ``n`` et ligne ``jp``, normalisation :
      ``sobsD1/D2(:, jp) = 1000 × sobsD1/D2(:, jp) / Xmean(jp)`` où
      ``Xmean(jp) = mean_i sobsD(i, jp)``.

    Parameters
    ----------
    sobs : np.ndarray (iim, jjm, nm) — observation calibrée.
    cal : np.ndarray (iim, jjm, nm) — carte de calibration.
    den : float — valeur de remplacement si cal == 0.

    Returns
    -------
    (sobsD1, sobsD2) chacun ``(iim, jjm, nm)``.
    """
    imd, jmd, nm = sobs.shape
    # sobsD1 = sobs × cal (dé-calibrage : retour à l'observation brute)
    c = cal.astype(np.float64)
    c = np.where(c == 0.0, den, c)
    sobsD1 = sobs.astype(np.float64) * c
    sobsD2 = sobs.astype(np.float64).copy()

    # normalisation par ligne (ms2.f : Xmean sur ip, par jp et canal)
    for n in range(nm):
        for jp in range(jmd):
            xm1 = float(sobsD1[:, jp, n].mean()) if imd else 0.0
            xm2 = float(sobsD2[:, jp, n].mean()) if imd else 0.0
            if xm1 != 0:
                sobsD1[:, jp, n] = 1000.0 * sobsD1[:, jp, n] / xm1
            if xm2 != 0:
                sobsD2[:, jp, n] = 1000.0 * sobsD2[:, jp, n] / xm2
    return sobsD1, sobsD2


def solarobs(
    observations: Sequence[str | Path],
    dark_avg: str | Path,
    cal: np.ndarray,
    xr: np.ndarray,
    yr: np.ndarray,
    config: Config,
    nfb1: int,
    nfb2: int,
    nfplot: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Traite les observations (ms2.f `solarobs`).

    Parameters
    ----------
    observations : sequence
        Fichiers ``*b1.fit`` triés lexicographiquement (= ordre btab.lis).
    dark_avg : str | Path
        Fichier binaire du dark moyen.
    cal : np.ndarray ``(iim, jjm, nm)``
        Carte de calibration (sortie de step3_calib).
    xr, yr : np.ndarray ``(nm, 3, 2)``
        Pavage de référence (sortie newgeom).
    config : Config
    nfb1, nfb2 : int
        Indices d'observations à traiter (1-based dans la liste triée).
    nfplot : int
        Indice de l'observation de référence (1-based) pour ``sobsplot``.

    Returns
    -------
    (sobstot, sobsplot, iim, jjm)
        - sobstot : cube ``(iim, jjm, nm, n_obs)`` —
        - sobsplot : ``(iim, jjm, nm)`` pour nfplot (
          une copie si nfplot parmi traitées, sinon zéros)
        - dims canaux.
    """
    ima = read_observation(str(observations[0]), config)  # pour dims
    # hauteur/b.
    iim, jjm = channels_dims(config.get("li"), config.get("lj"), config.get("milsec"))
    nm = config.nm
    n_obs = nfb2 - nfb1 + 1

    imadark = read_dark_averaged(dark_avg, config).astype(np.int32)
    sobstot = np.zeros((iim, jjm, nm, 10))   # ms2.f sobstot(2000,200,24,10)
    sobsplot = np.zeros((iim, jjm, nm))

    for idx, nf in enumerate(range(nfb1, nfb2 + 1)):
        if nf - 1 >= len(observations):
            break
        raw = read_observation(str(observations[nf - 1]), config)
        im = raw.astype(np.int32) - imadark
        sobs_n = extract_channels(im, xr, yr, config.get("li"), config.get("lj"),
                                  config.get("milsec"), nm)
        sobs_n = calobs(cal, sobs_n)
        sobstot[:, :, :, idx] = sobs_n
        if nf == nfplot:
            sobsplot = sobs_n.copy()

    return sobstot, sobsplot, iim, jjm