"""Étape 1 — Moyennes dark / flat (portage de ``ms1.f``, branche ``new/``).

Ce module reproduit le programme principal de ``ms1.f`` (apart les sous-routines
I/O pures) :
  1. lister les darks  ``m*x1.fit`` et les flats ``m*y1.fit`` (ordre lexico
     — le Fortran fait ``ls m*x1.fit`` / ``ls m*y1.fit``),
  2. pour chaque fichier : lecture FITS, permutation CCD → champ ``(isp, jsp)``,
  3. accumulation et moyenne (``denom = nfb - nfa + 1``),
  4. écriture du fichier binaire "moyen" (<=> ``x170330_..._00000`` / ``y...``),
     format : entête 512 int32 + données int16.

Chaque fichier moyen est la sortie de l'étape 1 et l'entrée de l'étape 2
(géométrie). Le nom du fichier moyen est dérivé du dernier fichier de série
(comme : ms1.f construit ``xname``/``yname`` depuis ``file(nfb)``).

⚠️ Endianness (leçon Bug 1 du portage) : astropy délivre déjà les FITS
big-endian ``>i2`` en ordre natif. On NE réapplique PAS de ``byteswap`` — le
``swap`` du Fortran n'existe que parce qu'il lit les octets bruts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits

from msdp.config import Config

# motifs des fichiers d'entrée par type de séquence (darks = x, flats = y)
_FILE_PATTERNS: dict[str, str] = {"dark": "*x1.fit", "flat": "*y1.fit"}
# préfixe du nom du fichier binaire moyen par type
_AVG_PREFIX: dict[str, str] = {"dark": "x", "flat": "y"}


@dataclass
class AveragedImage:
    """Résultat de l'étape 1 pour une séquence (dark ou flat).

    Attributes
    ----------
    file_type : str
        ``"dark"`` ou ``"flat"``.
    data : np.ndarray
        Image moyenne **permutée**, de forme ``(isp, jsp)``, dtype int16
        (valeurs = ``round(moyenne)`` commes ms1.f : ``+0.5`` puis tronc.).
    filename : str
        Nom du fichier binaire moyen (``x..._00000`` / ``y..._00000``).
    header : np.ndarray
        Entête 512 int32 écrit en tête du fichier moyen (format, dims).
    nfiles : int
        Nombre de fichiers réellement sommés (``nfb - nfa + 1``).
    """

    file_type: str
    data: np.ndarray
    filename: str
    header: np.ndarray
    nfiles: int


def _read_fits_native(path: str | Path) -> np.ndarray:
    """Lit un FITS et renvoie ses données en int16 natif (ordre hôte).

    Ne PAS byteswapp : astropy normalise déjà l'endianness. La permutation
    (axe) est traitée plus tard par :func:`_permute`.

    Parameters
    ----------
    path : str | Path
        Chemin du fichier ``.fit``.

    Returns
    -------
    np.ndarray
        Données, forme ``(NAXIS2 rows, NAXIS1 cols)`` commes livrées (row-major).
    """
    with fits.open(path) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"Pas de données dans {path}")
    arr = np.asarray(data)
    if arr.dtype != np.int16:
        arr = arr.astype(np.int16)
    return arr


def _permute(data: np.ndarray, isz: int, jsz: int) -> np.ndarray:
    """Permute CCD → champ de traitement (commes ``ms1.f`` readfits main).

    Le Fortran remplit ``tabpermu(ip, jp)`` avec ``ip = js+1-j`` (renversement
    des lignes) et ``jp = i`` (colonnes inchangées) — soit une simple
    **inversion de l'axe des lignes**. ``astropy`` livre ``data[row, col]``
    (lignes = dimension du CCD Y), et ``permuted`` a la même forme ``(js, is)``
    : ``permuted[js-1-j, i] = data[j, i]``  ⇔  ``permuted = data[::-1]``.

    Cette formulation est exactement équivalente à la boucle double validée de
    ``ms1.py``/``ms2.py`` (parité numérique conservée), sans risque d'erreur
    d'indexation colonne-major.

    Parameters
    ----------
    data : np.ndarray
        Données ``(rows=js, cols=is)`` livrées par astropy.
    isz : int
        ``is`` (dimension X CCD, nombre de colonnes).
    jsz : int
        ``js`` (dimension Y CCD, nombre de lignes).

    Returns
    -------
    np.ndarray
        Image permutée ``(js, is)`` = ``(isp, jsp)``, int64 (accumulateur).
    """
    del isz, jsz  # forme inchangée : seules les lignes sont retournées
    return data[::-1].astype(np.int64)


def list_sequence_files(directory: str | Path, file_type: str) -> list[str]:
    """Liste les fichiers d'une séquence (``m*x1.fit`` pour dark, ``m*y1.fit`` flat).

    L'ordre est lexicographique (important : le Fortran ``ls`` trie et utilise
    ``nfa..nfb`` comme indices de ce tri).

    Parameters
    ----------
    directory : str | Path
        Répertoire des données (``data/input/``).
    file_type : str
        ``"dark"`` ou ``"flat"``.

    Returns
    -------
    list[str]
        Noms de fichiers triés, à utiliser avec ``nfa``/``nfb`` (1-based).
    """
    pattern = _FILE_PATTERNS[file_type]
    d = Path(directory)
    return sorted(p.name for p in d.glob(pattern))


def _derive_avg_filename(files: Sequence[str], file_type: str) -> str:
    """Dérive le nom du fichier moyen depuis le dernier fichier de la série.

    Reproduit ms1.f (xname/yname depuis ``file(nfb)``) :
    - préfixe 1 caractère (``x`` dark, ``y`` flat) = ``file(33)``,
    - ``file(17:32)`` (date-heure, 16 caractères) complète le nom,
    - suffixe ``00000``.

    Exemple — dark ``m010_b0101_ms_20170330_09565453_x1.fit`` (dernier) :
    ``x170330_09565453_00000``.
    """
    last = files[-1]
    prefix = _AVG_PREFIX[file_type]
    # ms1.f (1-based) : xname(1:1)=file(33:33), xname(2:17)=file(17:32)
    # -> en 0-based : prefixe + file[16:32] + "00000"
    name = prefix + last[16:32] + "00000"
    return name


def average_sequence(
    directory: str | Path,
    file_type: str,
    nfa: int,
    nfb: int,
    config: Config,
) -> AveragedImage:
    """Moyenne une séquence dark/flat et renvoie l'image moyenne permutée.

    Parameters
    ----------
    directory : str | Path
        Répertoire des données FITS.
    file_type : str
        ``"dark"`` ou ``"flat"``.
    nfa, nfb : int
        Premier / dernier indice (1-based) dans la liste triée de fichiers.
    config : Config
        Configuration (``is``, ``js``, ``ipermu``).

    Returns
    -------
    AveragedImage
        L'image moyenne (permutée), son nom et son entête.
    """
    files = list_sequence_files(directory, file_type)
    if nfb > len(files):
        nfb_eff = len(files)
    else:
        nfb_eff = nfb
    if nfa > nfb_eff:
        raise ValueError(
            f"Séquence {file_type} invalide : nfa={nfa} > nfb={nfb_eff} "
            f"(disponibles {len(files)})"
        )

    isz = config.ccd_x   # 1536 (col)
    jsz = config.ccd_y   # 1024 (row)
    # dimensions champ permuté
    isp, jsp = (jsz, isz)

    accumulator = np.zeros((isp, jsp), dtype=np.int64)
    for idx in range(nfa - 1, nfb_eff):
        fname = files[idx]
        data = _read_fits_native(Path(directory) / fname)
        permuted = _permute(data, isz, jsz)
        accumulator += permuted

    denom = float(nfb_eff - nfa + 1)
    # ms1.f : tab2(ip,jp) = float(tabaver)/denom + 0.5  (arrondi)
    averaged = (accumulator / denom + 0.5).astype(np.int16)

    # entête : 512 int32, head(1)=3, head(2)=isp, head(3)=jsp, head(4)=1
    header = np.zeros(512, dtype=np.int32)
    header[:4] = [3, isp, jsp, 1]

    filename = _derive_avg_filename(files[:nfb_eff], file_type)
    return AveragedImage(
        file_type=file_type,
        data=averaged,
        filename=filename,
        header=header,
        nfiles=nfb_eff - nfa + 1,
    )


def write_averaged_file(img: AveragedImage, output_dir: str | Path) -> Path:
    """Ecrit le fichier binaire moyen (entete 512 int32 + donnees int16).

    Format (auto-consistant Python) : header 512 x int32 (head[0]=3,
    head[1]=isp, head[2]=jsp, head[3]=1), puis data isp x jsp int16
    ecrits en ordre colonne-major (F) — meme layout que le write du Fortran
    (do jp=1,jsp : write(tab2(i,jp), i=1,isp)). On ne vise pas la parite
    octet-a-octet avec le binaire Fortran ( non-formate, avec marqueurs de
    record) — uniquement la parite numerique au niveau ACDF2/miv. Le lecteur
    (step2) lit la meme convention, donc l'orientation est coherente en interne.
    """
    out = Path(output_dir) / img.filename
    with open(out, "wb") as fh:
        img.header.tofile(fh)
        # ordre F (colonne-major) : ip (index X) varie le plus vite
        data16 = img.data.astype(np.int16)
        fh.write(data16.tobytes(order="F"))
    return out


def run_step1(
    directory: str | Path,
    config: Config,
    output_dir: str | Path | None = None,
) -> dict[str, AveragedImage]:
    """Exécute l'étape 1 complète : moyennes dark et flat.

    Parameters
    ----------
    directory : str | Path
        Répertoire des FITS d'entrée.
    config : Config
        Configuration (nfa/nfb extraits de ``nfx1..nfy2``).
    output_dir : str | Path | None
        Répertoire où écrire les binaires (défaut : ``directory``).

    Returns
    -------
    dict[str, AveragedImage]
        ``{"dark": ..., "flat": ...}`` (clés absentes si la séquence est vide).
    """
    out_dir = Path(output_dir) if output_dir else Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, AveragedImage] = {}

    for ftype, (n1, n2) in {
        "dark": (config.get("nfx1", 1), config.get("nfx2", 0)),
        "flat": (config.get("nfy1", 1), config.get("nfy2", 0)),
    }.items():
        if not n2 or n2 <= 0:
            continue
        img = average_sequence(directory, ftype, n1, n2, config)
        out = write_averaged_file(img, out_dir)
        result[ftype] = img
        print(f"  ✓ {ftype}: {out.name}  ({out.stat().st_size} o)")

    return result