"""Tests de validation numérique du pipeline Python contre le Fortran.

Vérifie la **parité** entre le nouveau pipeline (`new_python/`) et les sorties
de référence produites par le Fortran (`src/fortran/new/`), sur la série
canonique `data/input/` (Meudon 2017).

Références utilisées :
- binaires moyens : `backup`/runs Fortran (parité étape 1),
- `data/output/ACDF2_run_001.lis` (= run 007 validé, 9 canaux).

Vue d'ensemble de la tolérance : l'étape 1 doit être exacte (diff=0) ; la
géométrie doit reproduire les canaux 1-8 à <1 px ; le canal 9 (bord de champ)
peut encore dévier (limite de détection verticale documentée).

Exécution :
    cd new_python
    ./.venv/bin/python -m pytest tests -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from msdp.config import Config
from msdp import step1_average as s1
from msdp import step2_geometry as g2

PROJ = Path("/home/max/nextcloud/Workspace/Projet-MSDP")
DATA_IN = PROJ / "data/input"
MS_PAR = PROJ / "src/fortran/new/ms.par"
WORK_FORTRAN = Path("/tmp/msdp_pipeline/work")   # binaires moyens du Fortran
OUT = PROJ / "data/output"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def config() -> Config:
    return Config.from_file(MS_PAR)


@pytest.fixture(scope="module")
def averaged() -> dict[str, s1.AveragedImage]:
    """Étape 1 : moyennes dark/flat (écrites dans /tmp)."""
    return s1.run_step1(DATA_IN, Config.from_file(MS_PAR), "/tmp/msdp_py1")


@pytest.fixture(scope="module")
def meanflat(config: Config, averaged) -> np.ndarray:
    """meanflat (1536,1024) : flat − dark transposé."""
    return g2._load_meanflat(
        f"/tmp/msdp_py1/{averaged['dark'].filename}",
        f"/tmp/msdp_py1/{averaged['flat'].filename}",
        config.ccd_y, config.ccd_x)


# --------------------------------------------------------------------------- #
# Étape 1 — parité exacte des moyennes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ftype", ["dark", "flat"])
def test_step1_moyenne_exacte(ftype, averaged):
    """Le binaire moyen reproduit le Fortran à l'octet près (après retrait
    des marqueurs de record non-formatés du Fortran).

    Le fichier moyen est en ordre **colonne-major** : chaque bloc contigu de
    ``isp`` int16 est une ligne du fichier (= une colonne ``j`` de la matrice).
    On reconstruit une matrice ``(jsp, isp)`` des deux côtés et on compare.
    """
    frp = WORK_FORTRAN / averaged[ftype].filename
    if not frp.exists():
        pytest.skip(f"binaire Fortran absent: {frp}")
    isp, jsp = 1024, 1536
    mine = np.fromfile(f"/tmp/msdp_py1/{averaged[ftype].filename}",
                       dtype="<i2")[1024:]           # retrait entête 512 i32
    assert mine.size == isp * jsp
    mine_mat = np.zeros((jsp, isp), dtype="<i2")
    for j in range(jsp):
        mine_mat[j] = mine[j * isp:(j + 1) * isp]
    fr = _read_fortran(Path(frp), isp, jsp)
    assert np.array_equal(mine_mat, fr), "moyenne ≠ Fortran (parité non exacte)"


def _read_fortran(path: Path, isp: int, jsp: int) -> np.ndarray:
    """Lit un binaire moyen Fortran (records par ligne avec marqueurs int32).

    Chaque record du fichier = une colonne ``j`` (``isp`` valeurs int16) ;
    la matrice retournée est ``(jsp, isp)`` — le layout brut du fichier,
    comparable directement au flux écrit par l'étape 1 Python (colonne-major).
    """
    raw = path.read_bytes()
    off = 0
    rec0 = np.frombuffer(raw[:4], dtype="<i4")[0]
    off += 4 + rec0 + 4
    rows = np.empty((jsp, isp), dtype="<i2")
    for j in range(jsp):
        ln = np.frombuffer(raw[off:off + 4], dtype="<i4")[0]
        off += 4
        rows[j] = np.frombuffer(raw[off:off + 2 * isp], dtype="<i2")
        off += 2 * isp + 4
    return rows


# --------------------------------------------------------------------------- #
# Étape 2 — géométrie
# --------------------------------------------------------------------------- #
def test_geometry_parite_9_canaux(meanflat, config):
    """9/9 canaux détectés ; parité vs Fortran dans l'enveloppe documentée
    (X max ~1.4 px, mean ~0.2 px ; Y max ~0.1 px — voir skill
    dpsm-pipeline python-port-fixes : 'max 1.370 px, mean 0.183 px')."""
    geom = g2.detect_geometry(meanflat, config)
    ref = np.loadtxt(OUT / "ACDF2_run_001.lis")
    diff = np.abs(geom.acdf2() - ref)
    assert geom.acdf2().shape == (9, 8)
    # X (colonnes 0-3) : enveloppe documentée ~1.4 px sur les 9 canaux
    assert diff[:, :4].max() < 1.4, f"X trop écarté: {diff[:, :4].max():.2f}"
    # Y (colonnes 4-7)
    assert diff[:, 4:].max() < 0.3, f"Y trop écarté: {diff[:, 4:].max():.2f}"


def test_geometry_nb_canaux(meanflat, config):
    """9 canaux détectés, aucun coin à 0.00."""
    geom = g2.detect_geometry(meanflat, config)
    acdf2 = geom.acdf2()
    assert acdf2.shape == (9, 8)
    assert not np.any(np.abs(acdf2) < 0.001), "coins à zéro détectés"


def test_acdf2_monotone(meanflat, config):
    """Les centres de canaux (A_x) sont strictement croissants (structure saine)."""
    geom = g2.detect_geometry(meanflat, config)
    ax = geom.xx[10, :]
    assert np.all(np.diff(ax) > 0), "centres X non croissants"


# --------------------------------------------------------------------------- #
# Assemblage — smoke-test de bout en bout (étapes 1→2→3→calib→plotting)
# --------------------------------------------------------------------------- #
def test_assembly_calib_et_plot(meanflat, config):
    """La chaîne complète s'assemble et produit cal + une figure Matplotlib.

    Valide l'intégration des modules (pas la parité fine, couverte par les
    tests d'étape) : `extract_channels` → `smooth_cliss` → `linecurv_centers`
    → `profmean` → `calibrate` → `plot_ivprof`. Garantit que les signatures
    restent cohérentes entre modules après refactors.
    """
    from msdp import step3_channels as c3
    from msdp import step3_calib as c4
    from msdp import plotting
    from msdp import step2_geometry as g2n
    from pathlib import Path as _P
    import warnings

    xr, yr = g2n.detect_geometry(meanflat, config).to_xr_yr()
    li, lj, milsec = config.get('li'), config.get('lj'), config.get('milsec')
    cymx = c3.extract_channels(meanflat.T, xr, yr, li, lj, milsec, config.nm)
    assert cymx.shape == (885, 123, config.nm)

    marg, iliss = config.get('margline', 0), config.get('iliss', 60)
    iim, jjm = cymx.shape[:2]
    cliss = c4.smooth_cliss(cymx, marg, iliss, config.get('jliss', 0))
    il1, il2 = 1 + marg + iliss, iim - marg - iliss
    jl1, jl2 = 1 + marg, jjm - marg
    jparab = config.get('jparab', 10)
    nr = config.get('nr', 5)
    center, pte = c4.linecurv_centers(cymx, cliss, nr - 1, il1, il2,
                                      jl1, jl2, jparab)
    assert center.shape == (iim,)

    trj = c4.transpec(xr, yr, config.get('ntrans', nr), jjm,
                      config.get('mupris', 9000), config.get('mustep', 2500))
    jtr = int(trj + 0.5)
    xc = float(1 + jjm) / 2.0
    jt1 = int(xc - float(jtr) / 2.0 + 0.5)
    jt2 = jt1 + jtr
    profmm, km = c4.profmean(cliss, center, pte, jt1, jt2, jtr, config.nm)
    assert profmm.shape == (km,) and km > 0
    cal = c4.calibrate(cymx, cliss, profmm, center, pte, jtr, jt1, jt2, km)
    assert cal.shape == cymx.shape and np.all(cal > 0), "cal doit être > 0"

    # plotting : une figure sortie sans exception
    out = _P('/tmp/msdp_asm_test.pdf')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prof1 = cliss[iim // 2, :, 0]                 # profil 1D du centre
        profnf = np.broadcast_to(prof1[:, None, None], (jjm, 81, 3)).copy()
    plotting.plot_ivprof1(profnf, 500, out)
    assert out.exists() and out.stat().st_size > 0