"""Pipeline MSDP — orchestrateur de bout en bout (port Python du flux Fortran).

Reproduit la chaîne ``ms1.f → ms2.f → ms3.f → ms4.f`` : moyennes dark/flat →
géométrie → canaux → calibration → observations → profils I/V + vitesses.
Toutes les étapes sont portées depuis ``src/fortran/new/`` et validées
séparément ; ce module les assemble dans le même ordre que le Fortran et écrit
les sorties (ACDF2.lis, PDFs, miv.lis) dans un dossier de run ``run_XXX/``.

Usage :
    from msdp.pipeline import run_pipeline
    result = run_pipeline(config, work_dir="...")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from msdp.config import Config
from msdp import step1_average as s1
from msdp import step2_geometry as s2
from msdp import step3_channels as s3c
from msdp import step3_calib as s3b
from msdp import step4_solarobs as s4s
from msdp import step4_ivmaps as s4i
from msdp import plotting


@dataclass
class PipelineResult:
    """Sorties du pipeline (chiffrées + chemins des fichiers produits)."""
    config: Config
    cymx: np.ndarray = field(default_factory=lambda: np.empty(0))
    cal: np.ndarray = field(default_factory=lambda: np.empty(0))
    xr: np.ndarray = field(default_factory=lambda: np.empty(0))
    yr: np.ndarray = field(default_factory=lambda: np.empty(0))
    sobstot: np.ndarray = field(default_factory=lambda: np.empty(0))
    profnf3: np.ndarray = field(default_factory=lambda: np.empty(0))
    alinea: np.ndarray = field(default_factory=lambda: np.empty(0))
    vitesses: list = field(default_factory=list)
    outputs: dict = field(default_factory=dict)


def run_pipeline(config: Config, data_dir: str | Path,
                 work_dir: str | Path | None = None,
                 run_label: str = "py") -> PipelineResult:
    """Exécute le pipeline complet.

    Parameters
    ----------
    config : Config
        Paramètres (ms.par).
    data_dir : str | Path
        Répertoire des FITS d'entrée (darks x1, flats y1, obs b1).
    work_dir : str | Path | None
        Répertoire de travail (binaires moyens + sorties). Défaut : goût.
    run_label : str
        Étiquette du run (numérotation comme run_pipeline.sh).

    Returns
    -------
    PipelineResult
    """
    data_dir = Path(data_dir)
    work_dir = Path(work_dir) if work_dir else data_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    res = PipelineResult(config=config)

    # ---- Étape 1 : moyennes dark/flat ----
    avg = s1.run_step1(data_dir, config, work_dir)
    dark = work_dir / avg["dark"].filename
    flat = work_dir / avg["flat"].filename

    # ---- Étape 2 : géométrie ----
    mf = s2._load_meanflat(dark, flat, config.ccd_y, config.ccd_x)
    geom2 = s2.detect_geometry(mf, config)
    res.xr, res.yr = geom2.to_xr_yr()
    res.outputs["ACDF2.lis"] = ""

    # ---- Étape 3 : canaux + calibration ----
    li, lj, milsec = config.get("li"), config.get("lj"), config.get("milsec")
    iim, jjm = s3c.channels_dims(li, lj, milsec)
    cymx = s3c.extract_channels(mf.T, res.xr, res.yr, li, lj, milsec, config.nm)
    res.cymx = cymx

    # calibration (chaîne step3_calib)
    marg = config.get("margline", 0); iliss = config.get("iliss", 60)
    cliss = s3b.smooth_cliss(cymx, marg, iliss, config.get("jliss", 0))
    il1, il2 = 1 + marg + iliss, iim - marg - iliss
    jl1, jl2 = 1 + marg, jjm - marg
    nr = config.get("nr", 5); jparab = config.get("jparab", 10)
    nc = (1 + config.nm) // 2
    # centre + pente du canal de référence (ncurv) : linecurv
    # et yln/center pour les panneaux A/B de calib (n=nc, nc+1)
    yln_all = np.zeros((config.nm, iim + 1))
    center_all = np.zeros((config.nm, iim))
    for n in (nc - 1, nc, nr - 1):          # 0-based nc, nc+1 (1-based) et ncurv
        yln, center, pte = s3b.linecurv_curve(
            cymx, cliss, n, il1, il2, jl1, jl2, jparab)
        yln_all[n, :iim + 1] = yln[:iim + 1]
        center_all[n, :] = center
        if n == nr - 1:
            center_ref, pte_ref = center, pte
    trj = s3b.transpec(res.xr, res.yr, config.get("ntrans", nr), jjm,
                       config.get("mupris", 9000), config.get("mustep", 2500))
    # jtr : translation entre canaux (Fortran : jtr = trjm+0.5 si ntrans≠0)
    jtr = int(trj + 0.5)
    # bornes du profil moyen (ms2.f calib) : xc=(1+jmd)/2 ; jt1=xc-jtr/2+0.5
    xc = float(1 + jjm) / 2.0
    jt1 = int(xc - float(jtr) / 2.0 + 0.5)
    jt2 = jt1 + jtr
    # centre du canal de référence pour la pente pte
    profmm, km = s3b.profmean(cliss, center_ref, pte_ref, jt1, jt2, jtr,
                              config.nm, nr=nr)
    res.cal = s3b.calibrate(cymx, cliss, profmm, center_ref, pte_ref,
                            jtr, jt1, jt2, km)
    # pronoc pour le panneau "Mean profile" de calib
    pronoc, _, _, _ = s3b._build_profmm(cliss, jt1, jt2, jtr, config.nm, nr)

    # ---- Étape 4a : observations ----
    obs = sorted(p.name for p in data_dir.glob("*b1.fit"))
    # répertoire des binaires pour les darks
    sobstot, sobsplot, iim_out, jjm_out = s4s.solarobs(
        [data_dir / o for o in obs], dark, res.cal, res.xr, res.yr, config,
        1, min(config.get("nfb2", 1), len(obs)), config.get("nfplot", 1))
    res.sobstot = sobstot
    # obsD1 (non calibré = sobs×cal) et obsD2 (calibré = sobs), normalisés
    sobsD1, sobsD2 = None, None
    if sobstot.size:
        sref = sobstot[:, :, :, 0]                # observation calibrée (nf0)
        sobsD1, sobsD2 = s4s.compute_obsD(sref, res.cal)

    # ---- Étape 4b : profils I/V ----
    lbdvels = (config.get("lbdvel1", 0), config.get("lbdvel2", 0),
               config.get("lbdvel3", 0))
    profnf3, vels = s4i.ivmap3(sobstot, res.cal, iim_out, jjm_out, config.nm,
                               lbdvels)
    res.profnf3 = profnf3
    res.vitesses = vels
    profnf4 = None
    if iim_out and jjm_out:
        profnf4, _ = s4i.ivmap4(sobstot, res.cal, iim_out, jjm_out,
                                config.nm, profnf3)

    # ---- Plotting (sortie) — tout en X/Y (arcsec) et N&B ----
    outdir = work_dir / f"run_{run_label}"
    outdir.mkdir(exist_ok=True)
    milsec = int(config.get("milsec", 500))
    iim_c, jjm_c = s3c.channels_dims(li, lj, milsec)
    im_ccd, jm_ccd = mf.shape            # meanflat (1536, 1024) — repère image
    # ACDF2.lis réel
    geom2.write_acdf2(outdir / "ACDF2.lis")

    plotting.plot_geo1(geom2, im_ccd, jm_ccd, outdir / "geo1.pdf")
    plotting.plot_geo2(geom2, im_ccd, outdir / "geo2.pdf")
    plotting.plot_geo3(geom2.xx, geom2.yy,
                       float(config.get("xdel", 25)), mf.shape[1],
                       outdir / "geo3.pdf")
    plotting.plot_geo4(geom2.xx, geom2.yy, config.nm, outdir / "geo4.pdf")

    if cymx.size:
        plotting.plot_calib(res.cal, iim, jjm, config.nm,
                            int(config.get("x1cal", 100)),
                            int(config.get("x2cal", 750)),
                            jtr, jt1, jt2, outdir / "calib.pdf",
                            yln=yln_all.T, center=center_all.T, nc=nc,
                            il1=il1, il2=il2,
                            pronoc=pronoc, profmm=profmm)
        plotting.plot_calmap(res.cal, outdir / "cal.pdf",
                             float(config.get("icalbl", 800)) / 1000.0,
                             float(config.get("icalwh", 1200)) / 1000.0)
    if sobstot.size:
        plotting.plot_obs(sobstot[:, :, :, 0], milsec, outdir / "obs.pdf",
                          vmin=float(config.get("iobsbl", 500)),
                          vmax=float(config.get("iobswh", 1800)))
        plotting.plot_obsD1(sobsD1, milsec, outdir / "obsD1.pdf",
                            vmin=float(config.get("iobsDbl", 800)),
                            vmax=float(config.get("iobsDwh", 1200)))
        plotting.plot_obsD2(sobsD2, milsec, outdir / "obsD2.pdf",
                            vmin=float(config.get("iobsDbl", 800)),
                            vmax=float(config.get("iobsDwh", 1200)))
    if profnf3.size:
        plotting.plot_ivprof1(profnf3, milsec, outdir / "ivprof1.pdf")
        if profnf4 is not None and profnf4.shape[2] >= 3:
            plotting.plot_ivprof2(profnf4, milsec, outdir / "ivprof2.pdf")
            plotting.plot_ivprof3(profnf4, milsec, outdir / "ivprof3.pdf")

    # vitesses -> miv.lis
    with (outdir / "miv.lis").open("w") as fh:
        fh.write("    intvel  lbdvel     xv2     yv2\n")
        for lv, xv, yv in vels:
            fh.write(f"       1  {lv:3d}  {xv:8.2f} {yv:8.2f}\n")
    res.outputs["run_dir"] = str(outdir)

    return res