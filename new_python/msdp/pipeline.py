"""Pipeline MSDP — orchestrateur de bout en bout (port Python du flux Fortran).

Reproduit la chaîne ``ms1.f → ms2.f → ms3.f → ms4.f`` : moyennes dark/flat →
géométrie → canaux → calibration → observations → profils I/V + vitesses.
Toutes les étapes sont portées depuis ``src/fortran/new/`` et validées
séparément ; ce module les assemble dans le même ordre que le Fortran et écrit
les sorties (ACDF2.csv, figures PNG, miv.csv) dans un dossier de run ``run_XXX/``.

Usage :
    from msdp.pipeline import run_pipeline
    result = run_pipeline(config, work_dir="...")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np

from msdp.config import Config
from msdp import step1_average as s1
from msdp import step2_geometry as s2
from msdp import step3_channels as s3c
from msdp import step3_calib as s3b
from msdp import step4_solarobs as s4s
from msdp import step4_ivmaps as s4i
from msdp import plotting
from msdp import logbook


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
    notes: dict = field(default_factory=dict)   # grandeurs clés pour le log


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
    timings: dict[str, float] = {}

    def _tick(label: str, t0: float | None = None) -> float:
        if t0 is None:
            return time.perf_counter()
        timings[label] = time.perf_counter() - t0
        return timings[label]

    _T0 = time.perf_counter()

    # ---- Étape 1 : moyennes dark/flat (= lecture inputs dark/flat) ----
    _t = time.perf_counter()
    avg = s1.run_step1(data_dir, config, work_dir)
    dark = work_dir / avg["dark"].filename
    flat = work_dir / avg["flat"].filename
    _tick("step1_moyennes (lecture dark/flat)", _t)
    res.notes["step1"] = {
        "dark": avg["dark"].filename, "flat": avg["flat"].filename,
        "n_dark": config.get("nfx1", 1), "n_flat": config.get("nfy1", 1),
    }

    # ---- Étape 2 : géométrie ----
    _t = time.perf_counter()
    mf = s2._load_meanflat(dark, flat, config.ccd_y, config.ccd_x)
    geom2 = s2.detect_geometry(mf, config)
    res.xr, res.yr = geom2.to_xr_yr()
    res.outputs["ACDF2.lis"] = ""
    _tick("step2_geometrie", _t)
    res.notes["step2"] = {
        "ja": list(geom2.ja), "val_qm": float(geom2.val_qm),
        "acdf2": geom2.acdf2().tolist(),
    }

    # ---- Étape 3 : canaux + calibration ----
    _t = time.perf_counter()
    li, lj, milsec = config.get("li"), config.get("lj"), config.get("milsec")
    iim, jjm = s3c.channels_dims(li, lj, milsec)
    cymx = s3c.extract_channels(mf.T, res.xr, res.yr, li, lj, milsec, config.nm)
    res.cymx = cymx
    _tick("step3a_canaux", _t)
    res.notes["step3a"] = {"cymx": [iim, jjm, config.nm],
                           "cymx_range": [float(cymx.min()), float(cymx.max())]}

    # calibration (chaîne step3_calib)
    _t = time.perf_counter()
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
    _tick("step3b_calib", _t)
    # échantillons de cal aux mêmes points que le log Fortran "calib.ps" :
    # positions i = x1cal / x2cal, j = 1..jjm pas 20. Structure :
    #   cal_probe = {icd: {canal(1-based): [val(j=1), val(j=21), ...]}}
    x1cal, x2cal = int(config.get("x1cal", 100)), int(config.get("x2cal", 750))
    cal_probe: dict[int, dict[int, list[float]]] = {}
    for icd in (x1cal, x2cal):
        if not (1 <= icd <= iim):
            continue
        cal_probe[icd] = {n: [float(res.cal[icd - 1, jj - 1, n - 1])
                              for jj in range(1, jjm + 1, 20)]
                          for n in range(1, config.nm + 1)}
    res.notes["step3"] = {
        "transpec": float(trj), "jtr": int(jtr),
        "jt1_jt2": [int(jt1), int(jt2)], "km": int(km),
        "pte(nr)": float(pte_ref), "center(1,nr)": float(center_ref[0]),
        "yic(nr)": float(center_ref[(1 + iim) // 2 - 1]),
        "cal_range": [float(res.cal.min()), float(res.cal.max())],
        "cal_probe": cal_probe,
        "ntrans": int(config.get("ntrans", nr)),
        "x1cal": x1cal, "x2cal": x2cal,
    }

    # ---- Étape 4a : observations (= lecture inputs observations) ----
    _t = time.perf_counter()
    obs = sorted(p.name for p in data_dir.glob("*b1.fit"))
    # répertoire des binaires pour les darks
    sobstot, sobsplot, iim_out, jjm_out = s4s.solarobs(
        [data_dir / o for o in obs], dark, res.cal, res.xr, res.yr, config,
        1, min(config.get("nfb2", 1), len(obs)), config.get("nfplot", 1))
    res.sobstot = sobstot
    _tick("step4a_solarobs (lecture+extract obs)", _t)
    # obsD1 (non calibré = sobs×cal) et obsD2 (calibré = sobs), normalisés
    sobsD1, sobsD2 = None, None
    if sobstot.size:
        sref = sobstot[:, :, :, 0]                # observation calibrée (nf0)
        sobsD1, sobsD2 = s4s.compute_obsD(sref, res.cal)
    icc = (iim_out + 1) // 2 - 1 if iim_out else 0
    jcc = (jjm_out + 1) // 2 - 1 if jjm_out else 0
    res.notes["step4a"] = {
        "n_obs": int(sobstot.shape[3]) if sobstot.size else 0,
        "sobs(ic,jc,n)": ([float(x) for x in sobstot[icc, jcc, :, 0]]
                          if sobstot.size and 0 <= icc < iim_out else []),
        "sobs_range": [float(sobstot.min()), float(sobstot.max())]
        if sobstot.size else [0.0, 0.0],
    }

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
    _tick("step4b_ivmaps", _t)
    res.notes["step4b"] = {
        "lbdvels": list(lbdvels),
        "profiles": [int(profnf3.shape[0]), int(profnf3.shape[1])]
        if profnf3.size else [0, 0],
        "n_vitesses": len(vels),
        "vitesses": [(int(x[0]), float(x[1]), float(x[2])) for x in vels],
    }

    # ---- Plotting (sortie = écriture des figures) — X/Y (arcsec) et N&B ----
    _t = time.perf_counter()
    outdir = work_dir / f"run_{run_label}"
    outdir.mkdir(exist_ok=True)
    milsec = int(config.get("milsec", 500))
    iim_c, jjm_c = s3c.channels_dims(li, lj, milsec)
    im_ccd, jm_ccd = mf.shape            # meanflat (1536, 1024) — repère image
    # ACDF2 (format CSV moderne)
    geom2.write_acdf2(outdir / "ACDF2.csv")

    plotting.plot_geo1(geom2, im_ccd, jm_ccd, outdir / "geo1.png")
    plotting.plot_geo2(geom2, im_ccd, outdir / "geo2.png")
    plotting.plot_geo3(geom2.xx, geom2.yy,
                       float(config.get("xdel", 25)), mf.shape[1],
                       outdir / "geo3.png")
    plotting.plot_geo4(geom2.xx, geom2.yy, config.nm, outdir / "geo4.png")

    if cymx.size:
        plotting.plot_calib(res.cal, iim, jjm, config.nm,
                            int(config.get("x1cal", 100)),
                            int(config.get("x2cal", 750)),
                            jtr, jt1, jt2, outdir / "calib.png",
                            yln=yln_all.T, center=center_all.T, nc=nc,
                            il1=il1, il2=il2,
                            pronoc=pronoc, profmm=profmm)
        plotting.plot_calmap(res.cal, outdir / "cal.png",
                             float(config.get("icalbl", 800)) / 1000.0,
                             float(config.get("icalwh", 1200)) / 1000.0)
    if sobstot.size:
        plotting.plot_obs(sobstot[:, :, :, 0], milsec, outdir / "obs.png",
                          vmin=float(config.get("iobsbl", 500)),
                          vmax=float(config.get("iobswh", 1800)))
        plotting.plot_obsD1(sobsD1, milsec, outdir / "obsD1.png",
                            vmin=float(config.get("iobsDbl", 800)),
                            vmax=float(config.get("iobsDwh", 1200)))
        plotting.plot_obsD2(sobsD2, milsec, outdir / "obsD2.png",
                            vmin=float(config.get("iobsDbl", 800)),
                            vmax=float(config.get("iobsDwh", 1200)))
    if profnf3.size:
        plotting.plot_ivprof1(profnf3, milsec, outdir / "ivprof1.png")
        if profnf4 is not None and profnf4.shape[2] >= 3:
            plotting.plot_ivprof2(profnf4, milsec, outdir / "ivprof2.png")
            plotting.plot_ivprof3(profnf4, milsec, outdir / "ivprof3.png")

    # vitesses -> miv.csv (colonnes : lbdvel, xv2, yv2)
    with (outdir / "miv.csv").open("w") as fh:
        fh.write("lbdvel,xv2,yv2\n")
        for lv, xv, yv in vels:
            fh.write(f"{lv},{xv:.6f},{yv:.6f}\n")
    _tick("write_outputs (figures+lis)", _t)
    timings["total"] = time.perf_counter() - _T0
    res.notes["timings"] = timings
    res.outputs["run_dir"] = str(outdir)
    res.outputs["run_label"] = run_label
    res.outputs.setdefault("ms_par", str(config.ms_par_path)
                           if hasattr(config, "ms_par_path") else "")

    # --- Log structuré JSON (remplace ms_run.lis du Fortran) ---
    logbook.write_log(res, outdir)
    res.outputs["ms_run.log"] = str(outdir / f"ms_run_{run_label}.json")

    return res