"""Logbook — génération d'un log de pipeline propre et structuré.

Remplace le ``ms.lis`` du Fortran (journal de brouillon, format non nominal)
par un log lisible par sections, construit à partir des grandeurs clés que
``run_pipeline`` collecte dans ``PipelineResult.notes``.

Écrit ``ms_run_<run_label>.lis`` dans le dossier de run.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def fmt_num(x: Any, nd: int = 3) -> str:
    """Formate une grandeurs en nombre lisible (float/None/list)."""
    if x is None:
        return "-"
    if isinstance(x, (int, float)):
        return f"{float(x):.{nd}f}"
    return str(x)


def _acdf2_block(rows: list[list[float]], nd: int = 2) -> str:
    """Tableau ACDF2 : une ligne par canal (A C D F en X puis Y)."""
    lines = ["    can  |   A_x    C_x    D_x    F_x |   A_y    C_y    D_y    F_y"]
    lines.append("    " + "-" * 58)
    for i, r in enumerate(rows, 1):
        vals = " ".join(f"{float(v):9.2f}" for v in r)
        lines.append(f"    {i:3d} |{vals}")
    if rows:
        lines.append("    " + "-" * 58)
        lines.append(f"      (canal 1: A=({rows[0][0]:.2f}, {rows[0][4]:.2f}))")
    else:
        lines.append("    (aucun canal détecté)")
    return "\n".join(lines)


def build_log(config: Any, notes: dict[str, Any], run_label: str,
              ms_par: str = "", outputs: dict[str, Any] | None = None) -> str:
    """Construit le log structuré à partir des notes collectées."""
    L = []
    bar = "#" * 70
    L.append(bar)
    L.append("#  MSDP pipeline — Python (new_python)")
    L.append(f"#  run         : {run_label}")
    L.append(f"#  date        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if ms_par:
        L.append(f"#  ms.par      : {ms_par}")
    L.append(bar)

    # --- Step 1 : moyennes ---
    s1 = notes.get("step1", {})
    L.append("\n[step1] Moyennes dark/flat")
    L.append(f"  dark         : {s1.get('dark', '-')}  (n_dark={s1.get('n_dark', '-')})")
    L.append(f"  flat         : {s1.get('flat', '-')}  (n_flat={s1.get('n_flat', '-')})")

    # --- Step 2 : géométrie ---
    s2 = notes.get("step2", {})
    L.append("\n[step2] Géométrie (newgeom)")
    L.append(f"  ja(1,2,3)    : {s2.get('ja', '-')}")
    L.append(f"  distorsion   : {fmt_num(s2.get('val_qm'))} px (quad. mean)")
    acdf2 = s2.get("acdf2")
    if acdf2:
        L.append(_acdf2_block(acdf2))

    # --- Step 3 : canaux + calib ---
    s3 = notes.get("step3", {})
    L.append("\n[step3] Canaux + calibration")
    cymx = s3.get("cymx")
    if cymx:
        L.append(f"  cymx         : {cymx[0]} × {cymx[1]} × {cymx[2]}"
                 f"  range [{fmt_num(s3.get('cymx_range',[0,0])[0])} .. "
                 f"{fmt_num(s3.get('cymx_range',[0,0])[1])}]")
    L.append(f"  transpec     : {fmt_num(s3.get('transpec'))}   "
             f"(jtr={s3.get('jtr', '-')}, jt1/jt2={s3.get('jt1_jt2', '-')})")
    L.append(f"  pte(nr)      : {fmt_num(s3.get('pte(nr)'))}")
    L.append(f"  center(1,nr) : {fmt_num(s3.get('center(1,nr)'))}")
    L.append(f"  yic(nr)      : {fmt_num(s3.get('yic(nr)'))}")
    L.append(f"  km           : {s3.get('km', '-')}")
    calr = s3.get("cal_range")
    if calr:
        L.append(f"  cal range    : [{fmt_num(calr[0])} .. {fmt_num(calr[1])}]")
    # échantillons cal (mêmes points que le log Fortran "calib.ps")
    cp = s3.get("cal_probe")
    if cp:
        for icd in sorted(cp, key=int):
            row = "  ".join(f"{v:8.4f}" for v in cp[icd][1])
            L.append(f"  cal i={icd:<4d} (canal1): {row}")

    # --- Step 4a : solarobs ---
    s4a = notes.get("step4a", {})
    L.append("\n[step4a] Observations (solarobs)")
    L.append(f"  n_obs        : {s4a.get('n_obs', 0)}")
    sobs = s4a.get("sobs(ic,jc,n)")
    if sobs:
        L.append("  sobs(ic,jc)  : " + "  ".join(f"{v:8.1f}" for v in sobs))
    sr = s4a.get("sobs_range")
    if sr:
        L.append(f"  sobs range   : [{fmt_num(sr[0])} .. {fmt_num(sr[1])}]")

    # --- Step 4b : profils / vitesses ---
    s4b = notes.get("step4b", {})
    L.append("\n[step4b] Profils I/V + vitesses")
    pro = s4b.get("profiles")
    if pro:
        L.append(f"  profnf3      : {pro[0]} × {pro[1]} profils")
    L.append(f"  lbdvels      : {s4b.get('lbdvels', '-')}")
    nv = s4b.get("n_vitesses", 0)
    L.append(f"  vitesses     : {nv}")
    for lv, xv, yv in s4b.get("vitesses", []):
        L.append(f"      intvel3  lbdvel={lv:3d}  xv={xv:9.2f}  "
                 f"yv={yv:9.2f}  (λ={xv:9.2f})")

    # --- Timings ---
    tim = notes.get("timings", {})
    if tim:
        L.append("\n[timings]  (secondes)")
        order = ["step1_moyennes (lecture dark/flat)", "step2_geometrie",
                 "step3a_canaux", "step3b_calib",
                 "step4a_solarobs (lecture+extract obs)", "step4b_ivmaps",
                 "write_outputs (figures+lis)"]
        for k in order:
            if k in tim:
                L.append(f"  {k:36s}: {tim[k]:8.3f}")
        for k, v in tim.items():
            if k not in order and k != "total":
                L.append(f"  {k:36s}: {v:8.3f}")
        if "total" in tim:
            L.append(f"  {'total':36s}: {tim['total']:8.3f}")

    # --- Sorties ---
    if outputs:
        L.append("\n[sorties]")
        for k, v in sorted(outputs.items()):
            if isinstance(v, (str, int, float)):
                L.append(f"  {k:14s}: {v}")
    return "\n".join(L) + "\n"


def write_log(res: Any, outdir: str | Path) -> Path:
    """Écrit ``ms_run_<run_label>.lis`` dans le dossier de run."""
    outdir = Path(outdir)
    run_label = res.outputs.get("run_label", "py") or "py"
    content = build_log(res.config, res.notes, run_label,
                        ms_par=res.outputs.get("ms_par", ""),
                        outputs={k: v for k, v in res.outputs.items()
                                 if k not in ("run_dir", "run_label",
                                              "ms_par", "ms_run.log")})
    path = outdir / f"ms_run_{run_label}.lis"
    path.write_text(content)
    return path