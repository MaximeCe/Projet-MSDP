#!/usr/bin/env python3
"""CLI du pipeline Python MSDP (new_python).

Exécute le pipeline complet (moyennes → géométrie → canaux → calib → obs →
profils I/V) et regroupe TOUTES les sorties dans un dossier ``run_NNN/`` sous
``new_python/outputs/`` (numéroté automatiquement), plus les logs versionnés
``ms_run_py_NNN.lis`` / ``ACDF2_run_py_NNN.lis``.

Usage :
    python run_pipeline_cli.py [ms.par] [data_dir]
      ms.par    : défaut src/fortran/new/ms.par
      data_dir  : défaut data/input
"""
from __future__ import annotations

import shutil
import os
import sys
import time
from pathlib import Path

# permet d'importer msdp depuis ce fichier (racine new_python/)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from msdp.config import Config
from msdp.pipeline import run_pipeline

PROJ = Path(__file__).resolve().parent.parent
MS_PAR_DEF = PROJ / "src/fortran/new/ms.par"
DATA_DIR_DEF = PROJ / "data/input"
OUT_ROOT = Path(__file__).resolve().parent / "outputs"
LOG_OUT = PROJ / "data/output"            # ms_run_py_NNN.lis consignés ici


def _next_run_n(out_root: Path) -> str:
    """Prochain numéro de run libre (run_NNN/) dans out_root."""
    mx = 0
    if out_root.is_dir():
        for d in out_root.glob("run_*"):
            if d.is_dir():
                try:
                    mx = max(mx, int(d.name.rsplit("_", 1)[1]))
                except (ValueError, IndexError):
                    pass
    return f"{mx + 1:03d}"


def main(argv: list[str]) -> int:
    ms_par = Path(argv[1]) if len(argv) > 1 else MS_PAR_DEF
    data_dir = Path(argv[2]) if len(argv) > 2 else DATA_DIR_DEF
    if not ms_par.is_file():
        print(f"ERREUR: ms.par introuvable: {ms_par}")
        return 1
    if not data_dir.is_dir():
        print(f"ERREUR: data_dir introuvable: {data_dir}")
        return 1

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_OUT.mkdir(parents=True, exist_ok=True)
    run_num = _next_run_n(OUT_ROOT)
    work = Path(f"/tmp/msdp_py_run_{run_num}")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    print("=" * 40)
    print("  Pipeline Python MSDP (new_python)")
    print(f"  ms.par  : {ms_par}")
    print(f"  data_dir: {data_dir}")
    print(f"  run     : run_{run_num}")
    print("=" * 40)

    config = Config.from_file(ms_par)
    t0 = time.time()
    res = run_pipeline(config, data_dir, work_dir=work, run_label=run_num)
    dt = time.time() - t0

    # --- regroupement des sorties dans new_python/outputs/run_NNN/ ---
    src_dir = work / f"run_{run_num}"
    run_dir = OUT_ROOT / f"run_{run_num}"
    run_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src_dir.iterdir()):
        if f.is_file():
            shutil.copy2(f, run_dir / f.name)
            n += 1
    # copie des logs versionnés dans data/output/
    for name in ("ACDF2.lis", "miv.lis"):
        p = run_dir / name
        if p.is_file():
            shutil.copy2(p, LOG_OUT / f"{Path(name).stem}_run_py_{run_num}.lis")
    logp = run_dir / f"ms_run_{run_num}.lis"
    if logp.is_file():
        shutil.copy2(logp, LOG_OUT / f"ms_run_py_{run_num}.lis")
    shutil.copy2(ms_par, LOG_OUT / f"ms_par_run_py_{run_num}.par")

    print(f"  ✓ {n} sorties regroupées dans {run_dir}/")
    print(f"  ⚙  duré: {dt:.1f} s  | vitesses: {len(res.vitesses)}")

    # --- Rapport de parité Fortran ↔ Python (validation.txt) ---
    try:
        from msdp.compare_runs import extract_fortran, extract_python, compare, \
            render, DEFAULT_TOLS, last_run_dir
        fdir = last_run_dir(LOG_OUT)
        if fdir is None:
            print("  ! pas de run Fortran trouvé → pas de rapport de parité")
        else:
            def _find(d: Path, stem: str) -> Path:
                c = [p for p in d.glob(f"{stem}*") if p.is_file()]
                return c[0] if c else (d / f"{stem}_001.lis")
            flog, facd, fmiv = (_find(fdir, "ms_run_"), _find(fdir, "ACDF2_run_"),
                                _find(fdir, "miv_run_"))
            f = extract_fortran(flog, facd, fmiv)
            p = extract_python(run_dir / f"ms_run_{run_num}.lis",
                               run_dir / "ACDF2.lis", run_dir / "miv.lis")
            report = render(compare(f, p, DEFAULT_TOLS))
            valpath = run_dir / "validation.txt"
            valpath.write_text(
                "# Parité MSDP Fortran ↔ Python\n"
                f"# Fortran : {fdir}\n"
                f"# Python  : {run_dir}\n" + "-" * 74 + "\n" + report)
            nfail = sum(1 for r in compare(f, p, DEFAULT_TOLS)
                        if r["stat"] == "FAIL")
            print(f"  ⚖  Validation: {valpath.name} "
                  f"({'À REVOIR' if nfail else 'VALIDÉ'})")
    except Exception as e:                     # ne jamais bloquer un run
        print(f"  ! rapport de parité échoué: {e}")

    print("=" * 40)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))