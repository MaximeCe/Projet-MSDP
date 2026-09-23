"""Comparateur Fortran ↔ Python MSDP — rapport de parité.

Prend un run Fortran (``data/output/run_NNN/``) et un run Python
(``new_python/outputs/run_NNN/``), extrait les valeurs clés de chacun (log
Fortran ``ms_run_*.lis`` / Python ``ms_run_*.json``, ``ACDF2`` / ``miv``) et
produit un rapport de *parité* direct (fichier de validation) indiquant écart
relatif / absolu et statut PASS / WARN / FAIL par grandeurs.

Formats : côté Python formats modernes (log JSON, ACDF2.csv, miv.csv) ;
rétrocompat sur l'ancien (log .lis texte, ACDF2.lis espace-séparé).

Usage :
    python -m msdp.compare_runs --fortran data/output/run_001 \\
                               --python  new_python/outputs/run_004
    # ou auto : dernier run Fortran + dernier run Python du projet.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

_PROJ = Path(__file__).resolve().parent.parent.parent

# --------------------------------------------------------------------------- #
# Extraction Fortran (log ms_run_NNN.lis)
# --------------------------------------------------------------------------- #
def parse_lines(path: Path) -> list[str]:
    """Lit un log en texte, lignes non vides."""
    out = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line:
            out.append(line)
    return out


def _grep(lines: list[str], pat: str) -> str | None:
    for ln in lines:
        if re.search(pat, ln):
            return ln
    return None


def extract_fortran(log: Path, acdf2: Path, miv: Path) -> dict[str, Any]:
    """Valeurs clés d'un run Fortran depuis son log + ACDF2 + miv."""
    lines = parse_lines(log)
    v: dict[str, Any] = {"source": "fortran"}

    # transpec
    m = re.search(r"transpec: ntrans,trjm1\s+(\d+)\s+([\d.]+)", "\n".join(lines))
    v["transpec"] = float(m.group(2)) if m else None

    # jtr, jt1, jt2 (before profmean: xc,jt1,jt2,jtr  62  41  83  42)
    m = re.search(r"before profmean.*?(?:\d+\.?\d*)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                  "\n".join(lines))
    if m:
        v["jt1"], v["jt2"], v["jtr"] = (float(m.group(1)), float(m.group(2)),
                                        float(m.group(3)))
    else:
        v["jt1"] = v["jt2"] = v["jtr"] = None

    # km
    m = re.search(r"profmean profmm km\s+(\d+)", "\n".join(lines))
    v["km"] = int(m.group(1)) if m else None

    # pte (n, yic, pte dy/dx  5  55.5-0.00635)  — yic positif, pte signé, collés
    m = re.search(r"yic, pte dy/dx\s+\d+\s+([\d.]+)([-+][\d.]+)",
                  "\n".join(lines))
    if m:
        v["yic"], v["pte"] = float(m.group(1)), float(m.group(2))
    else:
        v["yic"] = v["pte"] = None

    # ACDF2
    if acdf2.is_file():
        try:
            v["acdf2"] = np.loadtxt(acdf2)
        except Exception:
            v["acdf2"] = None
    else:
        v["acdf2"] = None

    # vitesses (miv) : lbdvel → (xv, yv)
    vel = []
    if miv.is_file():
        for ln in miv.read_text(errors="ignore").splitlines():
            m = re.match(r"\s*\d+\s+(\d+)\s+([\d.]+)\s+([\d.-]+)", ln)
            if m:
                vel.append((int(m.group(1)), float(m.group(2)),
                            float(m.group(3))))
    v["vitesses"] = vel

    # cal : lignes "calib.ps jmd,(y(nd),nd=1,jmd,20) 123  v1..v7"
    # 18 lignes = 2 positions i (x1cal, x2cal) × 9 canaux.
    # ordre Fortran : les 9 premières lignes = position x1cal, canaux 1..9 ;
    # les 9 suivantes = position x2cal. kannuc -> calib[p]{canal} = [7 vals]
    v["calib"] = parse_fortran_calib(lines)
    return v


def parse_fortran_calib(lines: list[str]) -> dict[int, dict[int, list[float]]]:
    """Parse les lignes ``calib.ps`` d'un log Fortran.

    Le Fortran écrit 18 lignes de validé (2 positions i \u00d7 9 canaux), chaque
    ligne : ``... 123  v1 v2 ... v7`` (j = 1,21,...,121 par pas de 20).

    Returns
    -------
    calib[pos] {canal 1-based: [7 valeurs]}  — 2 positions, 9 canaux chacune.
    """
    rows = []
    for ln in lines:
        if not ln.startswith("calib.ps"):
            continue
        m = re.search(r"calib\.ps.*?\d+\s+([\d.]+(?:\s+[\d.]+){6}\s*)", ln)
        if m:
            vals = [float(x) for x in m.group(1).split()]
            if len(vals) == 7:
                rows.append(vals)
    out: dict[int, dict[int, list[float]]] = {}
    if len(rows) < 18:
        return out                       # pas assez de données (log partiel)
    # 9 premières lignes = position x1cal, 9 suivantes = x2cal
    for pos_idx, icd in enumerate((1, 2)):
        out[icd] = {n: rows[pos_idx * 9 + (n - 1)] for n in range(1, 10)}
    return out


# --------------------------------------------------------------------------- #
# Extraction Python (log ms_run_NNN.json + ACDF2.csv + miv.csv)
# rétrocompatible : accepte l'ancien format (log .lis texte, ACDF2.lis, miv.lis)
# --------------------------------------------------------------------------- #
def _read_acdf2(path: Path):
    """ACDF2 : CSV (avec entête) ou .lis espace-séparé (ancien)."""
    if not path.is_file():
        return None
    line0 = path.read_text(errors="ignore").splitlines()[0]
    if "," in line0 and line0.strip().lower().startswith("can"):
        return np.loadtxt(str(path), delimiter=",", skiprows=1)
    if "," in line0:
        try:
            return np.loadtxt(str(path), delimiter=",")
        except ValueError:
            return np.loadtxt(str(path))
    try:
        return np.loadtxt(str(path))
    except ValueError:
        return None


def extract_python(log: Path, acdf2: Path, miv: Path) -> dict[str, Any]:
    """Valeurs clés d'un run Python (log JSON structuré ou ancien .lis texte)."""
    v: dict[str, Any] = {"source": "python"}

    # --- log : JSON (nouveau) ou texte (ancien) ---
    data: dict | None = None
    txt = ""
    if log.is_file():
        raw = log.read_text(errors="ignore")
        if log.suffix.lower() == ".json":
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = None
        if data is None:          # texte legacy
            txt = raw

    def _json(key: str, sub: str | None = None):
        if data is None:
            return None
        sec = data.get(key) if sub is None else (data.get(key) or {}).get(sub)
        return sec

    # transpec : JSON [step3].transpec ou texte
    if data is not None:
        s3 = _json("step3") or {}
        v["transpec"] = s3.get("transpec")
        v["jtr"] = s3.get("jtr")
        jt = s3.get("jt1_jt2")
        v["jt1"], v["jt2"] = (jt[0], jt[1]) if isinstance(jt, list) and len(jt) >= 2 \
            else (None, None)
        v["km"] = s3.get("km")
        v["pte"] = s3.get("pte(nr)")
        v["yic"] = s3.get("yic(nr)")
        if v["yic"] is None:
            v["yic"] = s3.get("center(1,nr)")
        cp = s3.get("cal_probe")
        if isinstance(cp, dict):       # clés JSON = str → normaliser en int
            cp = {int(k): {int(c): [float(x) for x in vals]
                           for c, vals in (inner or {}).items()}
                  for k, inner in cp.items()}
        v["calib_probe"] = cp or {}
    else:
        m = re.search(r"transpec\s*:\s*([\d.-]+)", txt)
        v["transpec"] = float(m.group(1)) if m else None
        m = re.search(r"jtr=(\d+)", txt)
        v["jtr"] = int(m.group(1)) if m else None
        m = re.search(r"jt1/jt2=\[(\d+),\s*(\d+)\]", txt)
        v["jt1"], v["jt2"] = (int(m.group(1)), int(m.group(2))) if m else (None, None)
        m = re.search(r"\bkm\s*:\s*(\d+)", txt)
        v["km"] = int(m.group(1)) if m else None
        m = re.search(r"pte\(nr\)\s*:\s*([\d.-]+)", txt)
        v["pte"] = float(m.group(1)) if m else None
        m = re.search(r"yic\(nr\)\s*:\s*([\d.-]+)", txt)
        v["yic"] = float(m.group(1)) if m else None
        if v["yic"] is None:
            m = re.search(r"center\(1,nr\)\s*:\s*([\d.-]+)", txt)
            v["yic"] = float(m.group(1)) if m else None
        probe: dict[int, dict[int, list[float]]] = {}
        for ln in txt.splitlines():
            m = re.search(r"cal i=(\d+).*?:(.*)", ln)
            if m and "canal1" in ln:
                icd = int(m.group(1))
                vals = [float(x) for x in re.findall(r"[\d.]+", m.group(2))]
                if len(vals) == 7:
                    probe[icd] = {1: vals}
        v["calib_probe"] = probe

    # ACDF2 (CSV ou .lis)
    v["acdf2"] = _read_acdf2(acdf2)

    # vitesses : miv.csv (lbdvel,xv,yv) ou .lis espacé
    vel = []
    if miv.is_file():
        for ln in miv.read_text(errors="ignore").splitlines():
            if ln.strip().startswith("lbdvel"):
                continue
            parts = [p for p in ln.replace(",", " ").split() if p.strip()]
            if len(parts) >= 3:
                try:
                    vel.append((int(float(parts[-3])), float(parts[-2]),
                                float(parts[-1])))
                except ValueError:
                    continue
    v["vitesses"] = vel
    return v


# --------------------------------------------------------------------------- #
# Comparaison + rapport
# --------------------------------------------------------------------------- #
def fmt_stat(kind: str, tol: float, val: float, ref: float) -> tuple[str, float, str]:
    """Statut selon type (rel/abs) et tolérance."""
    if kind == "rel":
        if ref == 0:
            return ("FAIL", abs(val), "FAIL") if abs(val) > tol else ("PASS", abs(val), "PASS")
        d = abs(val - ref) / abs(ref)
    else:
        d = abs(val - ref)
    if d <= tol:
        return "PASS", d, "PASS"
    return "WARN", d, "WARN"


def compare(f: dict, p: dict, tols: dict) -> list[dict]:
    """Compare les valeurs f (fortran) et p (python), retourne rows."""
    rows = []

    def row(key, ref, val, kind, tol):
        if ref is None or val is None:
            rows.append({"key": key, "ref": ref, "val": val, "diff": None,
                         "stat": "N/A"})
            return
        stat, d, _ = fmt_stat(kind, tol, float(val), float(ref))
        rows.append({"key": key, "ref": float(ref), "val": float(val),
                     "diff": d, "stat": stat})

    row("transpec", f.get("transpec"), p.get("transpec"), "rel", tols["transpec"])
    row("jtr", f.get("jtr"), p.get("jtr"), "abs", tols["jtr"])
    row("jt1", f.get("jt1"), p.get("jt1"), "abs", tols["jtr"])
    row("jt2", f.get("jt2"), p.get("jt2"), "abs", tols["jtr"])
    row("km", f.get("km"), p.get("km"), "abs", tols["km"])
    row("pte(nr)", f.get("pte"), p.get("pte"), "rel", tols["pte"])
    row("yic(nr)", f.get("yic"), p.get("yic"), "abs", tols["yic"])

    # ACDF2 : X tolerance px (cols 0-3), Y tolère moins (cols 4-7)
    if f.get("acdf2") is not None and p.get("acdf2") is not None:
        fa, pa = f["acdf2"], p["acdf2"]
        if fa.shape == pa.shape and fa.size:
            d = np.abs(fa - pa)
            rows.append({"key": "ACDF2 X max", "ref": 0.0,
                         "val": float(d[:, :4].max()), "diff": float(d[:, :4].max()),
                         "stat": "PASS" if d[:, :4].max() <= tols["acdf_x"] else "WARN"})
            rows.append({"key": "ACDF2 Y max", "ref": 0.0,
                         "val": float(d[:, 4:].max()), "diff": float(d[:, 4:].max()),
                         "stat": "PASS" if d[:, 4:].max() <= tols["acdf_y"] else "WARN"})
            rows.append({"key": "ACDF2 X mean", "ref": 0.0,
                         "val": float(d[:, :4].mean()), "diff": float(d[:, :4].mean()),
                         "stat": "PASS"})
            rows.append({"key": "ACDF2 Y mean", "ref": 0.0,
                         "val": float(d[:, 4:].mean()), "diff": float(d[:, 4:].mean()),
                         "stat": "PASS"})

    # vitesses : comparer par lbdvel commun
    fv = {lv: (xv, yv) for lv, xv, yv in f.get("vitesses", [])}
    pv = {lv: (xv, yv) for lv, xv, yv in p.get("vitesses", [])}
    common = sorted(set(fv) & set(pv))
    if not common:
        rows.append({"key": "vitesses (lbdvel communs)", "ref": None, "val": None,
                     "diff": None,
                     "stat": "N/A (lbdvel différents)"})
    for lv in common:
        fxv, fyv = fv[lv]; pxv, pyv = pv[lv]
        rows.append({"key": f"vit xv l={lv}", "ref": fxv, "val": pxv,
                     "diff": abs(pxv - fxv),
                     "stat": "PASS" if abs(pxv - fxv) <= tols["vel"] else "FAIL"})
        rows.append({"key": f"vit yv l={lv}", "ref": fyv, "val": pyv,
                     "diff": abs(pyv - fyv),
                     "stat": "PASS" if abs(pyv - fyv) <= 5 * tols["vel"] else "WARN"})

    # cal : comparaison des profils cal(i,j,1) aux 2 positions i (Fortran calib.ps
    #   pos 1 = x1cal, pos 2 = x2cal ; Python calib_probe{icd}).
    fcal, pcal = f.get("calib"), p.get("calib_probe")
    if not pcal:
        rows.append({"key": "cal", "ref": None, "val": None, "diff": None,
                     "stat": "N/A (log Python sans cal_probe)"})
    elif fcal and pcal:
        # associe les positions Python (croissantes) aux pos 1/2 Fortran
        p_icds = sorted(int(k) for k in pcal)
        for fidx, icd in ((1, p_icds[0] if len(p_icds) > 0 else None),
                          (2, p_icds[1] if len(p_icds) > 1 else None)):
            if icd is None:
                continue
            fr = fcal.get(fidx, {}).get(1)
            pv = pcal.get(icd, {}).get(1)
            if fr and pv and len(fr) == len(pv):
                dmax = max(abs(a - b) for a, b in zip(fr, pv))
                dmean = sum(abs(a - b) for a, b in zip(fr, pv)) / len(fr)
                rows.append({"key": f"cal i(pos{fidx}) max", "ref": 0.0,
                             "val": dmax, "diff": dmax,
                             "stat": "PASS" if dmax <= tols["cal"] else "WARN"})
                rows.append({"key": f"cal i(pos{fidx}) mean", "ref": 0.0,
                             "val": dmean, "diff": dmean,
                             "stat": "PASS" if dmean <= tols["cal"] / 2 else "WARN"})
            else:
                rows.append({"key": f"cal i(pos{fidx})", "ref": None, "val": None,
                             "diff": None, "stat": "N/A (log partiel)"})
    return rows


DEFAULT_TOLS = {
    "transpec": 0.005,   # rel
    "jtr": 0.5,          # abs px
    "km": 0,             # abs
    "pte": 0.015,         # rel (1.5% : géométrie exacte, pte matche ~1%)
    "yic": 0.2,           # abs px (centre au milieu ic, résidu sous-pixel)
    "acdf_x": 1.4,       # abs px
    "acdf_y": 0.3,       # abs px
    "vel": 1.0,          # abs px
    "cal": 0.15,         # abs (cal ~1 ; écart <15% du niveau)
}


def render(rows: list[dict]) -> str:
    def s(x) -> str:
        return "-" if x is None else f"{x:12.4g}"
    L = [f"{'grandeur':24s} {'fortran':>12s} {'python':>12s} {'diff':>10s}  stat"]
    L.append("-" * 74)
    for r in rows:
        L.append(f"{r['key']:24s} {s(r['ref'])} {s(r['val'])} {s(r['diff'])}"
                 f"  {r['stat']}")
    npass = sum(1 for r in rows if r["stat"].startswith("PASS"))
    nwarn = sum(1 for r in rows if r["stat"] == "WARN")
    nfail = sum(1 for r in rows if r["stat"] == "FAIL")
    L.append("-" * 74)
    L.append(f"Statut global : {npass} PASS, {nwarn} WARN, {nfail} FAIL"
             f"  →  {'VALIDÉ' if nfail == 0 else 'À REVOIR'}")
    return "\n".join(L) + "\n"


def last_run_dir(root: Path, prefix: str = "run_") -> Path | None:
    """Dernier sous-dossier run_NNN dans root (ordre numérique, pas mtime)."""
    runs = [d for d in root.glob(f"{prefix}*") if d.is_dir()]
    if not runs:
        return None
    def num(d: Path) -> int:
        try:
            return int(d.name.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            return -1
    return max(runs, key=num)


def main() -> int:
    ap = argparse.ArgumentParser(description="Parité Fortran↔Python (MSDP)")
    ap.add_argument("--fortran", help="dossier run Fortran (data/output/run_NNN)")
    ap.add_argument("--python", help="dossier run Python (new_python/outputs/run_NNN)")
    ap.add_argument("--out", default=None, help="fichier de rapport (défaut: stdout)")
    ap.add_argument("--tol", default=None, help="JSON de tolérances (optionnel)")
    args = ap.parse_args()

    data_out = _PROJ / "data/output"
    py_out = _PROJ / "new_python/outputs"
    fdir = Path(args.fortran) if args.fortran else last_run_dir(data_out)
    pdir = Path(args.python) if args.python else last_run_dir(py_out)
    if not fdir or not pdir:
        print("ERREUR: besoin d'un run Fortran et d'un run Python.")
        return 2

    def find_fr(d: Path, stem: str) -> Path:
        cands = [p for p in d.glob(f"{stem}*") if p.is_file()]
        return cands[0] if cands else (d / f"{stem}_001.lis")

    flog = find_fr(fdir, "ms_run_"); facd = find_fr(fdir, "ACDF2_run_")
    fmiv = find_fr(fdir, "miv_run_")
    # Python : privilégie les formats modernes (log .json, ACDF2.csv, miv.csv),
    # sinon rétrocompat (ms_run_*.lis, ACDF2.lis, miv.lis)
    def _pyfile(stem: str, *suffixes: str) -> Path:
        for sfx in suffixes:
            c = [p for p in pdir.glob(f"{stem}.{sfx}") if p.is_file()]
            if c:
                return c[0]
        c = [p for p in pdir.glob(f"{stem}*") if p.is_file()]
        return c[0] if c else pdir / f"{stem}.lis"

    plog = _pyfile("ms_run", "json", "lis")
    pacd = _pyfile("ACDF2", "csv", "lis")
    pmiv = _pyfile("miv", "csv", "lis")

    f = extract_fortran(flog, facd, fmiv)
    p = extract_python(plog, pacd, pmiv)
    tols = DEFAULT_TOLS
    if args.tol:
        import json
        try:
            tols = {**DEFAULT_TOLS, **json.loads(args.tol)}
        except Exception:
            pass

    rows = compare(f, p, tols)
    report = render(rows)
    header = ("# Parité MSDP Fortran ↔ Python\n"
              f"# Fortran : {fdir}\n"
              f"# Python  : {pdir}\n"
              + "-" * 74 + "\n")
    full = header + report

    if args.out:
        Path(args.out).write_text(full)
        print(f"Rapport écrit : {args.out}")
    else:
        print(full)
    nfail = sum(1 for r in rows if r["stat"] == "FAIL")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())