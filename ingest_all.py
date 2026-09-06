#!/usr/bin/env python3
"""
ingest_all.py — Orchestrateur MSDP : téléchargement → auto-params → run des 2
pipelines → validation → CSV → cleanup.

C'est le Volet B de la spec multi-instrument. Pour chaque date :
  1. (option) télécharger les données depuis BASS2000 (download_bass2000.py)
  2. estimer les paramètres (auto_params.estimate_params) → écrire ms.par.auto
     et ms.yml.auto dans un répertoire de travail dédié
  3. boucle de correction : si la géométrie échoue, tester quelques jeux de ja
     (et mingrad) et garder celui qui passe les checks (ou le meilleur)
  4. lancer run_pipeline.sh + run_pipeline_py.sh avec les configs
  5. valider les 2 ACDF2 (check_fortran_pipeline.py / check_python_pipeline.py)
  6. écrire une ligne dans le CSV (date, instr, im, jm, nm, ja, mingrad,
     check_fortran, check_python, ecart_fp, statut)
  7. (option) supprimer les .fit de la date

Usage :
    python3 ingest_all.py [--local] [--csv out.csv] [--keep] [--dry-run]
      --local    : traite les données déjà présentes dans data/input/ (pas de téléchargement)
      --csv      : chemin du CSV de sortie (défaut: data/output/ingest.csv)
      --keep     : ne pas supprimer les données après ingestion
      --dry-run  : ne rien exécuter, afficher le plan
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent   # ingest_all.py est à la racine du projet
SRC_PYTHON = PROJECT_DIR / "src" / "python"
DATA_INPUT = PROJECT_DIR / "data" / "input"
DATA_OUTPUT = PROJECT_DIR / "data" / "output"

sys.path.insert(0, str(SRC_PYTHON))
import auto_params  # noqa: E402


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def run_cmd(cmd, cwd=None, timeout=400, quiet=False):
    """Exécute une commande, retourne (returncode, stdout+stderr)."""
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout)
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode, out
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def find_flat_dir(dirpath: Path) -> str | None:
    """Cherche un fichier flat dans le répertoire.

    Deux conventions :
      - pipeline : *y1.fit
      - downloader BASS2000 : *_Flat_Field.fit
    """
    patterns = ["*y1.fit", "*Flat_Field.fit", "*flat*.fit"]
    for pat in patterns:
        for f in sorted(dirpath.glob(pat)):
            # ignorer les fichiers à 0 octet (fichiers vides côté BASS2000)
            if f.stat().st_size > 0:
                return str(f)
    return None


def prepare_pipeline_links(input_dir: Path):
    """Crée des symlinks en convention pipeline (m..._x1.fit / m..._y1.fit)
    pointant vers les fichiers BASS2000 téléchargés (Flat_Field / Dark_Current).

    ms1.f fait `ls m*x1.fit` / `ls m*y1.fit` : sans cette conversion, les flats
    et darks téléchargés (nommés *_Flat_Field.fit / *_Dark_Current.fit) ne sont
    pas vus par le pipeline. Retourne (nb_links_crees, [fichiers 0 octet ignorés]).
    """
    created = []
    empty = []
    # regrouper les fichiers par type
    for i, f in enumerate(sorted(input_dir.glob("*.fit"))):
        if f.stat().st_size == 0:
            empty.append(f.name)
            continue
        low = f.name.lower()
        # déterminer le type (flat ou dark) par le nom BASS2000
        if "flat" in low or "y1" in low:
            link = input_dir / f"m{100+i:03d}_stack_bass2000_y1.fit"
        elif "dark" in low or "x1" in low:
            link = input_dir / f"m{100+i:03d}_stack_bass2000_x1.fit"
        else:
            continue  # observation, pas de lien pipeline
        if not link.exists():
            link.symlink_to(f.name)
            created.append(link.name)
    return created, empty


# ---------------------------------------------------------------------------
# Boucle de correction ja/mingrad
# ---------------------------------------------------------------------------
def correction_boucle(wkdir: Path, params, max_tries=15):
    """Teste plusieurs jeux de params (ja, mingrad) et garde le meilleur.

    Pour chaque candidat : écrit ms.par.auto + ms.yml.auto, lance run_both,
    check les 2 ACDF2. Retourne (params_ok, best_params, n_tries).
    """
    best = params
    best_score = None
    # variations de ja autour de l'estimation, puis de mingrad
    base_ja = params.ja
    candidates = [params]  # le premier = l'estimation initiale (dérivée)
    ja_step = max(2, int(params.jm * 0.03))
    # ⚠️ 2026-09-06 : la grille faisait varier uniquement ja1/ja3 (anti-corrélés)
    # et JAISSAT ja2 figé — elle ne pouvait jamais atteindre une config valide
    # où ce sont ja2/ja3 qui comptent (ex. Meudon [100,501,851]). On couvre
    # maintenant les 3 coupes de façon indépendante : chaque coupe ±1 et ±2 pas.
    for j_i in range(3):
        for d in (-2 * ja_step, -ja_step, ja_step, 2 * ja_step):
            ja = list(base_ja)
            ja[j_i] = max(2, min(params.jm - 2, ja[j_i] + d))
            p = auto_params.Params(params.im, params.jm, params.nm,
                                   ja, params.interc, params.mingrad,
                                   params.xdel, params.jtriple)
            candidates.append(p)
    mg_alt = params.mingrad - 6 if params.mingrad > 6 else params.mingrad
    if mg_alt != params.mingrad:
        p = auto_params.Params(params.im, params.jm, params.nm, params.ja,
                               params.interc, mg_alt, params.xdel,
                               params.jtriple)
        candidates.append(p)

    stat_rows = []
    ok_params = None
    best_files = (None, None)  # (fortran_path, python_path) du meilleur run
    best_ecart = (None, None)
    for i, cand in enumerate(candidates[:max_tries]):
        wr = run_pipeline_with(wkdir, cand)
        if wr is None:
            continue
        f_ok, p_ok, fortran_ac, python_ac = wr
        # score : nb de checks passés (0..2) ; on veut 2
        score = (1 if f_ok else 0) + (1 if p_ok else 0)
        stat_rows.append((cand, score, f_ok, p_ok))
        if best_score is None or score > best_score:
            best_score, best = score, cand
            best_files = (fortran_ac, python_ac)
            best_ecart = compute_ecart(fortran_ac, python_ac)
        if score == 2:
            ok_params = cand
            break  # trouvé une config valide

    return ok_params, best, len(stat_rows), best_ecart, best_files


def run_pipeline_with(wkdir: Path, cand):
    """Écrit les configs pour un candidat et lance run_both.
    Retourne (f_ok, p_ok) : True si le check du pipeline correspondant passe."""
    ms_par = wkdir / "ms.par.auto"
    ms_yml = wkdir / "ms.yml.auto"
    cand.write_ms_par(str(ms_par))
    cand.write_ms_yml(str(ms_yml))

    # lancer les pipelines avec les configs auto
    rc_f, out_f = run_cmd([str(PROJECT_DIR / "run_pipeline.sh"), str(ms_par)],
                          cwd=str(PROJECT_DIR))
    rc_p, out_p = run_cmd([str(PROJECT_DIR / "run_pipeline_py.sh"), str(ms_yml)],
                          cwd=str(PROJECT_DIR))

    # find ACDF2 produits — numérotation désormais INDÉPENDANTE (2026-09-06) :
    # Fortran = ACDF2_run_*.lis (sans _py_) ; Python = ACDF2_run_py_*.lis.
    # (Avant on prenait les 2 plus récents par mtime, ce qui présupposait une
    # numérotation partagée — désormais on repère par NOM de préfixe.)
    def _latest(globpat):
        hits = sorted(DATA_OUTPUT.glob(globpat), key=lambda p: p.stat().st_mtime,
                      reverse=True)
        return hits[0] if hits else None

    fortran_ac = _latest("ACDF2_run_*.lis") if rc_f == 0 else None
    python_ac = _latest("ACDF2_run_py_*.lis") if rc_p == 0 else None
    f_ok = p_ok = False
    if fortran_ac is not None:
        rf, _ = run_cmd(["python3", str(PROJECT_DIR / "check_fortran_pipeline.py"),
                         str(fortran_ac), str(ms_par)])
        f_ok = (rf == 0)
    if python_ac is not None:
        rp, _ = run_cmd(["python3", str(PROJECT_DIR / "check_python_pipeline.py"),
                         str(python_ac), str(ms_yml)])
        p_ok = (rp == 0)
    return f_ok, p_ok, fortran_ac, python_ac


def compute_ecart(fortran_path, python_path):
    """Écart max et moyen entre les ACDF2 Fortran et Python (même grille canaux).

    ACDF2.lis : une ligne = un canal [A_X, C_X, D_X, F_X, A_Y, C_Y, D_Y, F_Y].
    On compare canal par canal, champ par champ. Retourne (ecart_max, ecart_moy).
    Si les fichiers n'existent pas ou ont un nombre de canaux différent, (None, None).
    """
    def load(path):
        if not path or not Path(path).exists():
            return None
        rows = []
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                return None
        return rows

    fr, py = load(fortran_path), load(python_path)
    if not fr or not py or len(fr) != len(py):
        return None, None
    diffs = []
    for rowf, rowp in zip(fr, py):
        if len(rowf) != len(rowp):
            return None, None
        for a, b in zip(rowf, rowp):
            diffs.append(abs(a - b))
    if not diffs:
        return None, None
    return max(diffs), sum(diffs) / len(diffs)


# ---------------------------------------------------------------------------
# Une date
# ---------------------------------------------------------------------------
def process_date(date: str, local: bool, keep: bool, dry: bool, csv_writer):
    wkdir = Path(tempfile.mkdtemp(prefix=f"ingest_{date}_"))

    # 1. Données : local (dans data/input) ou téléchargement
    # Dans TOUS les cas, convertir les noms BASS2000 (*_Flat_Field/_Dark_Current)
    # en conventions pipeline (m..._y1/x1.fit) attendues par run_pipeline*.sh.
    # ⚠️ Au préalable retirer les liens périmés pour ne pas mixer plusieurs dates.
    for stale in DATA_INPUT.glob("m*_stack_bass2000_*1.fit"):
        if stale.is_symlink():
            stale.unlink(missing_ok=True)
    created, empty = prepare_pipeline_links(DATA_INPUT)
    if empty:
        print(f"  ⚠  {date}: {len(empty)} fichier(s) 0 octet ignoré(s) "
              f"(problème BASS2000) : {empty[:3]}...")
    if created:
        print(f"  → {len(created)} liens pipeline créés (dont "
              f"{sum(1 for c in created if '_y1.' in c)} flats)")

    if local:
        flat = find_flat_dir(DATA_INPUT)
        if not flat:
            print(f"  ⚠  {date}: pas de flat dans data/input, skip")
            return None
    else:
        # téléchargement via download_bass2000.py
        print(f"  → Téléchargement {date}...")
        rc, out = run_cmd(["python3",
                           str(SRC_PYTHON / "bass2000" / "download_bass2000.py"),
                           "--date", date, "--type", "all", "--dest", str(DATA_INPUT)],
                          timeout=1800)
        if rc != 0:
            print(f"  ❌ Téléchargement {date} échoué")
            return None
        # convertir les noms BASS2000 en convention pipeline (m..._y1/x1.fit)
        created, empty = prepare_pipeline_links(DATA_INPUT)
        if empty:
            print(f"  ⚠  {date}: {len(empty)} fichier(s) 0 octet ignoré(s) "
                  f"(problème BASS2000) : {empty[:3]}...")
        if created:
            print(f"  → {len(created)} liens pipeline créés (dont "
                  f"{sum(1 for c in created if '_y1.' in c)} flats)")
        flat = find_flat_dir(DATA_INPUT)
        if not flat:
            print(f"  ❌ {date}: aucun flat après téléchargement")
            return None

    # 2. Estimation des paramètres
    print(f"  → Estimation paramètres depuis {Path(flat).name}...")
    params = auto_params.estimate_params(flat)

    # 3. Boucle de correction
    if dry:
        print(f"  [dry-run] {date}: nm={params.nm}, im={params.im}, "
              f"jm={params.jm}, ja={params.ja}, mingrad={params.mingrad}")
        return None

    ok, best, n_tries, (ecart_max, ecart_moy), (fortran_ac, python_ac) = \
        correction_boucle(wkdir, params)

    # 4. Écriture CSV
    row = {
        "date": date,
        "instr": "meudon" if params.im == 1536 else "other",
        "im": params.im, "jm": params.jm, "nm": params.nm,
        "ja1": best.ja[0], "ja2": best.ja[1], "ja3": best.ja[2],
        "mingrad": best.mingrad, "interc": best.interc,
        "check_fortran": "ok" if (ok) else "fail",
        "check_python": "ok" if (ok) else "fail",
        "ecart_max": (f"{ecart_max:.3f}" if ecart_max is not None else ""),
        "ecart_moy": (f"{ecart_moy:.3f}" if ecart_moy is not None else ""),
        "n_tries": n_tries,
        "statut": "ok" if ok else "best-effort",
    }
    csv_writer.writerow(row)
    print(f"  ✓ {date}: nm={params.nm}, statut={row['statut']} "
          f"(tries={n_tries}, écart max={row['ecart_max']})")

    # 5. Cleanup (sauf --keep et --dry)
    if not keep or True:
        pass
    shutil.rmtree(wkdir, ignore_errors=True)
    if not keep and not local:
        # supprime les .fit téléchargés pour cette date
        for f in DATA_INPUT.glob("*.fit"):
            f.unlink(missing_ok=True)
        print(f"  → Données {date} supprimées (--keep pour garder)")
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true",
                    help="traiter data/input/ sans téléchargement")
    ap.add_argument("--csv", default=str(DATA_OUTPUT / "ingest.csv"))
    ap.add_argument("--keep", action="store_true", help="ne pas supprimer les données")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dates", nargs="*", help="dates à traiter")
    args = ap.parse_args()

    DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv)
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "date", "instr", "im", "jm", "nm", "ja1", "ja2", "ja3",
            "mingrad", "interc", "check_fortran", "check_python",
            "ecart_max", "ecart_moy", "n_tries",
            "statut"])
        if write_header:
            writer.writeheader()

        dates = args.dates if args.dates else ["local"]
        for date in dates:
            print(f"\n=== Ingestion {date} ===")
            process_date(date, local=args.local, keep=args.keep,
                         dry=args.dry_run, csv_writer=writer)

    print(f"\nCSV : {csv_path}")


if __name__ == "__main__":
    main()