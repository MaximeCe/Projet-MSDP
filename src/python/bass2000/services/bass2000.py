"""
bass2000.py — Client BASS2000 pour le pipeline MSDP

Fonctions :
- get_observation_days(year): jours avec données pour une année
- get_sequences(date): liste des séquences d'une journée
- get_sequence_files(date, num_seq): fichiers d'une séquence
- find_best_calibration(date, num_seq): meilleure calibration (flat + dark)
- download_sequence(date, num_seq, dest): téléchargement complet
- download_with_calibration(date, num_seq, dest): obs + calibration
"""

import re, time, sys, os, json
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASS2000_BASE  = "https://bass2000.obspm.fr"
ARCHIVE_URL    = f"{BASS2000_BASE}/longterm_archive.php"
SEQ_JSON_URL   = f"{BASS2000_BASE}/longterm/getJsonSequenceObs.php"
FILE_INFO_URL  = f"{BASS2000_BASE}/longterm/get_fileinfo.php"
DATE_OBS_URL   = f"{BASS2000_BASE}/longterm/get_dateobs_data.php"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "DPSM-GUI/1.0 (MSDP pipeline)"})
SESSION.request = lambda method, url, **kw: requests.Session.request(
    SESSION, method, url, timeout=kw.pop("timeout", 300), **kw
)

# Cache: {year: [day_str, ...]}
_OBS_DAYS_CACHE: dict[int, list[str]] = {}
_OBS_MONTHS_CACHE: dict[int, list[int]] = {}

# ── Types ──────────────────────────────────────────────────────────
SEQ_OBSERVATION = "Observation"
SEQ_FLAT        = "Flat Field"
SEQ_DARK        = "Dark Current"

SEQ_TYPES = {
    "observation": SEQ_OBSERVATION,
    "flat":        SEQ_FLAT,
    "dark":        SEQ_DARK,
}


# ── API BASS2000 ───────────────────────────────────────────────────

def get_observation_days(year: int) -> list[str]:
    """Retourne la liste des dates (YYYY-MM-DD) avec données pour une année."""
    if year in _OBS_DAYS_CACHE:
        return _OBS_DAYS_CACHE[year]
    try:
        resp = SESSION.get(DATE_OBS_URL, params={"year": str(year), "instrume": "dpsm"})
        resp.raise_for_status()
        data = resp.json()
        days = []
        for point in data.get("points", []):
            ts_ms = point[0]
            dt = datetime.fromtimestamp(ts_ms / 1000)
            days.append(dt.strftime("%Y-%m-%d"))
        days.sort()
        _OBS_DAYS_CACHE[year] = days
        return days
    except Exception:
        return []


def get_observation_months(year: int) -> list[int]:
    """Retourne les mois (1-12) qui ont des données pour une année."""
    if year in _OBS_MONTHS_CACHE:
        return _OBS_MONTHS_CACHE[year]
    days = get_observation_days(year)
    months = sorted(set(int(d[5:7]) for d in days))
    _OBS_MONTHS_CACHE[year] = months
    return months


def get_days_in_month(year: int, month: int) -> list[str]:
    """Retourne les jours (YYYY-MM-DD) avec données pour un mois donné."""
    all_days = get_observation_days(year)
    return [d for d in all_days if int(d[5:7]) == month]


def clear_cache():
    """Vide le cache des jours/mois d'observation."""
    _OBS_DAYS_CACHE.clear()
    _OBS_MONTHS_CACHE.clear()


def get_sequences(date: str, hour: int = 0) -> list[dict]:
    """Récupère la liste des séquences pour une date."""
    try:
        resp = SESSION.post(ARCHIVE_URL, params={"instrume": "dpsm"},
                            data={"dategreg": date, "hour": str(hour), "Find": "Find"})
        resp.raise_for_status()
    except Exception as e:
        raise ConnectionError(f"Impossible de contacter BASS2000: {e}")

    sequences = []
    pattern = re.compile(
        r"<TR>(?:\s*)<TD>(\d+)</TD>"
        r"(?:\s*)<TD>(.+?)</TD>"
        r"(?:\s*)<TD>(\d+:\d+:\d+)</TD>"
        r"(?:\s*)<TD>(\d+:\d+:\d+)</TD>"
        r"(?:\s*)<TD>(\d+)</TD>"
        r"(?:\s*)</TR>",
        re.IGNORECASE | re.DOTALL
    )
    for m in pattern.finditer(resp.text):
        sequences.append({
            "num_seq": int(m.group(1)),
            "type":    m.group(2).strip(),
            "start":   m.group(3),
            "end":     m.group(4),
            "nfiles":  int(m.group(5)),
        })
    return sequences


def get_sequence_files(date: str, num_seq: int, hour: int = 0) -> list[dict]:
    """Récupère la liste des fichiers FITS d'une séquence (API JSON)."""
    try:
        resp = SESSION.get(SEQ_JSON_URL, params={
            "date": date, "instrume": "dpsm", "hour": str(hour), "numseq": str(num_seq),
        })
        resp.raise_for_status()
    except Exception as e:
        raise ConnectionError(f"Impossible de lister les fichiers: {e}")

    data = resp.json()
    files = []
    for row in data.get("data", []):
        dl_cell   = row[4] if len(row) > 4 else ""
        id_match  = re.search(r"Download\((\d+)", dl_cell)
        file_id   = int(id_match.group(1)) if id_match else None
        header_cell = row[5] if len(row) > 5 else ""
        path_match  = re.search(r"getFitsHeader\('([^']+)','([^']+)'\)", header_cell)
        files.append({
            "time":     row[0] if len(row) > 0 else "",
            "content":  row[1] if len(row) > 1 else "",
            "file_id":  file_id,
            "url_path": path_match.group(1) if path_match else None,
            "filename": path_match.group(2) if path_match else None,
        })
    return files


def get_download_url(file_id: int) -> Optional[str]:
    """Appelle l'API get_fileinfo pour obtenir l'URL directe de téléchargement."""
    try:
        resp = SESSION.get(FILE_INFO_URL, params={"id": file_id, "instrume": "dpsm"})
        resp.raise_for_status()
        m = re.search(r'href="([^"]+\.fit[^"]*)"', resp.text)
        if m:
            url = m.group(1)
            if not url.startswith("http"):
                url = BASS2000_BASE + url
            return url
    except Exception:
        pass
    return None


def _download_one_file(args: tuple) -> tuple[str, bool]:
    """Télécharge un fichier unique (appelé par ThreadPoolExecutor)."""
    url, filepath = args
    if filepath.exists():
        return str(filepath), True
    try:
        resp = SESSION.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return str(filepath), True
    except Exception:
        # Nettoyer fichier partiel
        if filepath.exists():
            filepath.unlink()
        return str(filepath), False


# ── Calibration matching ───────────────────────────────────────────

def _time_to_minutes(t: str) -> int:
    """Convertit HH:MM:SS en minutes depuis minuit."""
    parts = t.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def find_best_calibration(date: str, obs_num_seq: int,
                          sequences: list[dict]) -> dict:
    """
    Trouve les meilleures séquences de calibration (flat + dark) pour une
    observation donnée, en sélectionnant la plus proche temporellement
    (avant ou après) pour chaque type.

    Retourne: {flat: {num_seq, start, nfiles, delta_min},
               dark: {num_seq, start, nfiles, delta_min}}
    """
    obs_seq = next((s for s in sequences if s["num_seq"] == obs_num_seq), None)
    if not obs_seq:
        return {"flat": None, "dark": None}

    obs_start = _time_to_minutes(obs_seq["start"])
    best_flat = best_dark = None
    best_flat_delta = best_dark_delta = float("inf")

    for s in sequences:
        if s["num_seq"] == obs_num_seq:
            continue
        delta = abs(_time_to_minutes(s["start"]) - obs_start)
        if s["type"] == SEQ_FLAT and delta < best_flat_delta:
            best_flat_delta = delta
            best_flat = s
        if s["type"] == SEQ_DARK and delta < best_dark_delta:
            best_dark_delta = delta
            best_dark = s

    result = {}
    if best_flat:
        result["flat"] = {**best_flat, "delta_min": best_flat_delta}
    if best_dark:
        result["dark"] = {**best_dark, "delta_min": best_dark_delta}
    return result


# ── Téléchargement ─────────────────────────────────────────────────

def download_sequence(date: str, num_seq: int, seq_info: dict,
                       dest: Path, progress_callback=None,
                       limit: Optional[int] = None,
                       max_workers: int = 4) -> int:
    """
    Télécharge les fichiers d'une séquence en parallèle.
    Retourne le nombre de fichiers téléchargés.
    """
    files = get_sequence_files(date, num_seq)
    if not files:
        return 0
    if limit:
        files = files[:limit]

    # Construire les URLs et chemins
    download_tasks = []
    for f in files:
        ts = f["time"].replace(":", "")
        content_slug = re.sub(r'[^a-zA-Z0-9_-]', '', f["content"].replace(" ", "_"))
        name = f"{date}_{ts}_{content_slug}.fit"
        filepath = dest / name

        dl_url = None
        if f["file_id"]:
            dl_url = get_download_url(f["file_id"])
        elif f["url_path"] and f["filename"]:
            dl_url = f["url_path"].rstrip("/") + "/" + f["filename"]

        if dl_url:
            download_tasks.append((dl_url, filepath))

    # Téléchargement parallèle
    ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one_file, t): t for t in download_tasks}
        for future in as_completed(futures):
            _, success = future.result()
            if success:
                ok += 1
            if progress_callback:
                progress_callback()
            time.sleep(0.05)  # politesse légère

    return ok


def download_with_calibration(date: str, obs_num_seq: int,
                               sequences: list[dict],
                               dest: Path,
                               progress_callback=None,
                               obs_limit: Optional[int] = None,
                               max_workers: int = 4) -> dict:
    """
    Télécharge une observation + les meilleures calibrations (flat + dark).

    Retourne le nombre total de fichiers téléchargés.
    """
    result = {}
    obs_info = next((s for s in sequences if s["num_seq"] == obs_num_seq), None)
    if not obs_info:
        return result

    # Télécharger l'observation
    obs_dir = dest / f"obs_{obs_num_seq:03d}"
    obs_dir.mkdir(parents=True, exist_ok=True)
    n = download_sequence(date, obs_num_seq, obs_info, obs_dir,
                          progress_callback, obs_limit, max_workers)
    result["obs"] = {"num_seq": obs_num_seq, "files": n, "dir": str(obs_dir)}

    # Trouver et télécharger les calibrations
    calib = find_best_calibration(date, obs_num_seq, sequences)
    for ctype in ("flat", "dark"):
        if ctype in calib and calib[ctype]:
            c = calib[ctype]
            c_dir = dest / f"{ctype}_{c['num_seq']:03d}"
            c_dir.mkdir(parents=True, exist_ok=True)
            n = download_sequence(date, c["num_seq"], c, c_dir,
                                  progress_callback, None, max_workers)
            result[ctype] = {"num_seq": c["num_seq"], "files": n,
                            "dir": str(c_dir), "delta_min": c["delta_min"]}

    return result