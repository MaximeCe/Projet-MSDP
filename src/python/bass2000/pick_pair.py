#!/usr/bin/env python3
"""
pick_pair.py — Tire aléatoirement un couple (date, séquence flat, séquence dark)
à traiter par la boucle loop.sh.

Granularité "séquence flat+dark appariées" (choix utilisateur) :
  1. date aléatoire parmi les jours d'observation BASS2000 (DPSM)
  2. séquence "Flat Field" aléatoire de cette date
  3. sa "Dark Current" la plus proche temporellement sur la même date
  4. refuse les (date, flat_seq) déjà embarqués dans le run (anti-doublon)

Sortie : une ligne "DATE FLAT_SEQ DARK_SEQ" sur stdout, ou code de sortie 1
si aucune paire disponible (dates finies / toutes déjà traitées).
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from services.bass2000 import (          # noqa: E402
    get_observation_days,
    get_sequences,
    SEQ_FLAT,
    SEQ_DARK,
)

YEARS = range(2013, 2019)  # années avec données DPSM (voir download_bass2000.py)


def _mins(t: str) -> int:
    h, m, s = t.split(":")
    return int(h) * 60 + int(m) + int(s) / 60


def _nearest_dark(sequences, flat) -> dict | None:
    """Renvoie la dark la plus proche temporellement de la flat choisie."""
    darks = [s for s in sequences if s["type"] == SEQ_DARK]
    if not darks:
        return None
    t0 = _mins(flat["start"])
    return min(darks, key=lambda d: abs(_mins(d["start"]) - t0))


def pick(used: set[str], max_attempts: int = 60) -> tuple | None:
    """Tire une (date, flat_seq, dark_seq) non encore traitée.

    `used` : set de chaînes "DATE/FLAT_SEQ" déjà traitées dans le run.
    """
    for _ in range(max_attempts):
        year = random.choice(list(YEARS))
        days = get_observation_days(year)
        if not days:
            continue
        date = random.choice(days)
        sequences = get_sequences(date)
        if not sequences:
            continue
        flats = [s for s in sequences if s["type"] == SEQ_FLAT]
        if not flats:
            continue
        random.shuffle(flats)
        for flat in flats:
            key = f"{date}/{flat['num_seq']}"
            if key in used:
                continue
            dark = _nearest_dark(sequences, flat)
            if dark is None:
                continue
            return date, flat["num_seq"], dark["num_seq"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--used", nargs="+", default=[],
                    help="couples 'DATE/FLAT_SEQ' déjà traités (anti-doublon)")
    ap.add_argument("--seed", type=int, default=None, help="graine aléatoire (test)")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    used = set(args.used)
    result = pick(used)
    if result is None:
        print("AUCUNE_PAIRE_DISPONIBLE", file=sys.stderr)
        sys.exit(1)
    date, fseq, dseq = result
    print(f"{date} {fseq} {dseq}")


if __name__ == "__main__":
    main()
