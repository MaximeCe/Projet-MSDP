#!/usr/bin/env python3
"""
Téléchargement automatique des données DPSM depuis l'archive BASS2000.

Utilisation :
    python scripts/download_bass2000.py --date 2015-06-04                    # Liste les séquences
    python scripts/download_bass2000.py --date 2015-06-04 --seq 3            # Télécharge la séquence #3
    python scripts/download_bass2000.py --date 2015-06-04 --type flat        # Télécharge les Flat Field
    python scripts/download_bass2000.py --date 2015-06-04 --type dark        # Télécharge les Dark Current
    python scripts/download_bass2000.py --date 2015-06-04 --type observation # Télécharge les Observations
    python scripts/download_bass2000.py --date 2015-06-04 --all              # Télécharge TOUT
    python scripts/download_bass2000.py --years                             # Liste les années disponibles
"""

import argparse, sys, os, re, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.bass2000 import (
    get_observation_days, get_sequences, get_sequence_files, get_download_url,
    find_best_calibration, download_sequence, download_with_calibration,
    SEQ_TYPES,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def list_sequences(sequences: list[dict]):
    print(f"\n{'#':>4}  {'Type':<20} {'Début':<10} {'Fin':<10} {'Fichiers':>8}")
    print("-" * 60)
    for s in sequences:
        print(f"{s['num_seq']:>4}  {s['type']:<20} {s['start']:<10} {s['end']:<10} {s['nfiles']:>8}")


def main():
    parser = argparse.ArgumentParser(
        description="Téléchargement automatique des données DPSM (MSDP) depuis BASS2000"
    )
    parser.add_argument("--date", help="Date YYYY-MM-DD")
    parser.add_argument("--hour", type=int, default=0, help="Heure début (défaut=0)")
    parser.add_argument("--seq", type=int, help="Numéro de séquence")
    parser.add_argument("--type", choices=list(SEQ_TYPES.keys()) + ["all"],
                        help="Type : flat, dark, observation, all")
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Dossier destination (défaut: data/)")
    parser.add_argument("--limit", type=int, help="Max fichiers par séquence")
    parser.add_argument("--list-only", action="store_true", help="Afficher sans télécharger")
    parser.add_argument("--years", action="store_true", help="Lister les années disponibles")
    parser.add_argument("--days", action="store_true", help="Lister les jours de l'année")
    args = parser.parse_args()

    # Années disponibles
    if args.years:
        print("\n📅 Années avec données DPSM :")
        for y in range(2013, 2019):
            days = get_observation_days(y)
            if days:
                print(f"   {y}: {len(days)} jours d'observation")
        return

    # Jours d'une année
    if args.days and args.date:
        year = int(args.date[:4])
        days = get_observation_days(year)
        print(f"\n📅 {len(days)} jours avec données en {year} :")
        for d in days:
            print(f"   • {d}")
        return

    # Pas de date -> aide
    if not args.date:
        parser.print_help()
        return

    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Interrogation BASS2000 — {args.date}")
    sequences = get_sequences(args.date, args.hour)
    if not sequences:
        print("❌ Aucune séquence trouvée.")
        sys.exit(1)

    list_sequences(sequences)

    # Déterminer les séquences à traiter
    if args.type == "all":
        selected = sequences
    elif args.type:
        label = SEQ_TYPES[args.type]
        selected = [s for s in sequences if s["type"] == label]
        print(f"\n🔎 {label} → {len(selected)} séquence(s)")
    elif args.seq:
        selected = [s for s in sequences if s["num_seq"] == args.seq]
        if not selected:
            print(f"❌ Séquence #{args.seq} introuvable."); sys.exit(1)
    else:
        selected = []

    if args.list_only or not selected:
        print("\n💡 Ajoutez --seq N, --type flat/dark/observation ou --all pour télécharger.")
        return

    total = 0
    for s in selected:
        if s["num_seq"] == selected[0]["num_seq"]:
            # Pour la première séquence, proposer la calibration
            calib = find_best_calibration(args.date, s["num_seq"], sequences)
            print(f"\n📊 Calibration associée :")
            if "flat" in calib:
                print(f"   Flat #{calib['flat']['num_seq']} à T{calib['flat']['delta_min']:+d} min")
            if "dark" in calib:
                print(f"   Dark #{calib['dark']['num_seq']} à T{calib['dark']['delta_min']:+d} min")
        total += download_sequence(args.date, s["num_seq"], s, dest, limit=args.limit)
        time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"🏁 Terminé — {total} fichiers dans {dest}")


if __name__ == "__main__":
    main()