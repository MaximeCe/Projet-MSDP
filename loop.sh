#!/usr/bin/env bash
# loop.sh — Boucle d'ingestion MSDP : tirage aléatoire d'une séquence flat+dark,
# exécution des deux pipelines (Fortran + Python), CSV de résultats, sorties
# versionnées, et cleanup complet avant de repasser au téléchargement.
#
# Chaque itération :
#   1. tire aléatoirement une (date, séquence flat, séquence dark appariée) non
#      encore traitée dans CE run (anti-doublon)  [src/python/bass2000/pick_pair.py]
#   2. télécharge uniquement ces 2 séquences (download_bass2000.py --seq N)
#   3. lance ingest_all.py --local (auto-params → run_both_pipeline →
#      checks → ligne CSV + sorties versionnées) ; NB: --local ne télécharge pas,
#      il consomme ce qui est dans data/input
#   4. cleanup : vide les *.fit téléchargés → retour au téléchargement
#
# Usage :
#   ./loop.sh N [--keep] [--csv chemin.csv] [--seed S]
#     N      : nombre d'itérations de boucle (obligatoire)
#     --keep : ne PAS supprimer les données téléchargées après ingestion
#     --csv  : chemin CSV (défaut: data/output/ingest.csv)
#     --seed : graine aléatoire pour le tirage (test/reproductibilité)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${PROJECT_DIR}/../.venv/bin/python"
PICKER="${PROJECT_DIR}/src/python/bass2000/pick_pair.py"
DOWNLOADER="${PROJECT_DIR}/src/python/bass2000/download_bass2000.py"
DATA_INPUT="${PROJECT_DIR}/data/input"
DATA_OUTPUT="${PROJECT_DIR}/data/output"

[[ $# -ge 1 ]] || { echo "Usage: $0 <N> [--keep] [--csv F] [--seed S]"; exit 2; }
ITER="${1:-1}"; shift

KEEP=0; CSV="${DATA_OUTPUT}/ingest.csv"; SEED=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep)  KEEP=1 ;;
        --csv)   CSV="${2:?--csv requiert un chemin}"; shift ;;
        --seed)  SEED="${2:?--seed requiert une valeur}"; shift ;;
        *) echo "Argument inconnu: $1"; exit 2 ;;
    esac
    shift
done

# Fichier temporaire des couples déjà traités dans CE run (anti-doublon).
USED_FILE="$(mktemp)"; trap 'rm -f "$USED_FILE"' EXIT
USED_ARGS=()
load_used() {
    USED_ARGS=()
    while IFS= read -r k && [[ -n "$k" ]]; do USED_ARGS+=(--used "$k"); done < "$USED_FILE"
}

mkdir -p "$DATA_OUTPUT"

for ((i=1; i<=ITER; i++)); do
    echo "=========================================================="
    echo "  ITÉRATION ${i}/${ITER}"
    echo "=========================================================="

    # ---- 1. Tirage aléatoire (anti-doublon sur ce run) ----
    load_used
    pick_args=( "${USED_ARGS[@]}" )
    [[ -n "$SEED" ]] && pick_args+=(--seed "$((SEED + i))")
    PICK="$("$VENV" "$PICKER" "${pick_args[@]}")" \
        || { echo "  ❌ Plus de paire disponible — arrêt."; break; }
    read -r DATE FSEQ DSEQ <<< "$PICK"
    echo "  → Donnée tirée : date=$DATE  flat_seq=$FSEQ  dark_seq=$DSEQ"
    # Marquage immédiat (même si le run échoue, on ne reteste pas 2× ce run).
    echo "${DATE}/${FSEQ}" >> "$USED_FILE"

    # ---- 2. Téléchargement sélectif flat + dark ----
    echo "  → Téléchargement flat (#${FSEQ})..."
    "$VENV" "${DOWNLOADER}" --date "$DATE" --seq "$FSEQ" --type flat --dest "$DATA_INPUT" >/dev/null
    echo "  → Téléchargement dark  (#${DSEQ})..."
    "$VENV" "${DOWNLOADER}" --date "$DATE" --seq "$DSEQ" --type dark --dest "$DATA_INPUT" >/dev/null

    # ---- 3. Ingestion complète (auto-params → 2 pipelines → CSV + versioning) ----
    echo "  → Pipeline MSDP (Fortran + Python) sur ${DATE}"
    INGEST_ARGS=(--local --dates "$DATE" --csv "$CSV" --keep)
    "$VENV" "$PROJECT_DIR/ingest_all.py" "${INGEST_ARGS[@]}"

    # ---- 4. Cleanup : retour au téléchargement (sauf --keep) ----
    if [[ "$KEEP" == "0" ]]; then
        echo "  → Cleanup : données ${DATE} supprimées."
        rm -f "$DATA_INPUT"/m*_stack_bass2000_*1.fit "${DATA_INPUT}"/*.fit
    fi
    echo ""
done

echo "✅ Boucle terminée (${N} itérations)."
echo "   CSV : ${CSV}"
echo "   Couples traités :"
sort -u "$USED_FILE" | sed 's/^/   - /'