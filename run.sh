#!/usr/bin/env bash
# run.sh — Exécute ms1.py puis ms2.py dans un répertoire de travail propre
# Usage: ./run.sh [step1|step2|all] [dark_file] [flat_file]
#
# Crée un dossier data/work/ avec les symlinks nécessaires et lance le code
# depuis là, pour que les chemins relatifs de ms1.py/ms2.py fonctionnent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-/home/max/nextcloud/Workspace/.venv/bin/python}"
WORK_DIR="${SCRIPT_DIR}/data/work"
SRC_PY="${SCRIPT_DIR}/src/python"
DATA_INPUT="${SCRIPT_DIR}/data/input"
DATA_OUTPUT="${SCRIPT_DIR}/data/output"

STEP="${1:-all}"
DARK="${2:-${DATA_INPUT}/m010_b0101_ms_20170330_09564585_x1.fit}"
FLAT="${3:-${DATA_INPUT}/m011_b0101_ms_20170330_10013140_y1.fit}"

# Créer le répertoire de travail avec les symlinks
mkdir -p "${WORK_DIR}"
cp "${SRC_PY}/ms.yml"   "${WORK_DIR}/"
ln -sf "${DARK}"  "${WORK_DIR}/dark_2015.fits"
ln -sf "${FLAT}"  "${WORK_DIR}/flat_2015.fits"

# Symlinks sur les lights pour ms1.py (pattern m*x1.fit, m*y1.fit)
for f in "${DATA_INPUT}/lights/"*.fit "${DATA_INPUT}/"m0*.fit; do
    [ -f "$f" ] && ln -sf "$f" "${WORK_DIR}/$(basename "$f")" 2>/dev/null || true
done

echo "Working dir: ${WORK_DIR}"
echo "Dark:  $(readlink -f "${WORK_DIR}/dark_2015.fits")"
echo "Flat:  $(readlink -f "${WORK_DIR}/flat_2015.fits")"
echo ""

cd "${WORK_DIR}"

run_ms1() {
    echo "=== Step 1: Averaging ==="
    cp "${SRC_PY}/ms1.py" .
    "${PYTHON}" ms1.py
    mv ms.lis xtab.lis ytab.lis "${DATA_OUTPUT}/" 2>/dev/null || true
    echo ""
}

run_ms2() {
    echo "=== Step 2: Geometry ==="
    cp "${SRC_PY}/ms2.py" .
    "${PYTHON}" ms2.py
    mv geo*.pdf ACDF2.lis xryr.lis ms.lis "${DATA_OUTPUT}/" 2>/dev/null || true
    echo ""
}

case "${STEP}" in
    step1|1)  run_ms1 ;;
    step2|2)  run_ms2 ;;
    all|*)    run_ms1; run_ms2 ;;
esac

echo "=== Done. Output in ${DATA_OUTPUT}/ ==="
ls -la "${DATA_OUTPUT}/"
