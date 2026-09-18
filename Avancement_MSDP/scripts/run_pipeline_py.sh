#!/bin/bash
# run_pipeline_py.sh — Exécute le pipeline Python MSDP (équivalent de run_pipeline.sh)
#
# Usage:
#   ./run_pipeline_py.sh                          # utilise src/python/ms.yml
#   ./run_pipeline_py.sh /chemin/vers/ms.yml      # utilise un ms.yml custom
#
# Prérequis: python3, numpy, yaml, astropy, matplotlib installés.
#   (vérifier avec: python3 -c "import numpy,yaml,astropy,matplotlib")
#
# Ce script reproduit la logique de run_pipeline.sh mais pour le pipeline Python :
#   ms1.py -> moyenne dark (x) / flat (y)
#   ms2.py -> géométrie des canaux (newgeom) + plots geo1/2/3.pdf + ACDF2.lis
# Les numéros de version logs (ms_run_N / ACDF2_run_N) sont partagés avec le
# pipeline Fortran afin de garder une trace unique des exécutions.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="/tmp/msdp_py_pipeline"
WORK_DIR="${BUILD_DIR}/work"
SRC_PYTHON="${PROJECT_DIR}/src/python"
DATA_OUTPUT="${PROJECT_DIR}/data/output"

# ms.yml : argument ou défaut
MS_YML="${1:-${SRC_PYTHON}/ms.yml}"

if [ ! -f "${MS_YML}" ]; then
    echo "ERREUR: ${MS_YML} non trouvé"
    exit 1
fi

echo "============================================"
echo "  Pipeline Python MSDP"
echo "  ms.yml: ${MS_YML}"
echo "============================================"

# --- 1. Préparer le répertoire ---
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cp "${SRC_PYTHON}/ms1.py" "${SRC_PYTHON}/ms2.py" "${WORK_DIR}/"
cp "${MS_YML}" "${WORK_DIR}/ms.yml"

# --- 2. Préparer les données ---
echo "  Données..."
cd "${WORK_DIR}"

# Copier (lien) TOUS les darks (*x1.fit) et flats (*y1.fit) présents, triés par nom
# (ms1.py fait sort(glob('m*x1.fit')) / sort(glob('m*y1.fit')) - même logique que le Fortran)
NKEEP=0
NFKEEP=0
for f in $(ls "${PROJECT_DIR}/data/input/"*x1.fit 2>/dev/null | sort); do
    ln -sf "$f" .; NKEEP=$((NKEEP+1))
done
for f in $(ls "${PROJECT_DIR}/data/input/"*y1.fit 2>/dev/null | sort); do
    ln -sf "$f" .; NFKEEP=$((NFKEEP+1))
done
echo "  → Darks liés: ${NKEEP}  Flats liés: ${NFKEEP}"

# --- 3. Ajuster nfx2/nfy2 dans ms.yml au nombre réel de fichiers (comme le Fortran) ---
if [ "${NFKEEP}" -ge 1 ]; then
    sed -i "s/^nfy2: .*/nfy2: ${NFKEEP}/" ms.yml
fi
if [ "${NKEEP}" -ge 1 ]; then
    sed -i "s/^nfx2: .*/nfx2: ${NKEEP}/" ms.yml
fi
echo "  ms.yml: nfx2=${NKEEP} nfy2=${NFKEEP} (ajusté au nb de fichiers)"

# --- 4. Exécuter Step 1 (ms1.py : moyennes dark/flat) ---
echo ""
echo "  Exécution Step 1 (ms1.py) - Averaging..."
# ms1.py ouvre ms.lis en 'w' : on repart à zéro.
rm -f ms.lis xtab.lis ytab.lis channel.lis *00000
timeout 120 python3 ms1.py 2>&1 | tail -8
if [ ! -s ms.lis ]; then echo "ERREUR: ms1.py n'a rien produit"; exit 1; fi

# --- 5. Exécuter Step 2 (ms2.py : géométrie + ACDF2) ---
echo ""
echo "  Exécution Step 2 (ms2.py) - Geometry..."
# ms2.py ouvre ms.lis en 'a' (append) : la géométrie s'ajoute au log du Step 1.
# Sans argument, ms2.py cherche automatiquement les fichiers moyens *00000 produits.
timeout 120 python3 ms2.py 2>&1 | tail -8
if [ ! -s ACDF2.lis ]; then echo "ERREUR: ms2.py n'a pas produit ACDF2.lis"; exit 1; fi

echo "  Pipeline Python terminé."
echo ""

# --- 6. Sorties versionnées (numérotation PYTHON INDÉPENDANTE du Fortran) ---
mkdir -p "${DATA_OUTPUT}"

# 2026-09-06 : le compteur Python est désormais INDÉPENDANT de celui du Fortran.
# Avant, les deux pipelines partageaient le même compteur (ms_run_*.lis), ce qui
# faisait que Python comptait de 2 en 2 (toujours les numéros pairs après le
# Fortran). Désormais le Python utilise son propre préfixe 'ms_run_py_NNN' /
# 'ACDF2_run_py_NNN'. On peut ainsi compter chaque pipeline de 1,2,3… à part.
get_next_run_py() {
    local i=1 max=0 n
    for f in "${DATA_OUTPUT}"/ms_run_py_*.lis; do
        [ -e "$f" ] || continue
        n="${f##*_py_}"; n="${n%.lis}"
        n=$((10#$n))
        if [ "$n" -gt "$max" ]; then max="$n"; fi
    done
    printf '%02d' $((max + 1))
}
RUN_NUM="$(get_next_run_py)"

# Plots PDF versionnés (le pipeline Fortran utilise geo{1,2,3}_fortran_N; ici geo{1,2,3}_python_N)
for g in geo1 geo2 geo3; do
    if [ -f "${WORK_DIR}/${g}.pdf" ]; then
        cp "${WORK_DIR}/${g}.pdf" "${DATA_OUTPUT}/${g}_python_${RUN_NUM}.pdf"
        echo "  ✓ ${g}_python_${RUN_NUM}.pdf"
    fi
done

# Logs versionnés (préfixe _py_ pour la numérotation Python indépendante)
cp "${WORK_DIR}/ms.lis"     "${DATA_OUTPUT}/ms_run_py_${RUN_NUM}.lis"     2>/dev/null || true
cp "${WORK_DIR}/ACDF2.lis"  "${DATA_OUTPUT}/ACDF2_run_py_${RUN_NUM}.lis"  2>/dev/null || true
cp "${WORK_DIR}/ms.yml"     "${DATA_OUTPUT}/ms_par_run_py_${RUN_NUM}.yml" 2>/dev/null || true
echo "  ⚙  Logs versionnés: ms_run_py_${RUN_NUM}.lis / ACDF2_run_py_${RUN_NUM}.lis / ms_par_run_py_${RUN_NUM}.yml"

# Copies "courantes" (dernière exécution)
cp "${WORK_DIR}/ACDF2.lis" "${DATA_OUTPUT}/" 2>/dev/null || true
cp "${WORK_DIR}/ms.lis"    "${DATA_OUTPUT}/" 2>/dev/null || true

echo ""
echo "  Résultats: ${DATA_OUTPUT}/"
echo "  Logs: ${WORK_DIR}/ms.lis"
echo "============================================"