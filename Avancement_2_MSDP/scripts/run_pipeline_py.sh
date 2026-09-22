#!/bin/bash
# run_pipeline_py.sh — Exécute le pipeline Python MSDP (new_python, porté en 3.11)
#
# Usage:
#   ./run_pipeline_py.sh                          # ms.par par défaut (src/fortran/new/ms.par)
#   ./run_pipeline_py.sh /chemin/ms.par           # ms.par custom
#
# Le pipeline Python (new_python/msdp/) reproduit intégralement la chaîne
# Fortran ms1.f → ms2.f → ms3.f → ms4.f : moyennes dark/flat → géométrie →
# canaux → calibration → observations → profils I/V + vitesses.
#
# Sorties :
#   - Toutes les figures + ACDF2.lis + miv.lis dans new_python/outputs/run_NNN/
#   - Logs versionnés (ms_run_py_NNN, ACDF2_run_py_NNN, miv_run_py_NNN,
#     ms_par_run_py_NNN) dans data/output/   (numérotation Python indépendante)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
NEW_PY="${PROJECT_DIR}/new_python"
PYTHON="${NEW_PY}/.venv/bin/python"

# ms.par : argument ou défaut
MS_PAR="${1:-${PROJECT_DIR}/src/fortran/new/ms.par}"

if [ ! -f "${MS_PAR}" ]; then
    echo "ERREUR: ${MS_PAR} non trouvé"
    exit 1
fi

if [ ! -x "${PYTHON}" ]; then
    echo "ERREUR: venv introuvable (${PYTHON}). Créer le venv de new_python d'abord."
    exit 1
fi

echo "============================================"
echo "  Pipeline Python MSDP (new_python)"
echo "  ms.par: ${MS_PAR}"
echo "============================================"

# Le CLI numérote automatiquement le run_NNN (new_python/outputs/run_NNN/)
# et copie les logs versionnés dans data/output/.
cd "${NEW_PY}"
"${PYTHON}" run_pipeline_cli.py "${MS_PAR}"

echo ""
echo "  Résultats: ${NEW_PY}/outputs/"
echo "  Logs     : ${PROJECT_DIR}/data/output/"
echo "============================================"