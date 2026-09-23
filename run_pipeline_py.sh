#!/bin/bash
# run_pipeline_py.sh — Exécute le pipeline Python MSDP (new_python, porté en 3.11)
#
# Usage:
#   ./run_pipeline_py.sh                          # config.yml par défaut (new_python/config.yml)
#   ./run_pipeline_py.sh /chemin/config.yml       # config YAML moderne
#   ./run_pipeline_py.sh /chemin/ms.par           # config legacy (fallback)
#
# Le pipeline Python (new_python/msdp/) reproduit intégralement la chaîne
# Fortran ms1.f → ms2.f → ms3.f → ms4.f : moyennes dark/flat → géométrie →
# canaux → calibration → observations → profils I/V + vitesses.
#
# Sorties (formats modernes) :
#   - Toutes les figures PNG + ACDF2.csv + miv.csv dans new_python/outputs/run_NNN/
#   - Log structuré ms_run_*.json (+ rendu texte lisible .txt)
#   - Logs versionnés (ms_run_py_NNN.json, ACDF2_run_py_NNN.csv, miv_run_py_NNN.csv,
#     ms_par_run_py_NNN.yml) dans data/output/   (numérotation Python indépendante)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
NEW_PY="${PROJECT_DIR}/new_python"
PYTHON="${NEW_PY}/.venv/bin/python"

# config.yml : argument ou défaut (repli ms.par legacy si absent)
CFG="${1:-${NEW_PY}/config.yml}"

if [ ! -f "${CFG}" ]; then
    echo "ERREUR: ${CFG} non trouvé"
    exit 1
fi

if [ ! -x "${PYTHON}" ]; then
    echo "ERREUR: venv introuvable (${PYTHON}). Créer le venv de new_python d'abord."
    exit 1
fi

echo "============================================"
echo "  Pipeline Python MSDP (new_python)"
echo "  config: ${CFG}"
echo "============================================"

# Le CLI numérote automatiquement le run_NNN (new_python/outputs/run_NNN/)
# et copie les logs versionnés dans data/output/.
cd "${NEW_PY}"
"${PYTHON}" run_pipeline_cli.py "${CFG}"

echo ""
echo "  Résultats: ${NEW_PY}/outputs/"
echo "  Logs     : ${PROJECT_DIR}/data/output/"
echo "============================================"