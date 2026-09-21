#!/bin/bash
# run_both_pipeline.sh — Exécute les deux pipelines MSDP complets : Fortran puis Python
#
# Usage:
#   ./run_both_pipeline.sh                              # ms.par par défaut
#   ./run_both_pipeline.sh /chemin/ms.par               # ms.par custom
#
# Déroulé :
#   1. run_pipeline.sh     (Fortran : ms1..ms4 — sorties dans data/output/run_NNN/)
#   2. run_pipeline_py.sh  (Python new_python : chaîne complète — sorties dans
#      new_python/outputs/run_NNN/ + data/output/run_NNN/ pour les logs)
#
# Les deux pipelines ont des numérotations de run indépendantes :
#   - Fortran : data/output/run_NNN/  (ms_run_N / ACDF2_run_N)
#   - Python  : new_python/outputs/run_NNN/  (ms_run_py_N / ACDF2_run_py_N)
#
# À la fin, un récapitulatif indique les deux dossiers de run produits.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SELF_DIR="$PROJECT_DIR"

# Argument : ms.par [ms.yml]  (ms.yml conservé pour compat, non utilisé par new_python)
MS_PAR="${1:-}"
MS_YML="${2:-}"

echo "============================================"
echo "  Pipelines MSDP complets — Fortran + Python"
echo "============================================"
echo ""

# --- 1. Pipeline Fortran ---
echo "############ 1/2 : PIPELINE FORTRAN ############"
echo ""
if [ -n "${MS_PAR}" ]; then
    "${SELF_DIR}/run_pipeline.sh" "${MS_PAR}"
else
    "${SELF_DIR}/run_pipeline.sh"
fi
echo ""
echo "→ Fin du pipeline Fortran (sorties dans data/output/run_NNN/)."
echo ""

# --- 2. Pipeline Python (new_python) ---
echo "############ 2/2 : PIPELINE PYTHON (new_python) ############"
echo ""
if [ -n "${MS_PAR}" ]; then
    "${SELF_DIR}/run_pipeline_py.sh" "${MS_PAR}"
else
    "${SELF_DIR}/run_pipeline_py.sh"
fi
echo ""
echo "→ Fin du pipeline Python (sorties dans new_python/outputs/run_NNN/)."

echo ""
echo "============================================"
echo "  Terminé. Les deux pipelines ont tourné :"
echo "    • Fortran : $(ls -dt ${PROJECT_DIR}/data/output/run_* 2>/dev/null | head -1)"
echo "    • Python  : $(ls -dt ${PROJECT_DIR}/new_python/outputs/run_* 2>/dev/null | head -1)"
echo "============================================"