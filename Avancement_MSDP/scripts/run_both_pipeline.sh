#!/bin/bash
# run_both_pipeline.sh — Exécute les deux pipelines MSDP : Fortran puis Python
#
# Usage:
#   ./run_both_pipeline.sh                          # ms.par et ms.yml par défaut
#   ./run_both_pipeline.sh /chemin/ms.par           # ms.par custom (ms.yml par défaut)
#   ./run_both_pipeline.sh /chemin/ms.par /chemin/ms.yml
#
# Déroulé :
#   1. run_pipeline.sh     (Fortran : moyennes dark/flat + géométrie newgeom)
#   2. run_pipeline_py.sh  (Python  : même périmètre via ms1.py + ms2.py)
#
# Les deux scripts partagent la même numérotation de version (ms_run_N / ACDF2_run_N),
# donc ce script produit deux jeux de sorties versionnées consécutives.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SELF_DIR="$PROJECT_DIR"

# Arguments : ms.par [ms.yml]
MS_PAR="${1:-}"
MS_YML="${2:-}"

echo "============================================"
echo "  Pipeline MSDP — Fortran + Python"
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
echo "→ Fin du pipeline Fortran."
echo ""

# --- 2. Pipeline Python ---
echo "############ 2/2 : PIPELINE PYTHON ############"
echo ""
if [ -n "${MS_YML}" ]; then
    "${SELF_DIR}/run_pipeline_py.sh" "${MS_YML}"
else
    "${SELF_DIR}/run_pipeline_py.sh"
fi
echo ""
echo "→ Fin du pipeline Python."

echo ""
echo "============================================"
echo "  Terminé. Les deux pipelines ont tourné."
echo "============================================"