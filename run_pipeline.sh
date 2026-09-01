#!/bin/bash
# run_pipeline.sh — Compile et exécute le pipeline Fortran MSDP
#
# Usage:
#   ./run_pipeline.sh                         # utilise src/fortran/ms.par
#   ./run_pipeline.sh /chemin/vers/ms.par     # utilise un ms.par custom
#
# Prérequis: gfortran, gs (ghostscript), PGPLOT installé

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="/tmp/msdp_pipeline"
WORK_DIR="${BUILD_DIR}/work"
SRC_FORTRAN="${PROJECT_DIR}/src/fortran"
DATA_OUTPUT="${PROJECT_DIR}/data/output"

# ms.par : argument ou défaut
MS_PAR="${1:-${SRC_FORTRAN}/ms.par}"

if [ ! -f "${MS_PAR}" ]; then
    echo "ERREUR: ${MS_PAR} non trouvé"
    exit 1
fi

echo "============================================"
echo "  Pipeline Fortran MSDP"
echo "  ms.par: ${MS_PAR}"
echo "============================================"

# --- 1. Préparer le répertoire ---
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}" "${BUILD_DIR}/lib"
cp "${SRC_FORTRAN}/ms1.f" "${SRC_FORTRAN}/ms2.f" "${BUILD_DIR}/"
cp "${MS_PAR}" "${WORK_DIR}/ms.par"

# Symlink pour -lX11
ln -sf /usr/lib/x86_64-linux-gnu/libX11.so.6 "${BUILD_DIR}/lib/libX11.so"

# --- 2. Compiler ---
echo "  Compilation..."
cd "${BUILD_DIR}"
gfortran -g -o msdp ms1.f ms2.f \
    -lpgplot -L/usr/lib/x86_64-linux-gnu -L"${BUILD_DIR}/lib" -lX11 \
    -lgfortran -lquadmath 2>&1 | grep -i error && {
    echo "ERREUR de compilation"
    exit 1
}
echo "  OK"

# --- 3. Préparer les données ---
echo "  Données..."
cd "${WORK_DIR}"
ln -sf "${PROJECT_DIR}/data/input/m010_b0101_ms_20170330_09564585_x1.fit" .
ln -sf "${PROJECT_DIR}/data/input/m011_b0101_ms_20170330_10013140_y1.fit" .
ln -sf "${BUILD_DIR}/msdp" .

# --- 4. Exécuter ---
echo "  Exécution..."
rm -f ms.lis xtab.lis ytab.lis channel.lis geo*.ps geo*.pdf ACDF2.lis xryr.lis *.x* *.y*
export PGPLOT_FONT="/tmp/msdp_fortran/pgplot5_extracted/usr/lib/pgplot5/grfont.dat"
timeout 120 ./msdp 2>&1 | tail -15

# --- 5. Convertir PS → PDF (en /tmp puis copie — gs ne peut pas écrire direct dans snap Nextcloud) ---
echo ""
echo "  Conversion PS → PDF..."
mkdir -p "${DATA_OUTPUT}" /tmp/msdp_pdf_$$
for ps in geo1 geo2 geo3; do
    if [ -f "${WORK_DIR}/${ps}.ps" ] && [ -s "${WORK_DIR}/${ps}.ps" ]; then
        gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite \
           -sOutputFile="/tmp/msdp_pdf_$$/${ps}.pdf" \
           "${WORK_DIR}/${ps}.ps" 2>/dev/null
        cp "/tmp/msdp_pdf_$$/${ps}.pdf" "${DATA_OUTPUT}/${ps}.pdf"
        echo "  ✓ ${ps}.pdf"
    fi
done
rm -rf /tmp/msdp_pdf_$$
cp "${WORK_DIR}/ACDF2.lis" "${DATA_OUTPUT}/" 2>/dev/null || true
cp "${WORK_DIR}/ms.lis" "${DATA_OUTPUT}/" 2>/dev/null || true

echo ""
echo "  Résultats: ${DATA_OUTPUT}/"
echo "  Logs: ${WORK_DIR}/ms.lis"
echo "============================================"
