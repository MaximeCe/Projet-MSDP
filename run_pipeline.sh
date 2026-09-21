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
SRC_FORTRAN="${PROJECT_DIR}/src/fortran/new"
DATA_OUTPUT="${PROJECT_DIR}/data/output"

# ms.par : argument ou défaut (le Python new_python tourne sur le même new/ms.par
# pour une parité rigoureuse — mêmes nfx/nfy et même lbdvel 20/35/50).
MS_PAR="${1:-${PROJECT_DIR}/src/fortran/new/ms.par}"

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
cp "${SRC_FORTRAN}/ms1.f" "${SRC_FORTRAN}/ms2.f" "${SRC_FORTRAN}/ms3.f" "${SRC_FORTRAN}/ms4.f" "${BUILD_DIR}/"
cp "${MS_PAR}" "${WORK_DIR}/ms.par"

# Symlink pour -lX11
ln -sf /usr/lib/x86_64-linux-gnu/libX11.so.6 "${BUILD_DIR}/lib/libX11.so"

# --- 2. Compiler ---
echo "  Compilation..."
cd "${BUILD_DIR}"
gfortran -g -o msdp ms1.f ms2.f ms3.f ms4.f \
    -lpgplot -L/usr/lib/x86_64-linux-gnu -L"${BUILD_DIR}/lib" -lX11 \
    -lgfortran -lquadmath 2>&1 | grep -i error && {
    echo "ERREUR de compilation"
    exit 1
}
echo "  OK"

# --- 3. Préparer les données ---
echo "  Données..."
cd "${WORK_DIR}"

# Lier TOUS les darks (*x1.fit), flats (*y1.fit) et observations (*b1.fit) présents, triés par nom
# (le Fortran fait `ls m*x1.fit` / `ls m*y1.fit` / `ls m*b1.fit`, donc l'ordre lexicographique compte)
NKEEP=0
NFKEEP=0
NBKEEP=0
for f in $(ls "${PROJECT_DIR}/data/input/"*x1.fit 2>/dev/null | sort); do
    ln -sf "$f" .; NKEEP=$((NKEEP+1))
done
for f in $(ls "${PROJECT_DIR}/data/input/"*y1.fit 2>/dev/null | sort); do
    ln -sf "$f" .; NFKEEP=$((NFKEEP+1))
done
for f in $(ls "${PROJECT_DIR}/data/input/"*b1.fit 2>/dev/null | sort); do
    ln -sf "$f" .; NBKEEP=$((NBKEEP+1))
done
echo "  → Darks liés: ${NKEEP}  Flats liés: ${NFKEEP}  Obs liées: ${NBKEEP}"

# Ajuster nfy2/nfx2 dans ms.par au nombre réel de fichiers si >0
if [ "${NFKEEP}" -ge 1 ]; then
    sed -i "s/^    nfy2       1/    nfy2       ${NFKEEP}/" ms.par
    sed -i "s/^    nfy1       1/    nfy1       1/" ms.par
fi
if [ "${NKEEP}" -ge 1 ]; then
    sed -i "s/^    nfx2       1/    nfx2       ${NKEEP}/" ms.par
    sed -i "s/^    nfx1       1/    nfx1       1/" ms.par
fi

ln -sf "${BUILD_DIR}/msdp" .
echo "  ms.par: nfx2=${NKEEP} nfy2=${NFKEEP} (ajusté au nb de fichiers)"

# --- 4. Exécuter ---
echo "  Exécution..."
rm -f ms.lis xtab.lis ytab.lis channel.lis geo*.ps geo*.pdf ACDF2.lis xryr.lis *.x* *.y*
export PGPLOT_FONT="/tmp/msdp_fortran/pgplot5_extracted/usr/lib/pgplot5/grfont.dat"
timeout 120 ./msdp 2>&1 | tail -15

# --- 5. Convertir PS → PDF et tout regrouper dans data/output/run_NNN/ ---
echo ""
echo "  Conversion PS → PDF + regroupement dans run_NNN/..."

mkdir -p "${DATA_OUTPUT}" /tmp/msdp_pdf_$$

# --- Numéro de run & version : on compte les DOSSIERS run_NNN (pas de fichiers à plat) ---
# Les sorties vivent uniquement dans data/output/run_NNN/.
get_next_run() {
    local max=0 d num digits
    for d in "${DATA_OUTPUT}"/run_*; do
        [ -d "$d" ] || continue
        # extrait le nombre après "run_" (ex: run_014 -> 14)
        digits="${d##*run_}"
        case "$digits" in
            *[!0-9]*|'') continue ;;
        esac
        # enlève les zéros de tête pour éviter l'interprétation octale
        num=$((10#$digits))
        if [ "$num" -gt "$max" ]; then max="$num"; fi
    done
    printf '%03d' $((max + 1))
}
RUN_NUM="${RUN_NUM:-$(get_next_run)}"
VERSION_NUM="$RUN_NUM"        # un seul numéro par run (PDFs + logs + dossier)
RUN_DIR="${DATA_OUTPUT}/run_${RUN_NUM}"
mkdir -p "${RUN_DIR}"
echo "  → Run: ${RUN_NUM}  (dossier ${RUN_DIR}/)"

# --- Convertir les PS → PDF et les écrire DIRECTEMENT dans run_NNN/ ---
for ps in geo1 geo2 geo3 geo4 calib cal obs obsD1 obsD2 ivprof1 ivprof2 ivprof3; do
    if [ -f "${WORK_DIR}/${ps}.ps" ] && [ -s "${WORK_DIR}/${ps}.ps" ]; then
        gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite \
           -sOutputFile="${RUN_DIR}/${ps}_fortran_${VERSION_NUM}.pdf" \
           "${WORK_DIR}/${ps}.ps" 2>/dev/null
        echo "  ✓ ${ps}_fortran_${VERSION_NUM}.pdf"
    fi
done
rm -rf /tmp/msdp_pdf_$$

# --- Logs versionnés (directement dans run_NNN/) ---
cp "${WORK_DIR}/ms.lis"     "${RUN_DIR}/ms_run_${RUN_NUM}.lis"     2>/dev/null || true
cp "${WORK_DIR}/ACDF2.lis"  "${RUN_DIR}/ACDF2_run_${RUN_NUM}.lis"  2>/dev/null || true
cp "${WORK_DIR}/miv.lis"    "${RUN_DIR}/miv_run_${RUN_NUM}.lis"     2>/dev/null || true
cp "${MS_PAR}"              "${RUN_DIR}/ms_par_run_${RUN_NUM}.par" 2>/dev/null || true
echo "  ⚙  Logs: ms_run_${RUN_NUM}.lis / ACDF2_run_${RUN_NUM}.lis / miv_run_${RUN_NUM}.lis / ms_par_run_${RUN_NUM}.par"

echo ""
echo "  📁 Toutes les sorties du run sont dans ${RUN_DIR}/"
ls -1 "${RUN_DIR}" 2>/dev/null | sed 's/^/      /'

echo ""
echo "  Résultats: ${DATA_OUTPUT}/"
echo "  Logs: ${WORK_DIR}/ms.lis"
echo "============================================"
