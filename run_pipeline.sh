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

# Lier TOUS les darks (*x1.fit) et flats (*y1.fit) présents, triés par nom
# (le Fortran fait `ls m*x1.fit` / `ls m*y1.fit`, donc l'ordre lexicographique compte)
NKEEP=0
NFKEEP=0
for f in $(ls "${PROJECT_DIR}/data/input/"*x1.fit 2>/dev/null | sort); do
    ln -sf "$f" .; NKEEP=$((NKEEP+1))
done
for f in $(ls "${PROJECT_DIR}/data/input/"*y1.fit 2>/dev/null | sort); do
    ln -sf "$f" .; NFKEEP=$((NFKEEP+1))
done
echo "  → Darks liés: ${NKEEP}  Flats liés: ${NFKEEP}"

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

# --- 5. Convertir PS → PDF versionné (en /tmp puis copie — gs ne peut pas écrire direct dans snap Nextcloud) ---
echo ""
echo "  Conversion PS → PDF (versionné)..."

# Déterminer le prochain numéro de version libre
get_next_version() {
    local prefix="$1"
    local i=1
    while [ -f "${DATA_OUTPUT}/${prefix}_fortran_$(printf '%02d' ${i}).pdf" ]; do
        i=$((i + 1))
    done
    printf '%02d' "${i}"
}

mkdir -p "${DATA_OUTPUT}" /tmp/msdp_pdf_$$
for ps in geo1 geo2 geo3; do
    if [ -f "${WORK_DIR}/${ps}.ps" ] && [ -s "${WORK_DIR}/${ps}.ps" ]; then
        # Le numéro de version doit être identique pour les 3 plots d'un même run
        if [ -z "${VERSION_NUM:-}" ]; then
            VERSION_NUM="$(get_next_version geo1)"
            echo "  → Nouvelle version: _fortran_${VERSION_NUM}"
        fi
        gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite \
           -sOutputFile="/tmp/msdp_pdf_$$/${ps}.pdf" \
           "${WORK_DIR}/${ps}.ps" 2>/dev/null
        cp "/tmp/msdp_pdf_$$/${ps}.pdf" "${DATA_OUTPUT}/${ps}_fortran_${VERSION_NUM}.pdf"
        echo "  ✓ ${ps}_fortran_${VERSION_NUM}.pdf"
    fi
done
rm -rf /tmp/msdp_pdf_$$
# --- Logs versionnés (trace des itérations) ---
# Numéro de run : on repart du plus haut *_run_N consistant, puis on incrémente en
# verrouillant ms_run_N & ACDF2_run_N ensemble.
get_next_run() {
    local i=1 max=0 n
    # 2026-09-06 : NE compter QUE les runs Fortran. Le glob ms_run_*.lis matche
    # aussi ms_run_py_*.lis (numérotation Python indépendante) — on les exclut.
    for f in "${DATA_OUTPUT}"/ms_run_*.lis; do
        [ -e "$f" ] || continue
        case "$f" in
            *_py_*) continue ;;   # run Python (numérotation séparée)
        esac
        n="${f##*_run_}"; n="${n%.lis}"
        n=$((10#$n))
        if [ "$n" -gt "$max" ]; then max="$n"; fi
    done
    printf '%03d' $((max + 1))
}
RUN_NUM="${RUN_NUM:-$(get_next_run)}"
cp "${WORK_DIR}/ms.lis"     "${DATA_OUTPUT}/ms_run_${RUN_NUM}.lis"     2>/dev/null || true
cp "${WORK_DIR}/ACDF2.lis"  "${DATA_OUTPUT}/ACDF2_run_${RUN_NUM}.lis"  2>/dev/null || true
cp "${MS_PAR}"              "${DATA_OUTPUT}/ms_par_run_${RUN_NUM}.par" 2>/dev/null || true
echo "  ⚙  Logs versionnés: ms_run_${RUN_NUM}.lis / ACDF2_run_${RUN_NUM}.lis / ms_par_run_${RUN_NUM}.par"

cp "${WORK_DIR}/ACDF2.lis" "${DATA_OUTPUT}/" 2>/dev/null || true
cp "${WORK_DIR}/ms.lis" "${DATA_OUTPUT}/" 2>/dev/null || true

echo ""
echo "  Résultats: ${DATA_OUTPUT}/"
echo "  Logs: ${WORK_DIR}/ms.lis"
echo "============================================"
