#!/bin/bash
# test_regression.sh — Vérifie que les pipelines Fortran + Python n'ont pas
# régressé : pour un jeu de données donné (config par défaut), l'ACDF2 produit
# doit rester identique à une référence sauvegardée.
#
# Ce script est le "filet de sécurité" prévu avant toute refonte délicate
# (A2-bis, im/jm). Il garantit qu'un changement de code ne modifie pas la sortie
# sur les données de référence.
#
# Usage :
#   ./test_regression.sh [--update] [--verbose]
#     --update   : sauvegarde l'ACDF2 courant comme nouvelle référence
#     --verbose  : affiche le diff complet au lieu d'un résumé
#
# Comportement par défaut :
#   1. lance run_both_pipeline.sh (Fortran + Python) avec la config par défaut
#   2. compare l'ACDF2 Fortran produit à data/output/ACDF2.fortran.ref
#   3. compare l'ACDF2 Python produit  à data/output/ACDF2.python.ref
#   4. exit 0 si identiques (ou si --update), exit 1 sinon

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_OUTPUT="${PROJECT_DIR}/data/output"
REF_FORTRAN="${DATA_OUTPUT}/ACDF2.fortran.ref"
REF_PYTHON="${DATA_OUTPUT}/ACDF2.python.ref"
VERBOSE=0

for arg in "$@"; do
    case "$arg" in
        --update)  UPDATE=1 ;;
        --verbose) VERBOSE=1 ;;
        *) echo "Argument inconnu: $arg"; exit 2 ;;
    esac
done
UPDATE="${UPDATE:-0}"

echo "============================================"
echo "  TEST DE RÉGRESSION — Pipeline MSDP"
echo "============================================"

# --- Sauvegarder l'ACDF2 courant avant le run (comme nouvelle réf si --update) ---
# Les chemins du pipeline écrasent data/output/ACDF2.lis ; on sauvegarde l'état
# courant pour comparaison APRES le run, selon le mode choisi.

# --- S'assurer qu'on a une référence ---
if [ ! -f "${REF_FORTRAN}" ] || [ ! -f "${REF_PYTHON}" ]; then
    echo "⚠  Références manquantes (ACDF2.fortran.ref / ACDF2.python.ref)."
    echo "   Lancez ./test_regression.sh --update une première fois."
    if [ "${UPDATE}" != "1" ]; then
        exit 2
    fi
fi

# --- Étape 1 : lancer le pipeline complet ---
echo ""
echo "[1/2] Exécution des pipelines (Fortran + Python)..."
"${PROJECT_DIR}/run_both_pipeline.sh" > /tmp/regression_run.log 2>&1
echo "      OK (voir /tmp/regression_run.log)"

# --- Étape 2 : cueillir les ACDF2 produits ---
# run_both_pipeline.sh lance Fortran (run_pipeline.sh) PUIS Python
# (run_pipeline_py.sh). Donc les 2 runs les PLUS RÉCENTS créés sont, dans l'ordre,
# Fortran (avant-dernier) puis Python (dernier). On NE se fie PAS à la parité des
# numéros (désalignée par des runs manuels intercalés).
mapfile -t runs < <(ls -1t "${DATA_OUTPUT}"/ACDF2_run_*.lis 2>/dev/null)
if [ "${#runs[@]}" -lt 2 ]; then
    echo "❌ Moins de 2 ACDF2_run_*.lis disponibles."
    exit 1
fi
CUR_PYTHON="${runs[0]}"      # le plus récent = Python
CUR_FORTRAN="${runs[1]}"     # avant-dernier = Fortran

if [ -z "${CUR_FORTRAN}" ] || [ -z "${CUR_PYTHON}" ]; then
    echo "❌ Impossible de déterminer les ACDF2 Fortran et Python (runs)."
    exit 1
fi
echo "[2/2] ACDF2 produits :"
echo "      Fortran : ${CUR_FORTRAN}"
echo "      Python  : ${CUR_PYTHON}"

# --- Comparer avec les références ---
cmp_ref() {
    local cur="$1" ref="$2" label="$3"
    if [ "${UPDATE}" == "1" ]; then
        cp "$cur" "$ref"
        echo "✔ ${label} : référence mise à jour → $(basename "${ref}")"
        return 0
    fi
    if diff -q "$cur" "$ref" >/dev/null 2>&1; then
        echo "✔ ${label} : IDENTIQUE à la référence"
        return 0
    else
        echo "❌ ${label} : DIFFÉRENT de la référence"
        if [ "${VERBOSE}" == "1" ]; then
            diff "$cur" "$ref" | head -30
        fi
        return 1
    fi
}

ok=0
cmp_ref "$CUR_FORTRAN" "$REF_FORTRAN" "Fortran" || ok=1
cmp_ref "$CUR_PYTHON"  "$REF_PYTHON"  "Python"  || ok=1

echo ""
if [ "${UPDATE}" == "1" ]; then
    echo "✅ Références enregistrées (mode --update)."
elif [ "${ok}" == "0" ]; then
    echo "✅ RÉGRESSION OK : les sorties correspondent aux références."
else
    echo "❌ RÉGRESSION DÉTECTÉE : voir ci-dessus."
fi
echo "============================================"
exit "${ok}"