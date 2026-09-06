# `check_fortran_pipeline.py` — Vérification de la réussite du pipeline Fortran

> Chemin : `Projet-MSDP/check_fortran_pipeline.py`
> Rôle : **valider que les RÉSULTATS du pipeline Fortran MSDP sont bons**, pas
> seulement que l'algorithme s'est exécuté jusqu'au bout (`end geom`).

---

## 1. Pourquoi ce script ?

Le pipeline Fortran (`run_pipeline.sh` → `ms1.f`+`ms2.f`) peut **se terminer
normalement** (il atteint `end geom`) tout en produisant une **géométrie ****
**ratée** : canaux non détectés (valeurs `0.00`), débordement numérique
(`NaN`/`1.13E+33`), largeurs repliées... C'était exactement le cas du bug
d'indexation mémoriel (`xx(24,24)` vs `xx(20,9)`) qu'on a corrigé — le run 001
terminait sans erreur mais sortait une ACDF2 corrompue.

Ce script analyse le fichier **`ACDF2.lis`** (les coins A,C,D,F des 9 canaux,
sortie principale de la géométrie) et s'assure que la géométrie détectée est
**physiquement cohérente**.

---

## 2. Critères de validation

Le script vérifie 6 invariants géométriques :

| # | Critère | Ce qu'il détecte |
|---|---------|------------------|
| 1 | **Format** : 9 lignes × 8 valeurs numériques | fichier absent / mal écrit |
| 2 | **Pas de valeur nulle** (0.00) | canal non détecté / dégénéré (classique du bug d'indexation) |
| 3 | **Pas de NaN/infini** | overflow dans `intersec` (division par ~0) |
| 4 | **Bornes physiques** (X∈[0,1536], Y∈[0,1536]) | coordonnées hors image |
| 5 | **Centres X strictement croissants** + **pas régulier** (σ < 25 %) | canaux mélangés / non ordonnés |
| 6 | **Largeurs de canal cohérentes** (σ < 35 %) | canal replié, étiré ou dégénéré |

Chaque critère produit une ligne `✔` (passe) ou `❌` (échoue). Il suffit qu'un
critère échoue pour que le verdict final soit **ÉCHEC** (`exit code 1`).

---

## 3. Usage

```bash
# Depuis la racine du projet Projet-MSDP :
python3 check_fortran_pipeline.py                        # vérifie le dernier run (data/output/ACDF2.lis)
python3 check_fortran_pipeline.py data/output/ACDF2_run_009.lis        # un run précis
python3 check_fortran_pipeline.py data/output/ACDF2.lis data/output/ms_par_run_011.par  # + affiche les params
./check_fortran_pipeline.py                              # (exécutable directement)
```

### Arguments (optionnels)
| Arg | Défaut | Rôle |
|-----|--------|------|
| 1 – `ACDF2.lis` | `data/output/ACDF2.lis` | fichier de coins à contrôler |
| 2 – `ms_par` | — | affiché en tête de rapport (aucune influence sur le verdict) |

---

## 4. Sortie et code de retour

- **Exit `0`** → `SUCCÈS : la géométrie du pipeline Fortran est cohérente.`
- **Exit `1`** → `ÉCHEC : des anomalies ont été détectées.`

Permet une **intégration dans un flux CI / cron** : le script peut être appelé
après `run_both_pipeline.sh` pour décider si le run est exploitable.

---

## 5. Exemples réels

### Run **valide** (009, après correction) → exit 0
```
✔  9 canaux
✔  Centres X strictement croissants
✔  Pas inter-canaux régulier (150.6 px/ canal, σ=2.23%)
✔  Largeurs cohérentes (122.6 px, σ=0.45%)

SUCCÈS : la géométrie du pipeline Fortran est cohérente.
```

### Run **invalide** (001, avant correction) → exit 1
```
✔  9 canaux
❌ 16 valeur(s) nulle(s) (canal non détecté ou énantiomère)
❌ Centres X non croissants (2 décroissance(s))
❌ Pas inter-canaux irrégulier (rel.stdev=238.89%)
❌ Largeurs de canaux incohérentes (rel.stdev=95.50%)

ÉCHEC : des anomalies ont été détectées.
```
Le message « Largeurs incohérentes » de la version invalide correspond
**exactement** au symptôme du bug d'aliasing qu'on avait corrigé.

---

## 6. Structure du code

```
run_checks(path) -> (ok, messages)   # exécute tous les contrôles
  |-- parse_acdf2(path)              # lit 9x8 floats
  |-- stddev(xs)                     # écart-type
parse handler in main()              # affichage du rapport + exit code
```

Constantes modulables en tête de fichier (si l'instrument change) :
`NM_EXPECTED`, `IM_PIX`, `JM_PIX`, `MAX_ABS`, `CTR_X_TOL_FRAC`, `WIDTH_TOL_FRAC`.

---

## 7. Limites connues

- Le script est **statique** : il ne compare pas au flat/dark d'origine, ni aux
  plots `geo*.ps` — il valide la cohérence *interne* du fichier `ACDF2.lis`.
- Les tolérances (`25 %`/`35 %`) sont calibrées sur les données Meudon actuelles
  (9 canaux, Hα 6563 Å). Pour un instrument radicalement différent, ajuster les
  constantes en tête de fichier.
- Il ne vérifie **pas** la physique des valeurs absolues (positions en arcsec),
  seulement leur **régularité** — c'est le critère le plus discriminant pour
  détecter les bugs de portage/aliasing vus plus tôt.