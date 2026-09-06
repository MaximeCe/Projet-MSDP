# `check_python_pipeline.py` — Vérification de la réussite du pipeline Python

> Chemin : `Projet-MSDP/check_python_pipeline.py`
> Rôle : **valider que les RÉSULTATS du pipeline Python MSDP (ms1.py + ms2.py)
> sont bons**, pas seulement que le pipeline s'est exécuté sans erreur.

---

## 1. Pourquoi ce script ?

Comme le pipeline Fortran, le pipeline Python (`run_pipeline_py.sh` → ms1.py +
ms2.py) peut **se terminer sans erreur** tout en produisant une géométrie ratée :
canaux non détectés (`0.00`), `NaN` (overflow dans `intersect_lines`), largeurs
repliées.

Ce script contrôle le fichier **`ACDF2.lis`** produit par ms2.py et garantit que
la géométrie des 9 canaux est **physiquement cohérente**, sur la base des
**mêmes critères internes que `check_fortran_pipeline.py`** — il ne compare pas
au pipeline Fortran, il juge chaque run à sa propre cohérence.

---

## 2. Critères de validation

Le script vérifie exactement les mêmes invariants géométriques que la version
Fortran :

| # | Critère | Ce qu'il détecte |
|---|---------|------------------|
| 1 | **Format** : 9 lignes × 8 valeurs numériques | fichier absent / mal écrit |
| 2 | **Pas de valeur nulle** (0.00) | canal non détecté / dégénéré |
| 3 | **Pas de NaN/infini** | overflow dans `intersect_lines` |
| 4 | **Bornes physiques** (X,Y ∈ [0,1536]) | coordonnées hors image |
| 5 | **Centres X croissants** + **pas régulier** (σ < 25 %) | canaux mélangés / non ordonnés |
| 6 | **Largeurs de canal cohérentes** (σ < 35 %) | canal replié, étiré, dégénéré |

Chaque critère produit une ligne `✔` (passe) ou `❌` (échoue). Un seul échec
suffit pour que le verdict final soit **ÉCHEC**.

---

## 3. Usage

```bash
# Depuis la racine du projet :
python3 check_python_pipeline.py                                          # dernier run (data/output/ACDF2.lis)
python3 check_python_pipeline.py data/output/ACDF2_run_012.lis            # un run précis
python3 check_python_pipeline.py data/output/ACDF2_run_012.lis data/output/ms_par_run_012.yml
./check_python_pipeline.py                                                # exécutable
```

### Arguments (optionnels)
| Arg | Défaut | Rôle |
|-----|--------|------|
| 1 – `ACDF2.lis` | `data/output/ACDF2.lis` | fichier de coins à contrôler |
| 2 – `ms.yml` | — | affiché en tête de rapport (aucun impact sur le verdict) |

---

## 4. Sortie et code de retour

- **Exit `0`** → `SUCCÈS : la géométrie du pipeline Python est cohérente.`
- **Exit `1`** → `ÉCHEC : des anomalies ont été détectées.`

Intégrable en CI / cron / dans `run_both_pipeline.sh`.

---

## 5. Exemples réels

### Run Python **valide** (012) → exit 0
```
✔  9 canaux
✔  Centres X strictement croissants
✔  Pas inter-canaux régulier (150.7 px/ canal, σ=2.23%)
✔  Largeurs cohérentes (122.6 px, σ=0.47%)

SUCCÈS : la géométrie du pipeline Python est cohérente.
```

### Fichier **invalide** (canaux 4-5 nuls) → exit 1
```
✔  9 canaux
❌ 16 valeur(s) nulle(s) (canal non détecté)
❌ Centres X non croissants (2 décroissance(s))
❌ Pas inter-canaux irrégulier (rel.stdev=247.88%)

ÉCHEC : des anomalies ont été détectées dans la géométrie Python.
```

---

## 6. Structure du code

```
run_checks(path) -> (ok, messages)      # tous les contrôles internes
  |-- parse_acdf2(path)                 # lit 9×8 floats
  |-- stddev(xs)                        # écart-type
main()                                  # affichage du rapport + exit code
```

Constantes modulables : `NM_EXPECTED`, `IM_PIX`, `JM_PIX`, `MAX_ABS`,
`CTR_X_TOL_FRAC`, `WIDTH_TOL_FRAC`.

---

## 7. Limites connues

- Validation **statique & interne** : ne ré-ouvre pas les images, ne compare pas
  aux plots `geo*.pdf`. Elle valide uniquement la **cohérence interne** de
  `ACDF2.lis — c'est la même approche que pour le pipeline Fortran.
- Pour comparer explicitement un run Python au run Fortran correspondant,
  utiliser la différence entre `ACDF2_run_*.lis` (la version Python est
  mathématiquement plus juste sur `intersec`, écart de ~0.1–1.4 px documenté
  dans `ms2.py`).
- Les tolérances (`25 %`/`35 %`) sont calibrées sur les données Meudon 9 canaux ;
  pour un autre instrument, ajuster les constantes en tête de fichier.