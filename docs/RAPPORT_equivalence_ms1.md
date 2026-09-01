# Rapport d'équivalence — ms1.py / ms1.f (DPSM-16)

**Statut : ÉQUIVALENCE VALIDÉE** (après correction d'un bug d'axe)

- **Référence Fortran :** `DPSM-Fortran/ms1.f` (original V1) — identique à `exec-fortran/ms1.f` modulo CRLF.
- **Port Python :** `Literal translation/ms1.py`.
- **Données :** `m010_b0101_ms_20170330_09564585_x1.fit` (dark) et `m011_b0101_ms_20170330_10013140_y1.fit` (flat), entrées identiques de `DPSM-Fortran/`.
- **Référence de sortie :** `x170330_09564585_00000` / `y170330_10013140_00000` produits par le binaire `ms_pipeline` compilé (dans `DPSM-Fortran/`), paramètres réels de `ms.par` : `nfx1=1,nfx2=1,nfy1=1,nfy2=1`, `is=1536`, `js=1024`.

## Résultat

| Sortie | Taille analogue | Diff valeur | Coins (dark/flat) | Verdict |
|--------|----------------|-------------|-------------------|---------|
| Dark  (`x170330_...`) | 1536×1024 int16 | **0** / 1 572 864 px | 62 60 60 60 | ✅ Identique |
| Flat  (`y170330_...`) | 1536×1024 int16 | **0** / 1 572 864 px | 79 73 81 61 | ✅ Identique |

Les **valeurs** des tableaux issus de ms1.py corrigé sont **strictement égales** à celles du Fortran (diff absolue max = 0). Les fichiers ne sont pas octet-à-octet identiques (voir « Format binaire ») mais ce n'est pas exigé : l'objectif est l'équivalence des calculs/variables.

## Bug corrigé dans ms1.py : inversion d'axes dans `permute_data`

**Symptôme :** crash `IndexError: index 1024 is out of bounds for axis 0 with size 1024`, puis (si contourné) données corrompues.

**Cause :** `astropy.io.fits` retourne le flux FITS comme `data[NAXIS2, NAXIS1]` = `(js, is)` en ordre ligne-major. Le Fortran, lui, lit `tab2(i, j)` avec `i` = colonne (is=1536, sens fast) et `j` = ligne (js=1024). Le code `permuted[ip, jp] = data[i, j]` indexait `data` avec `i` en 1er axe, alors que le 1er axe numpy est `j` (ligne).

**Correct :**
```python
ip = self.js_ccd - 1 - j   # rotation 180° de l'axe ligne (js+1-j en 1-indexé)
permuted[ip, jp] = data[j, i]   # data[ligne=j, colonne=i]
```
(position `ip,jp` inchangée ; seule l'indexation de `data` passe de `[i,j]` à `[j,i]`).

## Divergence config : `iswap`

- `ms1.f` code en dur `iswap=1` (swap d'octets **toujours** appliqué) car il lit les octets bruts big-endian du FITS.
- `ms1.py` lit via astropy qui **déjà** normalise l'endianness (`>i2` → valeurs correctes en natif). Reproduire le swap via `data.byteswap()` **corrompt** les valeurs (62 → 15872).
- **Conclusion :** pour que ms1.py (astropy) produise les mêmes valeurs que ms1.f, il faut `iswap: 0` dans `ms.yml`. Ce n'est pas un bug du port (le docstring de `read_fits_file` l'explique) mais une **différence sémantique Fortran ↔ astropy** à documenter. Laisser `iswap: 1` donne des valeurs erronées.

## Format binaire (différence non bloquante)

- Sortie **Fortran** : Fichier séquentiel non formaté gfortran — 1 record en-tête (512 × int32) + 1536 records de 1024 int16, chaque record précédé/suivi d'un marqueur de longueur int32 (donc 3 160 072 o).
- Sortie **Python** : `header.tofile` + `data[:,j].tofile` → en-tête 2048 o + 1536 × 2048 o de données brutes (3 147 776 o), **sans** marqueurs de record.
- Les **valeurs numériques** (le contenu) sont identiques ; seule la couche de sérialisation change.

## Validation reproductible

`validate_ms1.py` (voir fichier joint) : lit la sortie brute ms1.py et la référence Fortran, compare les valeurs, affiche la diff max. Usage :
```
python validate_ms1.py x170330_09564585_00000  <ref_fortran>
```
Résultat attendu : `IDENTIQUE valeur: True`.