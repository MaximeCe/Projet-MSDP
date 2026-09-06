# Spécification — Pipeline MSDP multi-instruments

> **Statut** : version 1.0 (2026-09-05)
> **Objet** : rendre le pipeline MSDP (Fortran + Python) capable de tourner de
> **façon automatisée sur des jeux de données variés** (capteurs, spectrographes,
> résolutions, nombre/taille/espacement de canaux différents), avec détermination
> **automatique** des paramètres qui, aujourd'hui, sont réglés à la main.
>
> Ce document est la **référence** pour :
> - la refactorisation de `ms2.f` / `ms2.py` (rendre les constantes paramétrables) — **Volets A2**,
> - la construction du module d'auto-détermination `src/python/auto_params.py` — **Volets A-auto**,
> - la boucle d'ingestion multi-dates + CSV — **Volets B**.
>
> La **faisabilité du point le plus risqué** (estimation automatique de `nm`)
> est prouvée dans `docs/prototype-nm-estimation.md`.

---

## 1. Contexte et constat

Le pipeline actuel est calibré pour **Meudon : 9 canaux, CCD 1536×1024**. Les
configurations réellement déterminantes de la géométrie sont **encodées en dur**
dans le code :

| Paramètre | Valeur en dur | Lieu |
|---|---|---|
| `im` (dim CCD X) | 1536 | `ms2.f` newgeom |
| `jm` (dim CCD Y) | 1024 | `ms2.f` newgeom |
| `nm` (nb canaux) | 9 | `ms2.f` newgeom, `ms2.py` |
| `ja` (3 coupes) | 151, 501, 851 | `ms2.f` newgeom, `ms2.py` |
| `i1` (bande utile) | 5 | `ms2.f` newgeom |
| `i2` | im-4 | `ms2.f` newgeom |
| `jc` (coupe centrale) | ja(2)=501 | `ms2.f` newgeom |
| `xdel` (décalage klmn) | 25 | `ms2.f` newgeom, `ms2.py` |

Les paramètres `ms.par` qui ont un **impact réel** dans le chemin de détection
actif (`newgeom`) sont **seulement** :
`mingrad` (critique), `interp` (affinage), et en amont `idc` (soustraction dark).

La plupart des autres (si, sg*, milangi, milgeo, lip, jeps, intvi, intvj, leps,
n1, distor, interc) sont **lus mais morts / non utilisés** dans le chemin newgeom.

**Conséquence** : une boucle automatique « sur toutes les dates » échouerait pour
la majorité des instruments tant qu'on n'a pas :
1. rendu le pipeline paramétrable (im, jm, nm, ja, ...) ; puis
2. déterminé ces paramètres automatiquement.

---

## 2. Objectifs

La spécification poursuit 3 objectifs granulaires :

- **O1 — Paramétrisation** : sortir `im, jm, nm, ja, i1, i2, xdel, jtriple` des
  constantes en dur, avec **défaut = Meudon actuel**, sans casser les sorties
  actuelles (ACDF2 byte-identique au run de référence).
- **O2 — Auto-détermination** : un module `auto_params.py` estime, depuis un
  flat (+ dark + header FITS), les paramètres à fixer, et les **écrit** dans
  `ms.par` (Fortran) et `ms.yml` (Python).
- **O3 — Boucle d'ingestion** : orchestrer téléchargement → auto_params → run
  des 2 pipelines → check des 2 → CSV → suppression de la date, sur toutes les
  dates disponibles.

---

## 3. Architecture cible

```
data/input/                 # documents téléchargés (.fit) + dark/flat
src/python/auto_params.py   # estimation + écriture ms.par / ms.yml  (NOUVEAU)
src/fortran/ms1.f ms2.f     # refactorisés : constantes → ms.par
src/python/ms1.py ms2.py    # refactorisés : constantes → ms.yml
check_fortran_pipeline.py   # paramétrable (nm, im, jm)
check_python_pipeline.py    # paramétrable (nm, im, jm)
run_pipeline.sh             # (déjà ajuste nfx/nfy) — inchangé structure
run_pipeline_py.sh          # idem
run_both_pipeline.sh        # idem
ingest_all.py               # ORCHESTRATEUR : boucle dates + CSV + cleanup  (NOUVEAU)
```

---

## 4. VOLE T A2 — Refactorisation (condition nécessaire)

### 4.1 Principe
Chaque constante en dur devient un paramètre **lu de config**, avec une **valeur
par défaut strictement égale** à la valeur actuelle (Meudon). Ainsi, sans config
custom, les sorties restent identiques au run de référence (régression-zéro).

### 4.2 Tableau des nouveaux paramètres (à ajouter à `ms.par` et `ms.yml`)

| Param | ms.par | ms.yml | Défaut (Meudon) | Rôle |
|---|---|---|---|---|
| `im` | `im` | `im` | 1536 | dimension X image (newgeom) |
| `jm` | `jm` | `jm` | 1024 | dimension Y image (newgeom) |
| `nm` | `nm` | `nm` | 9 | nombre de canaux |
| `ja1, ja2, ja3` | `ja1..3` | `ja: [..]` | 151, 501, 851 | positions 3 coupes |
| `i1` | `i1` (existant) | `i1` (existant) | 5 | bande utile i (ne plus écraser) |
| `i2m` | `i2m` (existant) | `i2m` (existant) | 0 | borne i finie (i2=im-i2m) |
| `xdel` | `xdel` | `xdel` | 25 | décalage des coupes verticales klmn |
| `jtriple` | `jtriple` | `jtriple` | 1 | moyenne 3 lignes autour d'une coupe |

> **Valeurs par défaut = Meudon** : sans modification de config, comportement
> identique à l'actuel (régression-zéro vérifiée par diff ACDF2).

### 4.3 Dimensionnement des tableaux — décision requise

Deux options en Fortran 77 :

| Option | Description | Avantages | Inconvénients |
|---|---|---|---|
| **(a) bornes fixes maxi** | dim `xx(20,40)`, `ima(4096,4096)`, `meanflat(4096,4096)` | reste F77 pur, compile partout | mémoire sur-dimensionnée (~64 Mo/tableau à 4096²) |
| **(b) allocatable (F90+)** | `real*2, allocatable :: ima(:,:)` ; `allocate` eh im*jm | propre, pile réduite, gfortran OK | écart du style F77 ; exige gfortran (déjà le cas) |

**Recommandation** : **(b)** — le projet est déjà compilé avec `gfortran`
(`run_pipeline.sh`), donc F90+ est disponible. Adapter les sous-programmes qui
reçoivent ces tableaux (déclarer en `allocatable` et passer les bornes en
arguments).

### 4.4 Impact de refactorisation (liste de contrôle)

- [ ] `ms2.f geom()` : lire `im, jm, nm, ja1..3, i1, i2m, xdel, jtriple` via `par1`.
- [ ] `ms2.f newgeom()` : remplacer les constantes par les arguments/paramètres.
- [ ] `ms2.f SRECT()` : transit des nouvelles bornes (tableaux passés par argument).
- [ ] `ms1.f` : rien de structurel (il ne porte pas la géométrie) ; juste vérifier
      qu'`is`/`js` du main restent cohérents.
- [ ] `ms2.py` : lire `im, jm, ja, xdel, jtriple` de `ms.yml` (déjà `nm`).
- [ ] `check_fortran_pipeline.py` : `NM_EXPECTED`, `IM_PIX`, `JM_PIX` lus de config
      (ou argument) au lieu de constantes.
- [ ] `check_python_pipeline.py` : idem.
- [ ] **Régression-zero** : lancer `run_both_pipeline.sh` avec config par défaut et
      comparer `ACDF2_run_*.lis` au run de référence (009 Fortran / 012 Python).

### 4.5 Ordre d'exécution (boucles courtes, vérification à chaque étape)

1. Extraire `im, jm, nm, ja` → run + diff ACDF2 (doit être identique).
2. Extraire `i1, xdel, jtriple` → run + diff (identique).
3. Adapter `ms2.py` pareil → vérifier écart ~0.1-1.4 px vs Fortran.
4. Rendre les deux `check_*.py` paramétrables → tester valide/invalide.
5. Commit / documenter.

---

## 5. VOLE T A-auto — Module `auto_params.py`

### 5.1 Rôle
Branche **avant** le pipeline : analyse le flat (+ dark + header FITS) d'un jeu
de données et **écrit les paramètres** dans `ms.par` et `ms.yml`, pour que les
2 pipelines tournent sans réglage manuel.

### 5.2 Ce qu'il détermine et comment

| Paramètre | Méthode | Fiabilité |
|---|---|---|
| `im`, `jm` | header FITS (`NAXIS1/NAXIS2`) via astropy | ✅ triviale |
| `nm` | maxima locaux sur coupe à Y fixe + vote multi-lignes (voir prototype) | ✅ bonne, à affiner |
| espacement `interc` | médiane des écarts entre pics | ✅ bonne |
| `ja` (3 coupes) | 3 lignes équi-réparties dans la bande Y utile | ✅ bonne |
| `i1`, `i2` (bande utile) | zone du flat où le signal > seuil (projection) | ✅ bonne |
| `mingrad` | percentil des gradients (ex 90e) | ⚠️ heuristique — passe par la boucle de correction |
| `interp` | fixé à 1 | ✅ |
| `jtriple` | fixé à 1 | ✅ |

### 5.3 API

```python
# src/python/auto_params.py
def estimate_params(flat_path, dark_path, date=None) -> Mapping:
    """Retourne {im, jm, nm, ja:[..], i1, i2m, interc, mingrad, interp, xdel,
                 jtriple, metadata:{instrument?, confidence}}."""
    ...

class ConfigWriter:
    def write_ms_par(self, params, out="data/output/msrun.ms.par")   # format a8,i8
    def write_ms_yml(self, params, out="data/output/msrun.ms.yml")
```

`ms.par` généré doit être **complet** : reprendre la configuration de `ms.par`
de base et n'écraser que les champs déterminés (comportement identique aux
autres, mais avec `im/jm/nm/ja/...` renseignés).

### 5.4 Stratégie robuste : réglage itératif (le cœur)

`nm` et `mingrad` sont **ambiguës** (le prototype montre 9 vs 8 selon les
hyperparams). Une estimation mono-pass ne suffit pas. Architecture :

```
                        ┌────────────────────────────┐
   config estimée ───►  │ run_pipeline.sh            │
                        │ run_pipeline_py.sh         │
                        └─────────────┬──────────────┘
                                      ▼
                        ┌────────────────────────────┐
                        │ check_fortran_pipeline.py  │── ok ──► ACCTÉ (CSV)
                        │ check_python_pipeline.py   │
                        └─────────────┬──────────────┘
                                      │ échec
                                      ▼
                        ┌────────────────────────────┐
                        │ auto_params.scan_mingrad   │─► nouvelle config (seuil ≠)
                        │ + re-estimer nm si besoin  │
                        └────────────────────────────┘
```

Règles :
- **Scinder `mingrad`** sur une grille (ex 8..40) et garder la première config
  qui passe les checks.
- Si **nm** semble faux (échec persistant sur le nombre de canaux), ré-estimer
  avec un lissage/distance différents.
- Après **Nmax tentatives** (ex 6) sans succès : **échouer gracieusement**,
  garder la meilleure config dans le CSV (avec `check=fail`).

### 5.5 Métadonnées pour le CSV
Chaque itération expose : `date, instrument_détecté, im, jm, nm, ja_1..3,
mingrad, interc, interp, check_fortran(bool), check_python(bool),
écart_fortran_python, écart_max_ACDF2, n_tentatives, statut`.

---

## 6. VOLE T B — Orchestrateur `ingest_all.py` (+ CSV)

### 6.1 Boucle

```
pour chaque date dispo (BASS2000) :
   1. télécharger (download_bass2000.py --type all)
   2. auto_params.estimate + write ms.par/ms.yml
   3. run_pipeline.sh          (Fortran)
   4. check_fortran_pipeline.py
   5. (si échec) → boucle de correction auto_params
   6. run_pipeline_py.sh       (Python)
   7. check_python_pipeline.py
   8. écrire une ligne dans le CSV
   9. supprimer la date (les .fit de cette date dans data/input + sorties)
```

### 6.2 CSV cible
```csv
date,instr,im,jm,nm,ja1,ja2,ja3,mingrad,interc,interp,ck_fortran,ck_python,ecart_fp,ecart_max,n_try,status
2015-06-04,meudon9,1536,1024,9,151,501,851,18,15,1,True,True,1.36,0.00,1,ok
```

Le CSV et `auto_params.py` permettront de **cartographier les instruments**
(capteur/résolution/nombre de canaux) sur toutes les dates — l'objectif réel du
projet.

### 6.3 Robustesse de l'orchestrateur
- `set -euo pipefail` au niveau bash, `try/except` au niveau Python.
- Timeout par run (déjà présent : `timeout 120`).
- **Jamais de suppression avant vérification** que l'ingestion de la date est
  terminée (VOLE T 6.1 ordre 9).
- Journal des erreurs par date (pour reprendre après un crash au milieu).

---

## 7. Critères d'acceptation

1. **Régression zero** : sans config auto, `run_both_pipeline.sh` reproduit
   `ACDF2_run_009 / _012` à l'identique (spéc du VOLE T A2).
2. **`auto_params`** estime `im,jm,nm,ja,interc` correctement sur le flat Meudon
   (nm=9), et écrit un `ms.par`/`ms.yml` valides.
3. **Boucle de correction** corrige `mingrad`/`nm` automatiquement jusqu'à
   passage des checks (ou échec gracieux après Nmax).
4. **`ingest_all.py`** produit un CSV structuré sur ≥ 1 date, avec cleanup de la
   date, sans fichier orphelin.
5. La **doc** (`ms2.md`, `ms.par.md`, check-*.md) reste à jour.

---

## 8. Phases & livrables

| Phase | Contenu | Livrable | Statut |
|---|---|---|---|
| **A1** (ce doc) | spécification | `docs/specification-multi-instrument.md` | ✅ fait |
| **A2** | refactorisation ms2.f/.py + checks paramétrables | profs ms.par/ms.yml + regression-zero | **partiel** (voir journal) |
| **A2-bis** | rendre `im`/`jm` paramétrables (refonte SRECT+newgeom, bornes mémoires) | ms.par + tables élargies | ✅ fait (im/jm par header, bornes 4096²) |
| **A-auto** | module auto_params (estimation + correction) | `src/python/auto_params.py` | ✅ fait (esquisse correction) |
| **B** | orchestrateur boucle + CSV | `ingest_all.py` + CSV | ✅ fait (test local ok) |

### Journal d'avancement (option B — sécurisation)

| Date | Action | Résultat |
|---|---|---|
| 2026-09-05 | A2 : extraire `nm, ja1..3, xdel, jtriple` de ms.par (Fortran) | ✅ régression-zéro, nm=8 validé |
| 2026-09-05 | A2 : ms.yml + ms2.py lisent `ja, xdel, jtriple, im, jm` | ✅ équivalence 1.36 px |
| 2026-09-05 | B : check_fortran paramétrable (nm/im/jm depuis ms.par) | ✅ valide nm=8 |
| 2026-09-05 | B : check_python paramétrable (nm/im/jm depuis ms.yml) | ✅ valide nm=8 |
| 2026-09-05 | B : `test_regression.sh` créé (run both + compare ACDF2 réf) | ✅ régression OK |
| 2026-09-05 | **A2-bis** : `newgeom(meanflat,im,jm)` — im/jm en argument, meanflat→4096², transposition SRECT pilotée par dims | ✅ régression OK, mémoire 28 Mo |
| 2026-09-05 | **A-auto** : module `auto_params.py` (estime nm/im/jm/interc, écrit ms.par/ms.yml) + garde-fou `n>40` dans newgeom | ✅ nm=9, im/jm/interc fiables ; ja à valider |
| 2026-09-05 | **B** : orchestrateur `ingest_all.py` + boucle de correction ja/mingrad + CSV | ✅ test local : statut ok après 5 tries, CSV généré |

**`im`/`jm` paramétrables (A2-bis) : fait.** `newgeom` reçoit les dimensions en
argument (SRECT les déduit du header flat) ; les tableaux sont à bornes max 4096²
(int*2 = 32 Mo). Le pipeline s'adapte aux capteurs ≠1536×1024. La parité des runs
pour la régression = corrigée (2 runs les plus récents, pas la parité).

**A-auto : constat important sur `ja`.** `auto_params.py` estime fiablement
`nm`, `im`, `jm`, `interc` (nm=9 détecté, dimensions du header). En revanche, la
**position exacte des coupes `ja`** est sensible à l'instrument : une heuristique
locale sur le flat (équirépartition ou régularité de pics) **ne suffit pas** pour
garantir une géométrie valide. La phase A-auto doit donc **imperativement étre
adossée à la boucle de correction** (run du pipeline + validation ACDF2 + re-scan)
pour roder `ja`/`mingrad`, comme prévu en section 5.4. Sans elle, `ja` reste à
défaut. Garde-fou ajouté dans `newgeom` (`n>40`) pour éviter un segfault si un
ja mal positionné génère trop de bords.

Chaque phase a **ses propres boucles courtes** (une étape à la fois, vérification
après chaque), cohérent avec la méthode de travail du projet.

---

## 9. Risques & mitigations

| Risque | Mitigation |
|---|---|
| F77 vs F90 (allocatable) | déjà gfortran → option (b) ; sinon (a) bornes maxi |
| `nm` ambigu (9 vs 8) | vote + boucle de correction |
| canal saturé/éteint | tolérance + échec gracieux + trace CSV |
| `ja` hors des canaux (instrument décentré) | estimation auto de ja → critique |
| régression des runs actuels | regression-zero obligatoire à chaque étape A2 |
| cleanup trop agressif | suppression de date seulement après ingestion complète |

---

## 10. Références internes

- `docs/prototype-nm-estimation.md` — preuve de faisabilité de l'estimation nm.
- `docs/fortran/ms2.md` — code analysé (constantes en dur, newgeom).
- `docs/fortran/ms.par.md` — paramètres et lesquels sont morts/actifs.
- `docs/fortran/check-fortran-pipeline.md` / `check-python-pipeline.md` — checks à
  rendre paramétrables.