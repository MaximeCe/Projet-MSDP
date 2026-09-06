# Projet MSDP — Réduction de données spectro-héliographiques

Réduction des données du **MSDP** (Multichannel Subtractive Double Pass) du
Spectrohéliographe de Meudon (BSL, Hα 6562.8 Å) : moyennage dark/flat, calcul de
la géométrie des canaux, et extraction des coordonnées des coins (A, C, D, F).

Le projet fournit deux implémentations **équivalentes et validées croisées** :
- une référence **Fortran** (`src/fortran/`, original de Meudon) ;
- une reprise **Python** (`src/python/`, même algorithme) dont la sortie diffère de
  < ~1.4 px (décalage documenté du bug `1.-ac` vs `1-a*c` dans `intersec`).

---

## Structure

```
Projet-MSDP/
├── src/
│   ├── fortran/              # Code Fortran (référence)
│   │   ├── ms1.f             # Step 1 : moyennage dark/flat + lecture ms.par
│   │   ├── ms2.f             # Step 2 : géométrie des canaux (newgeom)
│   │   ├── ms.par            # Paramètres Fortran (a8,i8)
│   │   └── save/             # Versions originales (avant nettoyage/re-commentaire)
│   └── python/
│       ├── ms1.py            # Step 1 : moyennage (équivalent ms1.f)
│       ├── ms2.py            # Step 2 : géométrie (équivalent ms2.f newgeom)
│       ├── ms.yml            # Paramètres Python (équivalent ms.par)
│       ├── auto_params.py    # Estimation automatique des paramètres depuis le flat
│       └── bass2000/         # Client de téléchargement BASS2000
├── run_pipeline.sh           # Pipeline Fortran complet (compile + exécute)
├── run_pipeline_py.sh        # Pipeline Python complet
├── run_both_pipeline.sh      # Les deux, avec sorties versionnées
├── run.sh                    # Wrapper simple ms1.py → ms2.py (avec symlinks)
├── ingest_all.py             # Orchestrateur : télécharge → estime → run → check → CSV
├── check_fortran_pipeline.py # Validation géométrie Fortran
├── check_python_pipeline.py  # Validation géométrie Python
├── test_regression.sh        # Filet de régression (compare ACDF2 à une référence)
├── data/
│   ├── input/                # Données FITS téléchargées / maîtres
│   └── output/               # Sorties : ACDF2, geo*.pdf, ms.lis, logs versionnés
├── docs/
│   ├── specification-multi-instrument.md   # Spéc du pipeline multi-instrument (VOLeTS)
│   ├── prototype-nm-estimation.md          # Preuve de faisabilité (auto nm)
│   ├── ingest-all.md                       # Doc de l'orchestrateur
│   ├── fortran/             # Docs du code Fortran (ms1, ms2, ms.par, ms.lis)
│   │   └── check-*.md, bass2000-download.md
│   └── MSDP-*.md, sources/, media/         # Guides Pierre Mein
└── README.md
```

---

## Environnement

### Python
Dépendances : `numpy`, `pyyaml`, `astropy`, `matplotlib` (et `requests` pour
BASS2000). Un venv partagé existe à `/home/max/nextcloud/Workspace/.venv/`.

```bash
source ~/.venv/bin/activate
```

### Fortran
Compile avec `gfortran` + **PGPLOT** (tracés). `run_pipeline.sh` gère la
compilation et le lien PGPLOT automatiquement.

---

## Pipelines

### Pipeline Python seul (`ms1.py` → `ms2.py`)
```bash
# depuis src/python/, avec les données dans data/input/
python ms1.py   # Step 1 : moyennage dark/flat
python ms2.py   # Step 2 : géométrie
```
> `ms1.py` lit `m*x1.fit` (dark) et `m*y1.fit` (flat). Les fichiers moyens
> `*_00000` sont produits puis consommés par `ms2.py`. `run.sh` automatise ça.

### Pipelines complets (recommandé)

Les scripts `run_pipeline*.sh` compilent/configurent et lancent sur les données
de `data/input/`, avec sorties versionnées dans `data/output/` :

```bash
./run_pipeline.sh                     # Fortran seul
./run_pipeline_py.sh                  # Python seul
./run_both_pipeline.sh                # les deux (logs ms_run_N partagés)
./run_pipeline.sh /chemin/ms.par      # config custom
./run_pipeline_py.sh /chemin/ms.yml
```

### Orchestrateur d'ingestion (multi-datasets, robustesse)
```bash
python3 ingest_all.py --local              # traite data/input/ (sans téléchargement)
python3 ingest_all.py --dates 2015-06-04   # télécharge BASS2000 + ingère
```
Fait automatiquement : téléchargement → estimation auto_params → run des 2
pipelines → validation → **CSV `data/output/ingest.csv`** (dont écarts
Fortran/Python) → cleanup.

---

## Validation & robustesse

- **`check_fortran_pipeline.py` / `check_python_pipeline.py`** : vérifient la
  cohérence interne de la géométrie (nm canaux, pas régulier, pas de zéros/NaN).
  Exit `0` = succès. Paramétrables (nm, im, jm).
- **`test_regression.sh`** : exige que `run_both_pipeline.sh` reproduise un
  `ACDF2` de référence (filet contre les régressions).

```bash
./check_fortran_pipeline.py data/output/ACDF2_run_NNN.lis
./test_regression.sh --update   # 1re fois : enregistrer la référence
./test_regression.sh            # ensuite : vérifier la régression
```

---

## Paramétrisation multi-instrument (VOLeTS A2/A2-bis)

Le pipeline était initialement figé sur Meudon (9 canaux, CCD 1536×1024). Il est
désormais **paramétrable** depuis les fichiers de config :

| Param | ms.par / ms.yml | Rôle |
|-------|-----------------|------|
| `nm` | lu | nombre de canaux |
| `ja1, ja2, ja3` | lu | positions des 3 coupes de détection |
| `xdel`, `jtriple` | lu | détection des bords courts |
| `im`, `jm` | par header flat (A2-bis) | dimensions image |
| `mingrad`, `interp` | lu | seuil de gradient / interpolation |

`auto_params.py` estime `nm`, `im`, `jm`, `interc` automatiquement depuis un flat
(la position exacte de `ja` reste à valider par la boucle de correction de
`ingest_all.py` — constat documenté dans la spec).

Voir `docs/specification-multi-instrument.md` pour le détail des volets (A1, A2,
A2-bis, A-auto, B) et leur état.

---

## Bugs corrigés

### 1. Aliasing mémoire `newgeom` (`xx(24,24)` vs `xx(20,9)`) — ms2.f
`newgeom` remplissait `xx(24,24)` mais les `plotgeo*`/`ACDF2` lisaient via des
dummy `xx(20,9)`. En Fortran column-major, ça décalait la mémoire → canaux 4-6
corrompus (`0.00`/`NaN`). **Corrigé** : déclarations alignées `xx(20,40)`.

### 2. Normalisation des coupes verticales — port Python (ms2.py)
Dans la reprise Python de `newgeom`, les points verticaux k,m utilisaient la
normalisation globale au lieu de recalculer `zmax`/`zgmax` par colonne, faussant
les coins A,D. **Corrigé** : normalisation locale par colonne.

### 3. Décalage d'index k,l,m,n — ms2.py
Les points k,l,m,n étaient écrits aux lignes 7-10 (Fortran 1-based) au lieu de
6-9 (numpy 0-based), décalant tous les coins. **Corrigé** : écriture à `l-1`.

### 4. `im`/`jm` en dur — A2-bis (ms2.f)
`newgeom` écrasait `im=1536/jm=1024` inconditionnellement. **Corrigé** :
dimensions reçues en argument depuis le header flat (bornes tableau 4096²).

### 5. Garde-fou débordement `newgeom` (n>40)
Un `ja` mal positionné pouvait générer >40 bords → débordement `xx`/`iedge` →
segfault. **Corrigé** : borne `n>40`.

### 6. Downloader BASS2000
Imports manquants (`download_file`, `download_sequence`) et `limit=` passé en
position d'un callback → téléchargement planté. **Corrigé**.

---

## Données de test

- Meudon 2017-03-30 : `m010...x1.fit` (dark), `m011...y1.fit` (flat),
  `m012...b1.fit` (observations). Sauvegarde dans `backup_meudon_input/`.
- BASS2000 téléchargeables (ex. 2015-06-04 : flats, darks, observations).

---

## Docs associées

- `docs/specification-multi-instrument.md` — la référence du pipeline multi-instruments.
- `docs/fortran/ms1.md`, `ms2.md`, `ms.par.md`, `ms.lis.md` — analyse du code Fortran.
- `docs/fortran/check-*.md`, `docs/ingest-all.md`, `docs/prototype-nm-estimation.md`.
- Guides scientifiques de Pierre Mein : `MSDP-methods-2024-02`, `MSDP-geometry-2024-12`.