# Projet MSDP — Réduction de données spectro-héliographiques

Réduction des données du **MSDP** (Multichannel Subtractive Double Pass) du
Spectrohéliographe de Meudon (BSL, Hα 6562.8 Å).

## Structure

```
Projet-MSDP/
├── src/
│   ├── fortran/            # Code Fortran original (référence)
│   │   ├── ms1.f           # Step 1: dark/flat averaging + param parsing
│   │   ├── ms2.f           # Step 2: channel geometry detection
│   │   └── ms.par          # Fichier de paramètres Fortran
│   └── python/             # Traduction Python (équivalence validée)
│       ├── ms1.py          # Step 1: averaging
│       ├── ms2.py          # Step 2: geometry detection
│       ├── ms.yml          # Paramètres (équivalent ms.par)
│       ├── config.yml      # Config détaillée (alternative, ms.par style)
│       └── config_example.yml  # Config commentée complète
├── data/
│   ├── input/              # Données FITS (brutes, masters)
│   │   ├── *.fits, *.fit   # Images scientifiques
│   │   ├── lights/         # Images d'observation (scans)
│   │   └── master/         # Darks/Flats master (Darks.fits, Flats.fits)
│   └── output/             # Sorties générées
│       ├── geo*.pdf        # Plots de diagnostic géométrie
│       ├── ACDF2.lis       # Coordonnées coins A C D F par canal
│       ├── ms.lis          # Log de traitement
│       └── ms1_out.txt     # Sortie stdout ms1
├── docs/
│   ├── MSDP-geometry-2024-12.md   # Guide géométrie (Pierre Mein)
│   ├── MSDP-methods-2024-02.md    # Méthodes de traitement
│   ├── RAPPORT_equivalence_ms1.md # Rapport d'équivalence Fortran↔Python
│   ├── sources/            # Originaux ODT/PDF des docs
│   └── media/Pictures/     # Images extraites des docs
└── README.md
```

## Environnement

### Python (recommandé)
Le `.venv` est à la racine de `Workspace/` (shared) et contient les dépendances :

```bash
source ~/.venv/bin/activate  # ou /home/max/nextcloud/Workspace/.venv/bin/activate
```

Dépendances installées : `numpy`, `pyyaml`, `astropy`, `matplotlib`.

Lancer depuis `src/python/` avec les données dans `data/input/` :
```bash
cd Projet-MSDP/src/python
python ms1.py  # Step 1: dark/flat averaging
python ms2.py  # Step 2: geometry detection
```

> **Note** : `ms2.py` attend `dark_2015.fits`/`flat_2015.fits` dans le répertoire
> courant. C'est un chemin en dur dans `main()`. Utilisez le script wrapper
> `run.sh` pour symlinker les fichiers automatiquement.

### Fortran
Le code Fortran original compile avec `gfortran` mais nécessite PGPLOT
pour les tracés (non inclus). Utile pour référence/validation croisée.

```bash
cd src/fortran
gfortran -c ms1.f -o ms1.o   # compile
gfortran -c ms2.f -o ms2.o   # compile (sans PGPLOT, le lien échoue)
```

## Géométrie & bugs corrigés

### Bug `newgeom` : normalisation des coupes verticales (ms2.f)
Les points verticaux k,m servant au calcul des coins A,D (extrapolation
via `intersec`) échouaient pour les canaux 4-6, produisant des `0.00` / `********`
dans `ACDF2.lis`.

**Cause** : dans `newgeom` (SRECT), lors de la détection des bords verticaux
(l=7..10, points k,l,m,n), les variables `zmax`/`zgmax` servant à normaliser le
gradient vertical n'étaient **jamais recalculées** pour chaque coupe — les lignes
`zmax=0.`, `zmax=amax1(...)`, `zgmax=0.`, `piv=abs(...)`, `zgmax=amax1(...)`
étaient toutes commentées. La normalisation utilisait donc les valeurs globales
de la coupe horizontale centrale (zmax≈2072, zgmax≈668), faussant le seuil applicatif
de `mingrad` sur les colonnes à gradient faible.

**Correction** : décommenté les 5 lignes (ms2.f §détection des points k,l,m,n)
afin de recalculer `zmax`/`zgmax` **par coupe verticale** avant normalisation.

## Pipeline de traitement

1. **ms1.py / ms1.f** — Moyennage darks + flats
   - Lit les images FITS `m*x1.fit` (dark) et `m*y1.fit` (flat)
   - Produit les fichiers moyennés (binaire ou FITS)
2. **ms2.py / ms2.f** — Détection géométrique des canaux
   - Lit dark + flat moyennés
   - Détecte les bords des 9 canaux spectraux
   - Produit `geo1-3.pdf` (diagnostic) et `ACDF2.lis` (coins A C D F)
3. **Étape suivante** (non implémentée ici) — Calibration spectrale + extraction

## Données de test

Les FITS d'exemple proviennent du spectrohéliographe Meudon :
- `m010...x1.fit` — dark 2017-03-30
- `m011...y1.fit` — flat 2017-03-30
- `m002...b1.fit` — observations 2015-06-04
- `DPSM_8bd.fits` — image multi-canaux complète
