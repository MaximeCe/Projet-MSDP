# Nouveau pipeline MSDP — Documentation (dossier `src/fortran/new/`)

## Vue d'ensemble

Le dossier `src/fortran/new/` contient la version complète et étendue du pipeline Fortran MSDP, intégrant les **4 étapes** décrites dans la méthode Mein (doc/MSDP-methods-2024-02.md) :

| Étape | Description | Fichier(s) | Statut |
|-------|-------------|------------|--------|
| **1** | Moyennage dark/flat | `ms1.f` | ✅ Complet |
| **2** | Géométrie + Calibration | `ms2.f` | ✅ Complet (étendu) |
| **3** | Calibration λ + intensité / c-files | `ms2.f` (`calib`, `channels`) + `ms3.f` | ✅ Intégré |
| **4** | d-files (profils I/V, vitesses) | `ms3.f` (`ivmap3`) + `ms4.f` (`ivmap4`) | ✅ Intégré |

## Fichiers du pipeline

| Fichier | Lignes | Rôle principal |
|---------|--------|----------------|
| [`ms1.f`](ms1.md) | 694 | Programme principal, orchestration 1→2→3→4 |
| [`ms2.f`](ms2.md) | 3248 | Géométrie, canaux, calibration, observations, I/V maps |
| [`ms3.f`](ms3.md) | 374 | Interpolation degré 3 → `ivprof1.ps` |
| [`ms4.f`](ms4.md) | 477 | Interpolation degré 4 → `ivprof2.ps`, `ivprof3.ps` |

## Compilation

```bash
gfortran -g -o msdp ms1.f ms2.f ms3.f ms4.f \
    -lpgplot -L/usr/lib/x86_64-linux-gnu -lX11 \
    -lgfortran -lquadmath
```

Le script `run_pipeline.sh` (dans `Avancement_MSDP/scripts/`) compile et exécute automatiquement les 4 fichiers.

## Flux de données complet

```
ms1.f                          ms2.f                          ms3/4.f
─────────────────────────────────────────────────────────────────────
Lecture ms.par                 ──────────────────────────────────
   │
   ├─ nxy=1: darks (*x1.fit)    │
   │     moyenne → unité 31     │
   ├─ nxy=2: flats (*y1.fit)    │
   │     moyenne → unité 32     │
   │                            │
   ▼                            ▼
call geom(...)         ─────► SRECT → newgeom
   │                       │         │
   │                       │         └─ xr,yr (coins ABCDEF)
   │                       ▼
   │                  channels()     ──► cymx(i,j,n)  canaux bruts
   │                       │
   │                       ▼
   │                  calib()          ──► cal(i,j,n)  calibration
   │                       │
   ▼                       ▼
call solarobs(...)  ─────► lecture *b1.fit
   │                       soustraction dark
   │                       channels() → sobs
   │                       calobs()   → sobs calibré
   │                       stockage sobstot(i,j,n,nfb)
   │                       │
   ▼                       ▼
call ivmap3()        call ivmap4()
   │                       │
   ├─ DPMCAR deg 3      ├─ init: copy deg3→slot3
   ├─ profnf(...,1)     ├─ DPMCAR deg 4 (fenêtres 5 pts)
   ├─ ivprof1.ps        ├─ profnf(...,2) deg 4 pur
   └─ intvel3()         ├─ profnf(...,3) mix deg3/4
                        ├─ ivprof2.ps (deg 4)
                        └─ ivprof3.ps (mixte)
                        └─ intvel4() + intvel3()
```

## Sorties principales (dans `data/output/` versionnées)

### Géométrie (étape 2)
- `geo1_fortran_NN.pdf` — Bords détectés (3 coupes)
- `geo2_fortran_NN.pdf` — Coins ABCDEF
- `geo3_fortran_NN.pdf` — Distorsions AC/DF/CF
- `geo4_fortran_NN.pdf` — Bords interpolés
- `ACDF2_run_NNN.lis` — Coins 9 canaux × 8 valeurs (AₓAᵧ CₓCᵧ DₓDᵧ FₓFᵧ)

### Canaux & Calibration (étape 3)
- `flat2_fortran_NN.pdf` — Canaux extraits (via `map3`)
- `calib_fortran_NN.pdf` — Profils raies + centres
- `cal_fortran_NN.pdf` — Maps calibration par canal

### Observations (étape 4)
- `obs_fortran_NN.pdf` — Observations brutes
- `obsD1_fortran_NN.pdf` — Non calibrées
- `obsD2_fortran_NN.pdf` — Calibrées

### I/V Maps (étapes 3-4)
- `ivprof1_fortran_NN.pdf` — Profils interpolation degré 3
- `ivprof2_fortran_NN.pdf` — Profils interpolation degré 4
- `ivprof3_fortran_NN.pdf` — Profils mixtes degré 3/4
- `miv.lis` — Vitesses bisector (format: nfb, lbdvel, xv2, yv2)

## Paramètres ms.par critiques (nouveaux vs ancien)

| Paramètre | Section | Défaut | Rôle |
|-----------|---------|--------|------|
| `mingrad` | newgeom | 18 | Gradient min détection bords |
| `interp` | newgeom | 1 | Interpolation parabolique |
| `ja1/ja2/ja3` | newgeom | 151/501/851 | Positions 3 coupes horizontales |
| `icalib` | ms1/ms2 | 1 | Active calibration complète |
| `ivprof1/2/3` | ms3/ms4 | 0 | Génère plots I/V |
| `lbdvel1/2/3` | ms3/ms4 | 0 | Décalage λ pour vitesse |
| `nfb1/nfb2` | ms1 | 1/6 | Indices observations à traiter |

## Structures de données partagées

```fortran
! ms1.f → ms2.f → ms3/4.f
sobstot(2000,200,24,10)  ! Cube observations [i,j,λ,nfb] — CLÉ
cal(2000,200,24)         ! Calibration [i,j,n]
cymx(2000,200,24)        ! Canaux bruts
xr(24,3,2), yr(24,3,2)   ! Géométrie coins
profnf(2000,200,250,3)   ! Profils interpolés [i,j,nf,ideg]
                         ! ideg=1:deg3, 2:deg4, 3:mix
```

## Coordonnées (repères)

| Espace | Dimensions | Description |
|--------|------------|-------------|
| CCD brut | 1536 × 1024 | `imb` × `jmb` (ms1 lecture) |
| Permuté ms1 | 1024 × 1536 | `imc` × `jmc` (après permutation) |
| Canaux | 885 × 123 | `imd` × `jmd` × `nm=9` (ms2/ms3/ms4) |
| Filtergrams | <885 × <123 | `ime` × `jme` (produits finaux) |

## Exécution

```bash
# Depuis la racine du projet
cd Avancement_MSDP/scripts
./run_pipeline.sh [ms.par_custom]
```

Le script :
1. Copie les 4 `.f` + `ms.par` dans `/tmp/msdp_pipeline/`
2. Compile avec PGPLOT
3. Symlink les FITS `data/input/` (`*x1.fit`, `*y1.fit`, `*b1.fit`)
4. Exécute `./msdp` (timeout 120s)
5. Convertit `.ps` → `.pdf` versionnés dans `data/output/`
6. Archive logs `ms_run_NNN.lis`, `ACDF2_run_NNN.lis`, `ms_par_run_NNN.par`

## Validation

Le checker `check_fortran_pipeline.py` valide `ACDF2.lis` :
- 9 lignes × 8 valeurs numériques
- Pas de `0.00` ni `********` (overflow)
- Bornes physiques X∈[0,1536], Y∈[0,1536]
- Centres strictement croissants + step régulier (σ<25%)
- Largeurs cohérentes (σ<35%)
- **Nouveau (2026-09-06)** : Alignement bords Y (haut/bas) quasi-parallèles

## Différences majeures vs ancien pipeline (racine `src/fortran/`)

| Aspect | Ancien (racine) | Nouveau (`new/`) |
|--------|-----------------|------------------|
| `ms1.f` | 533 lignes, s'arrête après géométrie | 694 lignes, chaîne complète 1→2→3→4 |
| `ms2.f` | 1078 lignes, géométrie seule | 3248 lignes, +canaux+calib+obs+I/V |
| `ms3.f` | 374 lignes, non appelé | 374 lignes, **appelé** par solarobs |
| `ms4.f` | 0 octet (vide) | 477 lignes, **appelé** par solarobs |
| `sobstot` | Absent | Cube central 2000×200×24×10 |
| `cal` | Absent | Calibration complète |
| I/V maps | Non produites | `ivprof1/2/3.ps` + vitesses |

## Prochaines étapes (non implémentées)

D'après la méthode Mein (7 étapes), manquent encore :
- **Step 5** : d-files → profils de raie, filtergrammes, cartes I/V (partiellement fait via ivmap)
- **Step 6** : q-files → scans cibles complètes
- **Step 7** : Plots finaux

Le pipeline actuel couvre **étapes 1 à 4** de manière fonctionnelle et validée.