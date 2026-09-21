# PLAN — Refactorisation & traduction Fortran 77 → Python 3.11 du pipeline MSDP

> **Goal :** Réécrire intégralement le pipeline MSDP (4 étapes, 4 fichiers Fortran :
> ms1.f → ms2.f → ms3.f → ms4.f) en Python 3.11+ moderne, vectorisé (NumPy/SciPy),
> typé, testé (pytest), dans `/new_python`, avec un module `plotting.py` Matplotlib
> remplaçant PGPLOT — en validant la parité numérique run-par-run contre le Fortran.

---

## 1. Contexte & état des lieux

**Entrée de référence (source de vérité) :** `src/fortran/new/` — 4793 lignes F77, 4 fichiers :
`ms1.f` (694), `ms2.f` (3248), `ms3.f` (374), `ms4.f` (477).

**Port Python existant (à réutiliser comme base, PAS à réécrire de zéro) :** `src/python/`
— `ms1.py` (442 l, classe `MSDPProcessor`) et `ms2.py` (763 l, classe `GeometryProcessor`)
couvrent les **étapes 1-2** et sont **validés contre le Fortran** (9/9 canaux, Y ~0.01–0.05 px,
X ~0.1–0.7 px). Les étapes **3-4 sont entièrement à écrire** (≈2800 lignes F77 : canaux,
COEFF/pix, calib, profmean/transpec/trans/parafit, solarobs, mdark, calobs, ivmap3/4, intvel3/4).

**Traits du code F77 (cartographie chiffrée) :**
- ~34 sous-routines / fonctions
- **248 étiquettes de GOTO** (47 ms1, 144 ms2, 25 ms3, 32 ms4), 0 GOTO calculé
- **Aucun bloc COMMON** (bonne nouvelle — rien à déplier)
- ~200 appels PGPLOT (`pgbegin` 16, `pgline` 53, `pgtext` 44, `pgpoint`/`pgpt` 17, `pgbox` 27…)
- Boucles `DO … CONTINUE` avec étiquettes numériques partout
- Données : `integer*2`/`integer*4`, `real*4`, tableaux `tab2(1536,1536)`, `ima(1536,1536)`,
  cube `sobstot(2000,200,24,10)`, `cal(2000,200,24)`, `cliss(2000,200,24)`

**Chaîne d'appel principale (ms1.f main) :**
```
dark/flat averaging (Loop nxy=1:2, do 100 nf) → write averaged binaries
→ geom() [ms2] → icalib? → channels() → calib() → [fin main ms1.f]
  geom → SRECT (flat−dark, permute ima→meanflat) → newgeom → plotgeo1/2/3/4 → ACDF2.lis
  channels → map3 (flat2.ps) ; calib (cliss, cal) ; solarobs (obs*, sobstot)
  ↓ (appelés depuis la fin d' ivmap de ms2.f)
  ivmap3 [ms3] → profnf(...,ideg=1) ; intvel3 → miv.lis ; ivprof1.ps
  ivmap4 [ms4] → ivprof2.ps / ivprof3.ps ; intvel3/4 → miv.lis
```

**Environnement :** Python 3.11 dispo via `/home/max/nextcloud/Workspace/.venv` (vide), python global
3.14 avec numpy 2.5.2 / scipy 1.18.1 / matplotlib 3.11.1 / astropy 8.0.1 ; pytest et numba absents.

---

## 2. Découpage en sous-tâches séquentielles (plan d'exécution)

### Phase 0 — Environnement & fondations
1. Créer `new_python/pyproject.toml` + `requirements.txt` (numpy, scipy, matplotlib, astropy,
   pytest ; numba optionnel). Installer dans un venv dédié `new_python/.venv`.
2. Port config : `msdp/config.py` — dataclass `Config` + chargeur `ms.par` (format a8,i8)
   → équivalent structuré de `readpar`/`par1`. Documenter les 3 gotchas (droite-justification,
   octale run-number, label `end`).

### Phase 1 — Étape 1 (réutiliser ms1.py)
3. `msdp/step1_average.py` — portage/refactor de `MSDPProcessor` :
   lecture FITS (astropy, **sans** byteswap dupliqué — Bug 1), offset `integer*2`→int, moyenne
   dark/flat, permutation, écriture du binaire moyen. Isoler I/O (`io_fits.py`).
4. Test de parité : moyenne binaire identique au Fortran (voir données `backup_meudon_input`).

### Phase 2 — Étape 2 (réutiliser ms2.py)
5. `msdp/step2_geometry.py` — portage/refactor de `GeometryProcessor` :
   `SRECT` (flat−dark, `idc`), `newgeom` (local zmax/zgmax — Bug 2, k/l/m/n off-by-one — Bug 3,
   widening `(20,40)`), `smax` (interp parabolique), `intersec` (dénominateur `1-a*c`), écriture
   `ACDF2.lis`.
6. Test de parité : `ACDF2` vs `data/output/ACDF2_run_007.lis` (9/9 canaux, tolérances Y/X).

### Phase 3 — Étape 3 : canaux + calibration (à écrire, ms2.f)
7. `msdp/step3_channels.py` — `channels`, `COEFF` (coef distorsion), `pix`, `map3` (flat2.ps).
8. `msdp/step3_calib.py` — `calib`, `plot_line`, `linecurv`, `profmean`, `transpec`, `trans`,
   `parafit`, `plotcal2` → `calib.ps`, `cal.ps`.
9. Tests : centres de raie + maps de calibration cohérentes vs Fortran (`cal.ps`/valeurs logs).

### Phase 4 — Étape 4 : observations + I/V + vitesses (à écrire, ms2/ms3/ms4)
10. `msdp/step4_solarobs.py` — `solarobs`, `mdark`, `calobs` → `obs.ps`, `obsD1.ps`, `obsD2.ps`.
11. `msdp/step4_ivmaps.py` — `ivmap3`+`intvel3` (ms3), `ivmap4`+`intvel4` (ms4) → `ivprof1/2/3.ps`,
    `miv.lis` (vitesses, lbdvel 5/8/1).
12. Tests : `miv.lis` (xv2/yv2) vs Fortran ; note du pitfall segfault `intvel3/4` résolu.

### Phase 5 — Graphique & orchestration
13. `msdp/plotting.py` — équivalents Matplotlib des figures PGPLOT (isolation visu/calcul) :
    geo1-4, calib, cal, obs/obsD1/obsD2, ivprof1-3.
14. `msdp/pipeline.py` — orchestrateur `run_pipeline(config) → PipelineResult` reproduisant
    le flux ms1→ms2→ms3→ms4 et écrivant le même jeu de sorties (géométrie, ACDF2, PDFs, miv).

### Phase 6 — Tests & validation finale
15. `tests/test_pipeline.py` — `pytest` : parité numérique vs références Fortran
    (ACDF2_run_00X.lis, miv, moyenne binaire), unités (smax, intersec, config, COEFF/pix).
16. Exécution end-to-end sur `backup_meudon_input` (canonical 2017) + comparaison des PDFs.

---

## 3. Stratégies de traduction (consignes respectées)

| Problème F77 | Solution Python |
|---|---|
| GOTO / DO étiquetés | Boucles `for`/`while` pythoniques ; `break`/`continue` explicites ; fonctions extraites pour les blocs multiples |
| COMMON blocks | **Aucun** dans ce code — remplacés par dataclasses/module `config` |
| PGPLOT | `matplotlib` (Agg, non-interactif) — toute logique de visualisation isolée dans `plotting.py` |
| Indexation 1-based | Conversion 0-based systématique — **toujours** vérifier écriture vs lecture (leçon Bug 3) |
| Column-major | NumPy row-major ; vérifier chaque opération matricielle (leçon Bug 1/2) |
| Boucles non vectorisables | `@njit` Numba (optionnel, gardé derrière un flag) |
| Monolithe | Modules par étape + classes ; types `typing` + docstrings PEP 257 |

---

## 4. Fichiers attendus (arborescence cible)

```
new_python/
├── pyproject.toml      requirements.txt
├── msdp/
│   ├── __init__.py
│   ├── config.py           # Config dataclass + parseur ms.par (08-i8)
│   ├── io_fits.py          # lecture FITS, swap, offset, fichiers moyennés
│   ├── step1_average.py    # moyennes dark/flat  (réutilise ms1.py)
│   ├── step2_geometry.py   # SRECT/newgeom/smax/intersec (réutilise ms2.py)
│   ├── step3_channels.py   # channels/COEFF/pix/map3
│   ├── step3_calib.py      # calib/linecurv/profmean/transpec/trans/parafit
│   ├── step4_solarobs.py   # solarobs/mdark/calobs
│   ├── step4_ivmaps.py     # ivmap3+intvel3 / ivmap4+intvel4
│   ├── plotting.py         # Matplotlib (geo1-4, calib, cal, obs*, ivprof*)
│   └── pipeline.py         # orchestrateur de bout en bout
└── tests/
    ├── test_pipeline.py    # parité vs Fortran + unitaires
    └── data/               # petites fixtures + pointeurs vers backup_meudon_input
```

---

## 5. Validation (pytest, parité contre le Fortran)

- `tests/test_geometry.py` — ACDF2 vs `ACDF2_run_007.lis` (centres, coins, alignement Y).
- `tests/test_average.py` — binaire moyen + flat−dark vs `ms1.f`.
- `tests/test_calib.py` / `test_ivmaps.py` — `cal*`, `miv.lis` (xv2/yv2) vs logs Fortran
  `ms_run_00X.lis` (valeurs numériques imprimées).
- Bout en bout : `python -m msdp.pipeline ms.par` → 16 sorties == run Fortran.

---

## 6. Risques & décisions ouvertes

- **Étapes 3-4 jamais portées** : parité à établir la première fois (pas de référence Python).
  La référence numérique sera les valeurs imprimées dans `ms_run_*.lis` + les `.ps`→PDF.
- **Numba** : utile sur `newgeom`/`calib`/`profmean` ; en cas de perf correcte, rester pure NumPy.
- **Données de test** : `backup_meudon_input/` (canonical 2017) — la seule série qui tourne sans
  segfault (le dataset BASS2000 2015 segfaultait `newgeom`).
- **Environnement** : créer un venv dédié dans `new_python/` pour ne pas dépendre du python global.