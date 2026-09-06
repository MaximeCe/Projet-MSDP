# Téléchargement des données DPSM/MSDP depuis BASS2000

Ce script Python permet de **télécharger automatiquement les données DPSM
(MSDP) depuis l'archive BASS2000** de l'Observatoire de Paris (Meudon).

> **Origine** : copié depuis `DPSM-GUI/scripts/download_bass2000.py` (avec son
> module `services/bass2000.py`) pour être utilisable dans Projet-MSDP.
>
> | Source (DPSM-GUI) | Copie Projet-MSDP |
> |---|---|
> | `scripts/download_bass2000.py` | `src/python/bass2000/download_bass2000.py` |
> | `services/bass2000.py` | `src/python/bass2000/services/bass2000.py` |

---

## 1. Prérequis

Le script dépend du module `services/bass2000.py` situé **dans son
sous-dossier** `services/`. Il s'en sert via
`sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` puis
`from services.bass2000 import ...`, donc le fonctionnement est identique
au dossier d'origine.

Package Python requis :

```bash
pip install requests
```

## 2. Utilisation

Depuis le dossier du script :

```bash
cd src/python/bass2000

# Lister les séquences d'une date
python3 download_bass2000.py --date 2015-06-04

# Lister les années disponibles
python3 download_bass2000.py --years

# Lister les jours d'une année
python3 download_bass2000.py --days --date 2015-06-04

# Télécharger une séquence précise
python3 download_bass2000.py --date 2015-06-04 --seq 3

# Télécharger par type
python3 download_bass2000.py --date 2015-06-04 --type flat
python3 download_bass2000.py --date 2015-06-04 --type dark
python3 download_bass2000.py --date 2015-06-04 --type observation

# Tout télécharger
python3 download_bass2000.py --date 2015-06-04 --all

# Options supplémentaires
python3 download_bass2000.py --date 2015-06-04 --seq 3 --dest /chemin/data --limit 50
```

### Options

| Option | Description |
|--------|-------------|
| `--date YYYY-MM-DD` | Date de l'observation (obligatoire pour télécharger) |
| `--hour N` | Heure de début (défaut 0) |
| `--seq N` | Numéro de séquence à télécharger |
| `--type {flat,dark,observation,all}` | Type de séquences |
| `--dest DIR` | Dossier destination (défaut `data/`) |
| `--limit N` | Limite le nombre de fichiers par séquence |
| `--list-only` | Affiche sans télécharger |
| `--years` | Liste les années avec données |
| `--days` | Liste les jours de l'année (avec `--date`) |

### Conventions des fichiers téléchargés

Chaque fichier est nommé `{date}_{heure}_{contenu}.fit`, par exemple :
```
2015-06-04_10200351_flat.fit
```

---

## 3. Fonctionnement (module `services/bass2000.py`)

Le module dialogue avec les endpoints BASS2000 :

| Endpoint | Rôle |
|----------|------|
| `longterm_archive.php` | Liste des séquences d'une journée (scraping HTML `<TR>`) |
| `longterm/getJsonSequenceObs.php` | Fichiers FITS d'une séquence (API JSON) |
| `longterm/get_fileinfo.php` | URL directe de téléchargement d'un fichier |
| `longterm/get_dateobs_data.php` | Jours d'observation (pour `--years`/`--days`) |

### Fonctions principales

| Fonction | Description |
|----------|-------------|
| `get_observation_days(year)` | Liste des `YYYY-MM-DD` avec données pour une année (mise en cache) |
| `get_observation_months(year)` | Mois avec données d'une année |
| `get_days_in_month(year, month)` | Jours avec données d'un mois |
| `get_sequences(date, hour)` | Séquences d'une date : `{num_seq, type, start, end, nfiles}` |
| `get_sequence_files(date, num_seq)` | Fichiers FITS d'une séquence : `{time, content, file_id, url_path, filename}` |
| `get_download_url(file_id)` | URL de téléchargement directe depuis l'API |
| `find_best_calibration(date, num_seq, sequences)` | Meilleure calibration **flat + dark** temporellement la plus proche |
| `download_sequence(date, num_seq, dest, ...)` | Télécharge une séquence en **parallèle** (4 workers) |
| `download_with_calibration(...)` | Télécharge l'obs + ses flat/dark associés |
| `clear_cache()` | Vide le cache des jours/mois |

### Types de séquences

| Clé CLI | Label BASS2000 |
|---------|----------------|
| `observation` | `Observation` |
| `flat` | `Flat Field` |
| `dark` | `Dark Current` |

### Détails d'implémentation
- **Session HTTP** partagée avec User-Agent `DPSM-GUI/1.0 (MSDP pipeline)` et
  timeout de 60 s.
- **Mise en cache** des jours/mois d'observation (variables `_OBS_DAYS_CACHE`,
  `_OBS_MONTHS_CACHE`).
- **Téléchargement parallèle** via `ThreadPoolExecutor(max_workers=4)`, avec
  `iter_content(8192)` et nettoyage des fichiers partiels en cas d'erreur.
- Fichiers déjà présents → passés (pas de re-téléchargement).

---

## 4. Intégration avec le pipeline Projet-MSDP

Les fichiers `.fit` téléchargés sont destinés à `data/input/` et ensuite
traités par :
- **`run_pipeline.sh`** (Fortran : moyennes dark/flat + géométrie)
- **`run_pipeline_py.sh`** (Python : même périmètre via ms1.py/ms2.py)
- **`run_both_pipeline.sh`** (les deux, avec sorties versionnées)

Exemple de commande complète pour une observation du 2015-06-04 (séquence 3)
avec calibration :

```bash
mkdir -p data/input
cd src/python/bass2000
python3 download_bass2000.py --date 2015-06-04 --type all --dest ../../data/input
```

> ⚠️ **Note sur la calibration** : la sélection de la meilleure calibration
> (flat + dark) est faite par `find_best_calibration` en choisissant la
> séquence du même type **temporellement la plus proche** avant/après
> l'observation. Ça correspond au comportement attendu, mais valider toujours
> les dates/heures dans la sortie du script.