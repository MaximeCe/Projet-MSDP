# `ingest_all.py` — Orchestrateur d'ingestion MSDP (Volet B)

> Chemin : `Projet-MSDP/ingest_all.py`
> Rôle : automatiser l'ingestion de jeux de données MSDP, de bout en bout :
> **téléchargement → estimation auto-params → run des 2 pipelines → validation →
> CSV → cleanup**, avec une **boucle de correction** pour roder les paramètres
> sensibles (ja, mingrad).

---

## 1. Boucle pour chaque date

```
1. (option) Télécharger les données depuis BASS2000  (download_bass2000.py)
2. Estimer les paramètres depuis le flat             (auto_params.estimate_params)
3. Boucle de correction : tester plusieurs jeux (ja, mingrad)
   ├─ run_pipeline.sh + run_pipeline_py.sh avec la config candidate
   ├─ check_fortran_pipeline.py / check_python_pipeline.py
   └─ garder la config qui passe les 2 checks (ou la meilleure)
4. Écrire une ligne dans le CSV
5. (option, sauf --keep) supprimer les données de la date
```

---

## 2. Usage

```bash
# Traiter les données déjà dans data/input/ (sans téléchargement)
python3 ingest_all.py --local

# Enchaîner sur toutes les dates BASS2000
python3 ingest_all.py --dates 2015-06-04 2015-06-05

# Options
python3 ingest_all.py --local --keep          # ne pas supprimer les données
python3 ingest_all.py --local --dry-run       # ne rien exécuter (affiche le plan)
python3 ingest_all.py --local --csv /tmp/x.csv  # CSV custom
```

| Option | Rôle |
|--------|------|
| `--local` | supervise les données déjà présentes dans `data/input/` (pas de téléchargement) |
| `--dates D1 D2 ...` | dates à traiter (sans `--local`, tente le téléchargement BASS2000) |
| `--csv PATH` | chemin du CSV (défaut `data/output/ingest.csv`) |
| `--keep` | ne pas supprimer les `.fit` après ingestion |
| `--dry-run` | ne lancer aucun pipeline, affiche simplement le plan |

---

## 3. Structure du CSV

| Colonne | Description |
|---------|-------------|
| `date` | date (ou `local`) |
| `instr` | instrument détecté (`meudon` si im=1536, sinon `other`) |
| `im, jm` | dimensions image (du header flat) |
| `nm` | nombre de canaux estimé |
| `ja1, ja2, ja3` | coupes de détection (après correction) |
| `mingrad` | seuil de gradient |
| `interc` | espacement inter-canaux estimé |
| `check_fortran, check_python` | `ok` si le check du pipeline passe |
| `ecart_max, ecart_moy` | **écart max / moyen (px)** entre les points de l'ACDF2 Fortran et Python du run retenu (même canal, même champ) |
| `n_tries` | nombre de tentatives de la boucle de correction |
| `statut` | `ok` (géométrie valide) ou `best-effort` (meilleure config échoue quand même) |

### Exemple réel (test local)
```
local,meudon,1536,1024,9,213,512,810,18,139.625,ok,ok,1.310,0.187,5,ok
```
`ecart_max=1.310` / `ecart_moy=0.187` : les coins A,C,D,F (X et Y) diffèrent d'au
plus 1.31 px entre les deux pipelines — cohérent avec le décalage théorique du
bug `1.-ac` vs `1-a*c` (~0.1-1.4 px).

---

## 4. Boucle de correction (le cœur)

L'estimation initiale de `ja` par `auto_params` (équirépartition) ne garantit pas
une géométrie valide (constat A-auto). `ingest_all.py` implémente donc la boucle
prévue par la spec :

- **candidats testés** : ja initial ± décalages (3 % de la hauteur), puis `mingrad`
  alternatif (mingrad−6) ;
- à chaque candidat, on lance réellement les 2 pipelines et on exécute les checks ;
- on garde le premier candidat qui passe les **2 checks** (`score==2`) ;
- sinon, le meilleur score → `statut=best-effort`.

Résultat observé sur le flat Meudon : l'estimation initiale `ja=[153,512,870]`
échouait ; la correction a trouvé `ja=[213,512,810]` qui passe les 2 checks.

---

## 5. Limites & notes

- **Coût** : la boucle de correction lance plusieurs fois les 2 pipelines (chaque
  itération ~une dizaine de secondes). Pour N dates, c'est N × tries.
- **`--local`** est le mode de test (un flat présent dans `data/input/`). Le mode
  BASS2000 (`--dates`) nécessite la connexion réseau et a `timeout=600`.
- Le cleanup supprime les `.fit` de la date **seulement** en mode non-`--local`
  (il ne touche pas à `data/input/` en mode local, pour ne pas détruire le jeu de
  test).
- `download_bass2000.py` et `auto_params.py` sont des dépendances directes
  (dans `src/python/` ou `src/python/bass2000/`).

---

## 6. Intégration

L'orchestrateur s'appuie sur tous les composants construits dans les volets
précédents :

| Composant | Rôle |
|---|---|
| `auto_params.py` | estimation nm/im/jm/interc/ja |
| `run_pipeline.sh` | pipeline Fortran |
| `run_pipeline_py.sh` | pipeline Python |
| `check_fortran_pipeline.py` / `check_python_pipeline.py` | validation |
| `download_bass2000.py` | récupération des données (option) |