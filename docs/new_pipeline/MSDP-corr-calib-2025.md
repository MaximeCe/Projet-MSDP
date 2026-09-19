# MSDP — Corrections après calibration (Guide logiciel 2025-10)

**Auteur** : Pierre Mein  
**Document source** : `docs/sources/MSDP-corr-calib-2025.pdf`  
**Date** : Octobre 2025

---

## Résumé

Ce document présente des observations complémentaires sur la **fonction de calibration** MSDP (`cal(X,Y,n)`), permettant de détecter et corriger des défauts optiques du spectrographe après calibration des canaux.

---

## Fonction de calibration

La fonction de calibration est définie comme :

> **cal(X,Y,n) = Intensité champ plat / Intensité profil moyen à même λ**  
> (en tenant compte de la fonction d'onde le long de l'axe Y)

Les **intensités calibrées** s'obtiennent par division :
> **I_cal(X,Y,n) = I_obs(X,Y,n) / cal(X,Y,n)**

Si les fichiers de *flat field* donnent d'excellents résultats statistiques, les **petits défauts d'intensité** dus aux réglages du spectrographe sont corrigés.

---

## Corrections optiques via la fonction de calibration

Les Figures 1 à 3 montrent la fonction de calibration et les résultats pour les **9 canaux X×Y** avec 25 fichiers *flat field* (bonnes statistiques).

### Figure 1 — `cal.ps` : Fonction de calibration
- Issue des *flat fields* 09:59:39 → 10:01:35
- Échelle : noir = 0.8, blanc = 1.2
- Perturbations visibles :
  - **Poussières noires** : canal 1 (y=270), canal 7 (y=800), canal 8 (y=600)
  - **Pupilles réduites** progressivement en X et Y (canaux 6, 7, 9, y<150)
  - **Obscurité constante** sur un canal entier (canal 1, Y>810)

### Figure 2 — `obsD1.ps` : Observation brute (sans calibration)
- Observation cible à 10:20:33 (après flat field)
- Division par intensité moyenne le long de chaque X (≈ λ constant)
- Même échelle : 0.8 – 1.2
- Mêmes perturbations que Fig.1 + structures solaires

### Figure 3 — `obsD2.ps` : Observation **avec calibration**
- Les perturbations des Fig.1 et 2 **ont presque disparu**
- Preuve de l'efficacité de la calibration
- Structures solaires fortes peuvent masquer de petites fluctuations de calibration
- Corrections vérifiées :
  - Poussières disparues (canal 1 y=270, canal 7 y=800, canal 8 y=600) ; canal 7 y=700 : structures solaires remplacent la partie droite de la poussière (probablement poussière sur lentille de champ à la sortie)
  - Pupilles réduites (canaux 6, 7, 9, y<150) → maintenant constantes en X et Y
  - Obscurité constante canal 1 (Y>810) disparue → peut correspondre à problèmes près des fentes du *slicer*

### Figure 4 — `obs.ps` : Intensité observée et calibrée **sans division par moyenne/X**
- Même que Fig.3 sans normalisation par X
- Noir ≈ centre de raie, blanc ≈ continuum
- Perturbations corrigées mais dynamique trop grande (noir-blanc) pour détection utile

---

## Interprétation des défauts détectés

| Défaut | Localisation (Fig.1/2) | Correction (Fig.3) | Origine probable |
|--------|------------------------|-------------------|------------------|
| Poussières sombres | Canaux 1, 7, 8 | Disparues | Poussière sur lentille de champ (sortie) |
| Pupilles réduites X/Y | Canaux 6, 7, 9 (y<150) | Constantes | Vignettage optique |
| Canal entier sombre | Canal 1 (Y>810) | Disparu | Problème près fentes *slicer* |

---

## Fichiers MSDP concernés

| Fichier | Paramètre ms.par | Description |
|---------|------------------|-------------|
| `cal.ps` | `ical=1` | Fonction de calibration (Fig.1) |
| `obsD1.ps` | `iobsD1=1` | Obs. non calibrées / moyenne X (Fig.2) |
| `obsD2.ps` | `iobsD2=1` | Obs. calibrées / moyenne X (Fig.3) |
| `obs.ps` | `iobs=1` | Obs. calibrées brutes (Fig.4) |

La calibration est activée par `iflat=1` et `icalib=1` dans `ms.par`.

---

## Pipeline Fortran associé

Dans le nouveau pipeline (`src/fortran/new/`) :

- **ms1.f** → appelle `calib()` si `icalib=1`
- **ms2.f** → `calib()` : construit `cal(i,j,n)` via lissage → centres raies → profil moyen → division
- **ms2.f** → `solarobs()` → produit `obsD1.ps` (non calibré) et `obsD2.ps` (calibré) via `plotcal2()`
- **ms2.f** → `calobs()` applique `sobs/cal` pour les observations

---

## Usage pratique

1. Acquérir **≥20 flat fields** pour bonnes statistiques (25 utilisés ici)
2. Activer calibration : `iflat=1`, `icalib=1`, `ical=1`, `iobsD2=1`
3. Lancer pipeline → génère `cal.ps`, `obsD1.ps`, `obsD2.ps`, `obs.ps`
4. Analyser `cal.ps` pour identifier défauts optiques (poussières, vignettage, *slicer*)
5. Vérifier correction sur `obsD2.ps` : défauts doivent disparaître
6. Utiliser positions (canal, X, Y) pour localiser physiquement les défauts dans le spectrographe

---

## Notes

- La calibration corrige les **défauts multiplicatifs** (transmission, vignettage, poussières)
- Ne corrige **pas** les défauts additifs (courant obscurité → géré par soustraction dark)
- Sensibilité limitée par le rapport signal/bruit des flat fields
- Structures solaires fortes peuvent masquer résidus de calibration
- La division par moyenne/X (`obsD1/obsD2`) permet visualisation simultanée toutes λ

---

## Références croisées

- `docs/new_pipeline/ms2.md` — Implémentation `calib()`, `solarobs()`, `plotcal2()`
- `docs/sources/MSDP-geometry-2024-12.pdf` — Géométrie canaux
- `docs/sources/MSDP-methods-2024-02.pdf` — Méthode complète 7 étapes (Steps 3-4 = calibration)