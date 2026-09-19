# ms2.f — Géométrie, calibration, observations solaires

## Rôle
Contient toute la logique de :
1. **Géométrie** : détection bords canaux, calcul coins (`geom` → `SRECT` → `newgeom`)
2. **Canaux** : ré-échantillonnage CCD→canaux (`channels`)
3. **Calibration** : fonction calibration intensités `cal(i,j,n)` (`calib` + sous-routines)
4. **Observations solaires** : lecture `*b1.fit`, soustraction dark, calibration, construction `sobstot` (`solarobs`)
5. **Appel I/V maps** : `ivmap3` (ms3.f) et `ivmap4` (ms4.f)

## Points d'entrée (subroutines publiques)
| Subroutine | Appelée par | Description |
|------------|-------------|-------------|
| `geom` | ms1.f | Entrée géométrie, lit params, appelle SRECT+newgeom |
| `channels` | ms1.f (via calib) | Ré-échantillonne image→canaux, remplit `cymx` |
| `calib` | ms1.f | Calibration intensités, produit `cal(i,j,n)` |
| `solarobs` | ms1.f | Lit obs, soustrait dark, calibre, remplit `sobstot` |
| `mdark` | SRECT, solarobs | Lit dark moyen (unité 31) |
| `calobs` | solarobs | Applique calibration `sobs/cal` |

## Flux géométrie (`geom` → `SRECT` → `newgeom`)
```
geom(nw,win,nm,iux,iuy,iuz,gname,istop,ima,ijcam,imima,jmima,
     xr,yr,imc,jmc,iim,jjm)
  ├─ Lecture paramètres ms.par (igeo, interc, si/sgi/sj/sgj, milangi, jeps, intvi, intvj, leps, n1, distor, normsq, norm, largrid)
  ├─ Lecture flat moyen (iuy=32) → dimensions im,jm
  ├─ SRECT : préparation, soustraction dark (iux=31), détection seuils
  ├─ newgeom : détection bords précis (gradient + interpolation parabolique)
  └─ Retour : xr(24,3,2), yr(24,3,2) = coins A,C,D,F par canal (3 coupes × 2 bords)
```

### Paramètres géométrie (ms.par)
| Param | Défaut | Rôle |
|-------|--------|------|
| `igeo` | 1 | Active géométrie |
| `interc` | 15 | Distance inter-canaux approx (px) |
| `si` | 15 | Seuil intensité bord long (vs X) |
| `sgi` | 8 | Seuil gradient bord long |
| `sj` | 15 | Seuil intensité bord court (vs Y) |
| `sgj` | 5 | Seuil gradient bord court |
| `milangi` | -40 | Angle approx bords longs (rad/1000) |
| `milangj` | ? | Angle approx bords courts |
| `jeps` | 20 | Tolérance position bords |
| `intvi` | 60 | Intégration ± pour bords //i |
| `intvj` | 30 | Intégration ± pour bords //j |
| `leps` | 50 | Fenêtre recherche gradient max |
| `n1` | 1 | Premier canal utile |
| `mingrad` | 18 | Gradient min (lu dans newgeom via par1) |
| `interp` | 1 | Interpolation parabolique (1=oui) |

## Détection bords (`newgeom`)
- 3 coupes horizontales : `ja1=151`, `ja2=501`, `ja3=851` (1-based, param ms.par)
- Pour chaque coupe : scan horizontal, détection pics gradient > `mingrad`
- Interpolation parabolique (`SMAX`) si `interp=1`
- Groupement ordinal : n-ième pic gauche ↔ n-ième pic droit = canal n
- Intersection droites (A/F, C/D) via `intersec` → coins ABCDEF

**Sortie** : `xr(24,3,2)`, `yr(24,3,2)` — 16 points par canal (abcdef + klmn + ABCDEF)

## Canaux (`channels`)
- Calcule coefficients bilinéaires/bicubiques par canal (`COEFF`)
- Interpolation bilinéaire (`PIX`) : CCD (1024×1536) → canaux (885×123)
- Remplit `cymx(iim,jjm,nm)` = intensité par canal
- Appelle `map3` → `flat2.ps`

## Calibration (`calib`)
1. **Lissage** `cliss` = moyenne 2D sur `cymx` (±iliss, ±jliss)
2. **Centres raies** (`plot_line` → `linecurv` → `trans`) : 
   - Détection centres par canal (parabole ±jparab)
   - Régression linéaire par morceaux (`leastsq`) → `center(i,n)`, pente `pte(n)`
   - Translation λ inter-canaux `trj` = pente moyenne
3. **Profil moyen** (`profmean`) : 
   - Empile profils canaux (inversés Ts) → `profmm(km)` continu
   - Coefficients relatifs `coeff(n)` = ratio `profm(jt2,n)/profm(jt1,n-1)`
4. **Calibration intensités** : `cal(i,j,n) = cymx(i,jp,n) / profmm(kp)`
   - `jp = j + pte(n)*(i-ic)` (correction courbure)
   - `kp = 1 + jmd - jp + (n-1)*jtr` (index profil continu)
5. **Plots** : `calib.ps` (profils), `cal.ps` (maps calibration)

## Observations solaires (`solarobs`)
```
do nf = nfb1, nfb2           ! chaque fichier *b1.fit
   lecture FITS (iu=12)
   soustraction dark (mdark → imadark)
   channels(xr,yr,ima,...)    ! → sobs(i,j,n)
   calobs(cal, sobs, ...)     ! calibration → sobs/cal
   stockage : sobstot(i,j,n,nf) = sobs(i,j,n)
   si nf==nfplot : copie sobsplot
enddo
ivmap3(sobstot, cal, iim, jjm, nm, nfb1, nfb2, profnf)
ivmap4(sobstot, cal, iim, jjm, nm, nfb1, nfb2, profnf)
```

## Plots produits
| Fichier | Subroutine | Description |
|---------|------------|-------------|
| `geo1.ps` | `newgeom` | Bords détectés (3 coupes) |
| `geo2.ps` | `plotgeo2` | Coins ABCDEF |
| `geo3.ps` | `plotgeo3` | Distorsions AC/DF/CF |
| `geo4.ps` | `plotgeo4` | Bords interpolés |
| `flat2.ps` | `map3` | Canaux extraits (cymx) |
| `calib.ps` | `calib` | Profils raies + centres |
| `cal.ps` | `calib` | Maps calibration par canal |
| `obs.ps` | `calib` | Observations brutes |
| `obsD1.ps` | `calib` | Non calibrées (×den) |
| `obsD2.ps` | `calib` | Calibrées (/den) |

## Sous-routines utilitaires clés
| Sub | Rôle |
|-----|------|
| `SRECT` | Préparation, seuils, soustraction dark |
| `newgeom` | Détection bords + coins (cœur géométrie) |
| `SMAX` | Interpolation parabolique max gradient |
| `intersec` | Intersection 2 droites (coins) |
| `plotgeo1/2/3/4` | Plots PGPLOT géométrie |
| `channels` | Ré-échantillonnage CCD→canaux |
| `COEFF` | Coeffs transformation par canal |
| `PIX` | Interpolation bilinéaire point par point |
| `calib` | Calibration complète (lissage→centres→profil→cal) |
| `plot_line` | Trace centres raies |
| `linecurv` | Détection centres (parabole + moindres carrés) |
| `transpec` | Translation λ théorique (mupris/mustep) |
| `trans` | Optimisation translation empirique |
| `profmean` | Profil continu empilé + coeffs |
| `plotcal2` | Plot 2D générique (PGGRAY) |
| `map3` | Plot canaux (flat2.ps) |
| `solarobs` | Boucle observations + I/V maps |
| `mdark` | Lecture dark moyen |
| `calobs` | Division par calibration |
| `parafit` | Lissage parabole (utilitaire) |
| `DPMCAR` | Moindres carrés double précision |

## Structures de données principales
```fortran
! Géométrie
xr(24,3,2), yr(24,3,2)  ! 24 canaux max, 3 coupes, 2 bords (gauche/droite)
                         ! indices 1-6: abcdef, 7-10: klmn, 11-16: ABCDEF

! Canaux
cymx(2000,200,24)       ! Canaux bruts [i,j,n]
cal(2000,200,24)        ! Calibration [i,j,n]
profmm(2000)            ! Profil continu empilé
center(2000,24)         ! Centres raies lissés par canal
pte(24)                 ! Pentes centre/i par canal

! Observations
sobstot(2000,200,24,10) ! Cube complet [i,j,λ,nfb] (10 obs max)
sobs(2000,200,24)       ! Observation courante calibrée
sobsplot(2000,200,24)   ! Copie pour plot (nfplot)
```

## Différences majeures avec l'ancien ms2.f (racine)
- **3248 lignes** vs 1078 (×3) : ajout `channels`, `calib`, `solarobs`, plots
- `geom` signature étendue : ajoute `iim,jjm` en sortie (dims canaux)
- `newgeom` appelé **après** `SRECT` (pas avant)
- `calib` complète avec lissage, centres raies, profil continu, calibration
- `solarobs` : lecture obs, construction `sobstot`, appelle `ivmap3/4`
- Tous les plots PGPLOT activables via ms.par (`igeo1-4`, `icalib`, `icalplot`, `iobs`, `iobsD1/2`)
- Supprime `geom` dummy calls, code réel uniquement