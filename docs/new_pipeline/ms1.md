# ms1.f — Programme principal du pipeline MSDP (Fortran 77)

## Rôle
Point d'entrée du pipeline MSDP. Orchestre les 4 étapes majeures :
1. **Moyennage dark/flat** (nxy=1,2)
2. **Géométrie** (appel `geom` dans ms2.f)
3. **Calibration** (appel `channels` → `calib` dans ms2.f)
4. **Observations solaires + I/V maps** (appel `solarobs` → `ivmap3/ivmap4` dans ms3.f/ms4.f)

## Flux principal
```
do nxy=1,2                ! dark (x1.fit) puis flat (y1.fit)
   lecture fichiers FITS
   moyenne des séquences
   écriture fichiers binaires moyens (unité 31/32)
enddo

call geom(...)            ! étape 2 : géométrie (ms2.f)
call channels(...)        ! extraction canaux (ms2.f)
call calib(...)           ! calibration intensités (ms2.f)

call solarobs(...)        ! lecture obs (b1.fit), soustraction dark,
                          ! extraction canaux, calibration,
                          ! stockage cube sobstot(2000,200,24,10)
call ivmap3(...)          ! interpolation degré 3 (ms3.f)
call ivmap4(...)          ! interpolation degré 4 (ms4.f)
```

## Paramètres ms.par lus
| Paramètre | Description |
|-----------|-------------|
| `nfx1`, `nfx2` | Premier/dernier dark (`*x1.fit`) |
| `nfy1`, `nfy2` | Premier/dernier flat (`*y1.fit`) |
| `nfb1`, `nfb2` | Premier/dernier observation (`*b1.fit`) |
| `is`, `js` | Dimensions CCD (1536, 1024) |
| `icalib` | Active la calibration (1=oui) |
| `ivprof1`, `ivprof2`, `ivprof3` | Génère les plots I/V maps |

## Fichiers produits
| Fichier | Unité | Description |
|---------|-------|-------------|
| `x<date>_<time>_00000` | 31 | Dark moyen |
| `y<date>_<time>_00000` | 32 | Flat moyen |
| `channel.lis` | 95 | Log canaux |
| `ms.lis` | 3 | Log principal détaillé |
| `miv.lis` | 4 | Log I/V maps (intvel) |

## Sous-programmes internes
- `readpar` : lecture complète `ms.par` (format a8,i8)
- `par1(name,nw,nombre)` : lecture paramètre unique par nom
- `readfits` : lecture FITS avec permutation + byte-swap
- `counthead` / `openold38` / `opennew22` / `openold22` : utilitaires FITS
- `DPMCAR` : moindres carrés double précision (polynôme ≤10 termes)

## Variables globales clés
```fortran
dimension sobstot(2000,200,24,10)  ! Cube observations (i,j,λ,nfb)
dimension cal(2000,200,24)         ! Fonction calibration
dimension cymx(2000,200,24)        ! Canaux extraits (bruts)
dimension ima(1536,1536)           ! Image courante
dimension xr(24,3,2), yr(24,3,2)   ! Coins géométrie (xr/yr par canal)
```

## Coordonnées (commentaires entête)
```
CCD (brut)        : imb=1536, jmb=1024
Permuté (ms1)     : imc=1024, jmc=1536
Canaux (ms2/ms3)  : imd=885,  jmd=123,  nm=9
Filtergrams/Doppler : ime<885,  jme<123,  nm=9
```

## Différences avec l'ancien ms1.f (racine)
- Déclare `sobstot` et `cal` (absents avant)
- Boucle `nxy=1,2` au lieu de `1,4` (ne lit pas les obs ici)
- Appelle `channels` + `calib` + `solarobs` + `ivmap3/4`
- Ajoute `miv.lis` (unit 4) pour les maps I/V
- Commentaires restructurés avec diagramme de flux ASCII