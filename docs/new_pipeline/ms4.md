# ms4.f — Interpolation profils I/V degré 4

## Rôle
Sous-programme `ivmap4` appelé par `solarobs` (ms2.f) après `ivmap3`.
Effectue l'interpolation polynomiale degré 4 (5 points) des profils spectraux,
produit deux figures : `ivprof2.ps` (degré 4 pur) et `ivprof3.ps` (mixte degré 3/4).

## Signature
```fortran
subroutine ivmap4(sobstot, cal, iim, jjm, nm, nfb1, nfb2, profnf)
```
| Argument | Type | Description |
|----------|------|-------------|
| `sobstot` | `real*4(2000,200,24,10)` | Cube observations [i,j,λ,nfb] |
| `cal` | `real*4(2000,200,24)` | Calibration |
| `iim`, `jjm` | `integer` | Dimensions canaux (885, 123) |
| `nm` | `integer` | Nombre canaux (9) |
| `nfb1`, `nfb2` | `integer` | Indices obs (forcés à 1) |
| `profnf` | `real*4(2000,200,250,3)` | **In/Out** : profils [i,j,nf,ideg] |

## Algorithme
### 1. Initialisation — copie degré 3 → degré 4
```fortran
do jj = 21, jjm, 40
  do nf = 1, nfm (81)
    profnf(iic, jj, nf, 3) = profnf(iic, jj, nf, 1)  ! copie deg 3
  enddo
enddo
```
Prépare le slot `ideg=3` pour le mix final.

### 2. Interpolation degré 4 par fenêtre glissante (DPMCAR 5 points)
Pour `j = 21, 61, 101` (3 positions verticales) et `n0 = 2,3,4` (canaux centraux) :
```
Fenêtre 5 canaux : n0-1, n0, n0+1, n0+2, n0+3  →  λ indices relatifs 1..5
DPMCAR(xx, yy, P, 5, 5, COEF) → polynôme degré 4 (COEF(1..5))
Évalue 41 points (nff1..nff2) → profnf(iic, j, nff, 2)  ! degré 4 pur
                                         profnf(iic, j, nff, 3)  ! copie pour ivprof3
```
- `nff1 = (n0-1)*10 + 1` (ex: 11, 21, 31)
- `nff2 = nff1 + 40` (ex: 51, 61, 71)
- Grille finale : 81 points λ (1 + 8×10) couverte par 3 fenêtres chevauchantes

### 3. Plot `ivprof2.ps` — Interpolation degré 4
3 images (jj=21,61,101) : profil degré 4 (41 pts) + points originaux (9 canaux).
Vitesse bisector degré 4 : `intvel4` si `lbdvel1/2 ≠ 0`.

### 4. Plot `ivprof3.ps` — Mixte degré 3 + 4
Même 3 positions. Utilise `profnf(...,3)` = degré 3 pour les bords, degré 4 au centre.
Vitesse bisector : `intvel3` (sur profil mixte) si `lbdvel1/2/3 ≠ 0`.
Annotations 'A'/'B' aux extrémités des profils.

## Paramètres ms.par
| Paramètre | Rôle |
|-----------|------|
| `lbdvel1` | Décalage λ vitesse 1 (pour intvel3 dans ivprof3) |
| `lbdvel2` | Décalage λ vitesse 2 |
| `lbdvel3` | Décalage λ vitesse 3 |
| `ivprof2` | 1 = affiche `ivprof2.ps` |
| `ivprof3` | 1 = affiche `ivprof3.ps` |

## Fichiers produits
| Fichier | Description |
|---------|-------------|
| `ivprof2.ps` | 3 pages : profils degré 4 (fenêtres 41 pts) |
| `ivprof3.ps` | 3 pages : profils mixtes deg 3/4 (81 pts) |
| `miv.lis` | (partagé) Log vitesses : `nfb, lbdvel, xv2, yv2` |

## Structures de données
```fortran
profnf(2000,200,250,3)  ! [i,j,nf,ideg]
                         ! ideg=1 : degré 3 (ms3)
                         ! ideg=2 : degré 4 pur (ms4)
                         ! ideg=3 : mixte deg 3/4 (ms4 final)
profic(81,3,3)          ! [nf, nfig, ideg] profils aux 3 positions j
pro1(81), pro2(81), pro3(81)  ! tampons par position
pro5(250), X5(100)      ! pour plot degré 4
velint(200,2000,3,2,10) ! non utilisé (réservé)
```

## Sous-programme interne
- `intvel4(jj, pro, n0, lbdvel, nff1, nff2, nlm)` : bisector degré 4
  - `pro(10,100)` : coefficients/interpolations par fenêtre
  - Opère sur fenêtre `nff1..nff2` (41 pts) pour un `n0` donné
  - Même principe zero-crossing que `intvel3` mais sur profil degré 4

## Différences clés vs ms3.f
| Aspect | ms3.f (deg 3) | ms4.f (deg 4) |
|--------|---------------|---------------|
| Points par fenêtre | 4 (n-1..n+2) | 5 (n0-1..n0+3) |
| Degré polynôme | 3 | 4 |
| Fenêtres | 8 (n=2..nm-2) | 3 (n0=2,3,4 aux 3 j) |
| Points évalués/fenêtre | 10 | 41 |
| Couverture λ | 81 pts continus | 81 pts par 3 fenêtres chevauchantes |
| `profnf` écrit | slot 1 | slots 2 et 3 |
| Vitesse | `intvel3` | `intvel4` (deg 4) + `intvel3` (mixte) |
| Figures | `ivprof1.ps` | `ivprof2.ps` + `ivprof3.ps` |

## Flux complet dans solarobs
```
solarobs (ms2.f)
  ├─ ... construction sobstot ...
  ├─ call ivmap3(sobstot, cal, iim, jjm, nm, 1, 2, profnf)  ! remplit profnf(...,1)
  └─ call ivmap4(sobstot, cal, iim, jjm, nm, 1, 2, profnf)  ! remplit profnf(...,2) et (...,3)
```

## Note importante
- `nfb1=1, nfb2=2` dans l'appel mais **forcés à 1** au début de chaque sous-programme
- Seule la **première observation** (`nf=1`) du cube est traitée
- Position horizontale fixée : `iic = (iim+1)/2` (centre)
- 3 positions verticales : `j = 21, 61, 101` (≈ 1/6, 1/2, 5/6 de jjm=123)