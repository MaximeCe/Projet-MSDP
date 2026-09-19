# ms3.f — Interpolation profils I/V degré 3

## Rôle
Sous-programme `ivmap3` appelé par `solarobs` (ms2.f) après construction du cube `sobstot`.
Effectue l'interpolation polynomiale degré 3 des profils spectraux par canal,
puis calcule les cartes Intensité/Vitesse (bisector method) et produit `ivprof1.ps`.

## Signature
```fortran
subroutine ivmap3(sobstot, cal, iim, jjm, nm, nfb1, nfb2, profnf)
```
| Argument | Type | Description |
|----------|------|-------------|
| `sobstot` | `real*4(2000,200,24,10)` | Cube observations [i,j,λ,nfb] |
| `cal` | `real*4(2000,200,24)` | Calibration (non utilisée ici) |
| `iim`, `jjm` | `integer` | Dimensions canaux (885, 123) |
| `nm` | `integer` | Nombre canaux (9) |
| `nfb1`, `nfb2` | `integer` | Indices obs à traiter (forcés à 1) |
| `profnf` | `real*4(2000,200,250,3)` | **Sortie** : profils interpolés [i,j,nf,1] |

## Algorithme
### 1. Interpolation degré 3 par canal (DPMCAR)
Pour chaque position spatiale `(i=iic, j=1:jjm par 10)` et chaque triplet de canaux centrés :
```
n = 2 à nm-2 (canaux 2..7 pour nm=9)
  4 points : canaux n-1, n, n+1, n+2  →  λ = 0, 10, 20, 30
  DPMCAR(x,y,P,4,4,COEF) → polynôme degré 3
  Évalue 10 points entre n et n+1 → indices nf = 10*(n-1) .. 10*(n-1)+10
```
Résultat : **81 points λ** par profil (1 + 8×10) stockés dans `profnf(iic, j, nf, 1)`.

### 2. Plots PGPLOT → `ivprof1.ps`
3 images (pages PGPLOT) :
| Image | Position j | Titre |
|-------|------------|-------|
| 1 | jj=21 | Profile in X=20 |
| 2 | jj=61 | Profile in X=60 |
| 3 | jj=101 | Profile in X=100 |

Chaque image : profil continu 81 points + points originaux (tous les 10 pts).

### 3. Vitesse (bisector) — `intvel3`
Si `lbdvel1/2/3 ≠ 0` (param ms.par) :
- Calcule différence `pro(nl+lbdvel) - pro(nl)` (décalage λ cubique)
- Trouve passage par zéro → position `xv2`, intensité `yv2`
- Écrit dans `miv.lis` (unité 4) : `nfb, lbdvel, xv2, yv2, xv2 km/s`
- Trace lignes horizontale + verticale sur le plot

## Paramètres ms.par
| Paramètre | Rôle |
|-----------|------|
| `lbdvel1` | Décalage λ pour vitesse 1 (0=inactif) |
| `lbdvel2` | Décalage λ pour vitesse 2 |
| `lbdvel3` | Décalage λ pour vitesse 3 |
| `ivprof1` | 1 = affiche `ivprof1.ps` via okular |

## Fichiers produits
| Fichier | Unité | Description |
|---------|-------|-------------|
| `ivprof1.ps` | PGPLOT | 3 pages : profils interpolés X=20,60,100 |
| `miv.lis` | 4 | Log vitesses (format 2i8,3f8.2) |

## Structures de données
```fortran
profnf(2000,200,250,3)  ! [i,j,nf,ideg] ideg=1 pour degré 3
                        ! nf=1..81 (1 + 8×10 points interpolés)
prof(250)               ! Tampon profil 81 pts
COEF(4)                 ! Coeffs polynôme degré 3
xx(250), yy(250)        ! Pour plots
```

## Sous-programme interne
- `intvel3(yy, lbdvel, nlm, xv2, yv2)` : méthode bisector
  - `yy(250)` : profil intensité
  - `lbdvel` : décalage en indices λ cubiques
  - Trouve `nla1` où `diff(nla1)*diff(nla1+lbdvel) ≤ 0`
  - Interpolation linéaire → `xv2` (position), `yv2` (intensité)

## Différences avec ms4.f
| Aspect | ms3.f (degré 3) | ms4.f (degré 4) |
|--------|-----------------|-----------------|
| Polynôme | 4 points → degré 3 | 5 points → degré 4 |
| Points λ | 81 (1+8×10) | 81 (même grille) |
| Vitesse | `intvel3` (bisector simple) | `intvel4` (bisector degré 4) |
| Plot | `ivprof1.ps` | `ivprof2.ps` + `ivprof3.ps` |
| `profnf` output | `profnf(...,1)` | `profnf(...,2)` et `profnf(...,3)` |

## Note d'usage
- Appelé **une seule fois** par `solarobs` avec `nfb1=nfb2=1`
- Ne traite que la première observation du cube `sobstot`
- Position spatiale fixée à `i=iic=(iim+1)/2` (centre horizontal)
- Échantillonne `j=1:jjm par 10` (≈12 profils verticaux)