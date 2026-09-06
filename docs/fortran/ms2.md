# `ms2.f` — Géométrie des canaux (newgeom)

> Code **FORTRAN 77** — fichier source : `src/fortran/ms2.f` (1422 lignes)
> Rôle : **calcul de la géométrie des canaux** de l'image MSDP à partir du champ
> plat (et du dark), avec détection des bords de canaux, correction des distorsions
> et tracé des courbes/contours de contrôle (`geo1.ps`, `geo2.ps`, `geo3.ps`).
>
> Ce fichier définit la routine `geom` appelée par `ms1.f` et regroupe tout le
> sous-programme `SRECT` (détection), ainsi que les nouvelles routines
> `newgeom`, `intersec` et les traceurs `plotgeo1/2/3`.
>
> **⚠️ Historique** : une corruption mémoire (`ACDF2.lis` / figures) a été identifiée
> et corrigée — voir la **[Section 9 — Bogue corrigé (indexation d'array)](#9-bogue-corrigé-indexation-darray)**.

---

## 1. Rôle général

Les canaux MSDP sont des bandes parallèles de champ. La géométrie détermine, pour
chaque canal, la position précise de ses 4 coins (formant un quadrilatère) en
coordonnées écran/CCD. On détecte les **bords longs** (appelés i) et **bords courts**
(j) au moyen du gradient d'intensité du champ plat, puis on **corrige la distorsion**
(courbure) éventuelle et on produit des courbes de contrôle.

Conventions de dimensions (voir `ms.par`) :

| Étape | Coordonnées | Dimensions CCD |
|-------|-------------|----------------|
| CCD | `imb` `1536` × `jmb` `1024` | image brute |
| permutation/cartes | `imc` `1024` × `jmc` `1536` | cartes XY |
| canaux | `imd` `885` × `jmd` `123`, `nm=9` | canaux + longueur d'onde |

---

## 2. `geom` — point d'entrée (appelé par `ms1.f`)

```fortran
subroutine geom(nw,win,nm,iux,iuy,iuz,gname,istop,ima,ijcam,
    1                                        imima,jmima,xr,yr,imc,jmc)
```
- **Entrées** : `nw`, `win`, `nm` (nb. de canaux), `iux`/`iuy`/`iuz` (unités des
  fichiers dark/flat/field-stop), `gname`, `istop`, `ima`, `ijcam`.
- **Sorties** : `imima`, `jmima`, `xr(24,3,2)`, `yr(24,3,2)`, `imc`, `jmc`.

Étapes :
1. **Lecture des paramètres** `par1` : `igeo, interc, si, sj, sgi, sgj, milangi,
   milangj, milgeo, i1, i2m, j1, j2m, lip, jeps, intvi, intvj, leps, n1, distor,
   normsq, norm, largrid`.
2. Calcule `n2 = nm-n1+1`, `nc = (n1+n2)/2`.
3. Lit l'en-tête du fichier moyen flat (`iuy`) → `im=kz(2)`, `jm=kz(3)`.
4. Appelle **`SRECT`** avec tous ces paramètres.
   (`istop=1` termine si demandé.)

> Note : beaucoup de valeurs anciennes (angles fixes, `nbcln`, `interc*coef`) sont
> **commentées** ; c'est la version "newgeom" qui fait le travail.

---

## 3. `SRECT(...)` — détection des bords (ancienne approche, en partie conservée)

Signature longue (voir le code) : prend les seuils, intervalles, angles,
`iux,iuy,gname,ima,imima,jmima,milgeo,istop,ijcam,nw,normsq,norm,largrid,xr,yr`.

Travail principal réalisé **avant** `newgeom` :

1. **Seuils automatiques** : si `sip=0`, on balaye automatiquement les seuils sur
   les valeurs `ksi,ksgi,ksj,ksgj` (19 valeurs). `nsm=20` boucles de seuils ; le
   dernier passage (`nsol`) est le passage "de validité". Si `sip≠0`, on utilise les
   seuils explicites `si,sgi`.
2. **Lecture du dark et du flat** (`iux`, `iuy`) puis **soustraction dark** :
   ```fortran
   lec(i) = lec(i) - lecx(i)
   if (lec(i).lt.0) lec(i) = 1
   ima(i,j) = lec(i)
   ```
3. Stocke toute l'image `ima(1..im, 1..jm)`.
4. **Construit `meanflat`** (flat − dark transposé) :
   ```fortran
   do j=1,1024 ; do i=1,1536 ; meanflat(i,j) = ima(j,i) ; enddo ; enddo
   ```
5. Appelle **`newgeom(meanflat)`** — la géométrie proprement dite.

> La boucle `nseuils` et la plupart des diagnostics `ima` sont conservés ; le rôle
> de détection "à l'ancienne" (xbord/ybord, points a–f/k–l/m–n) est désormais
> **remplacé** par `newgeom`, plus robuste.

---

## 4. `SMAX(z, i, eps)` — interpolation parabolique du maximum

- Calcule le décalage parabolique autour d'un maximum de gradient :
  `a = z(i+1)+z(i-1)-2z(i)`, `b = z(i+1)-z(i-1)`, puis `eps = -b/(2a)`.
- Retourne sans rien si `a=0`. Active si le paramètre `interp=1`.

---

## 5. `newgeom(meanflat)` — détection des bords (méthode moderne)

C'est le cœur du calcul. Notre `meanflat` fait 1536×1024.

### 5.1 Phase d'initialisation
- `newgeom(meanflat, im, jm)` : dimensions `im`/`jm` reçues **en argument**
  (SRECT les déduit du header flat via la transposition). Fini le codage en dur
  (A2-bis). Tableaux à bornes max 4096².
- `nm` (nb de canaux) lu depuis `ms.par` (défaut 9).
- **3 coupes** horizontales en j : `ja = [ja1, ja2, ja3]` lus depuis `ms.par`
  (défaut 151, 501, 851). Paramétrables depuis la refactorisation A2.
- Lecture de `mingrad` (seuil minimum de gradient), `interp` (interpolation),
  `xdel` (décalage détection k,l,m,n) et `jtriple` (moyenne 3 lignes) depuis `ms.par`.
- Normalisation de la coupe centrale `zc` et de son gradient `zgc` à 0–100 %.
- Déclaration des tableaux de points de contrôle (bornes max 40 canaux pour
  permettre `nm` variable) :
  ```fortran
  dimension ... xx(20,40), yy(20,40), distort(2,40)
  ```
  Les sous-programmes `plotgeo1/2/3` déclarent aussi `xx(20,40)/yy(20,40)`
  (cohérents avec `newgeom`).
  > ⚠️ Ces tableaux étaient `xx(20,9)` avant A2 ; élargis à `(20,40)` pour
  > autoriser un nombre de canaux différent de 9.
  > où index **nl (1..16)** = point (abcdef / klmn / ABCDEF) et **nc (1..nm)** =
  > canal. La déclaration reste **alignée entre `newgeom` et les `plotgeo*`**
  > (mêmes bornes `(20,40)`) — voir la **[Section 9](#9-bogue-corrigé-indexation-darray)**.

### 5.2 Boucle sur les 3 coupes `nj=1,3`
Pour chaque coupe `jj=ja(nj)` :
- Extrait `z(i)` et son gradient `zg(i)` (moyenné sur 3 lignes si `jtriple=1`),
  normalisés à 0–100 %.
- **Pour les deux signes** de gradient (bord gauche/droit, `is=1,2`) :
  - cherche les positions `i` où le gradient (× signe) dépasse le seuil `zgt` **et**
    est un maximum local (`piv2 ≥ piv1` et `piv2 ≥ piv3`) ;
  - enregistre `iedge(n,is)` et, si `interp=1`, affine par `smax` ;
  - stocke `xx(l,n)`, `yy(l,n)` avec `l=nj` (coupes 1–3) ou `l=nj+3` (4–6).

Résultat : pour chaque canal `n`, les **6 points a,b,c,d,e,f**
(`nl = 1..6`) = les intersections des 2 bords avec les 3 coupes.

### 5.3 Distorsion (courbure)
Pour chaque canal : `distort = x(milieu) − (x(début)+x(fin))/2` sur chaque bord,
puis **écart quadratique moyen** `valqm` (en pixels). Journalise les valeurs par
canal.

### 5.4 Points k,l,m,n (bords courts en j)
- `xdel=25` : on prend des coupes verticales **juste à l'extérieur** des bords longs
  (décalées de ±25 pixels en i), sur les segments bas/haut du canal.
- On y détecte le gradient maximum pour déterminer les extrémités **bas/haut**
  `k,l,m,n` (`nl = 7..10`).

### 5.5 Points A,B,C,D,E,F — coins du quadrilatère
On calcule les **intersections** entre bords longs et bords courts :
- `A = intersection(ab, km)`, `B = intersection(c, ...)`, etc.
- La routine **`intersec`** est appelée pour chaque coin :
  ```fortran
  A : x1=x2?,  lignes (a,b) x (k,m)
  C : lignes (b,c) x (l,n)
  D : lignes (d,e) x (k,m)
  F : lignes (e,f) x (l,n)
  ```
- `B` et `E` sont simplement recopiés des points `b`/`e`.

### 5.6 Sorties
- Écrit **`ACDF2.lis`** (8 valeurs X/Y par canal : AX, CX, DX, FX, AY, CY, DY, FY).
- Appelle `plotgeo1`, `plotgeo2`, `plotgeo3` pour tracer les courbes de contrôle.

---

## 6. `intersec(x1,x2,x3,x4, y1,y2,y3,y4, xres,yres)` — intersection de 2 droites

Intersection du **bord long** (modélisé `x = a y + b`, pente faible) avec le
**bord court** (modélisé `y = c x + d`) :
```fortran
a = (x2-x1)/(y2-y1) ; b = x1 - a*y1
c = (y3-y4)/(x3-x4) ; d = y3 - c*x3
xres = (a*d + b)/(1 - a*c)
yres = c*xres + d
```
Rend `xres,yres` et journalise a,b,c,d.

---

## 7. Traceurs PGPLOT

### 7.1 `plotgeo1(zc, zgc, zgt, i1,i2,im,jm, nm, xx, yy, ja)` → `geo1.ps`
3 panneaux verticaux :
- haut : profil d'intensité de la coupe centrale `zc` ;
- milieu : gradient `zgc` avec les seuils `±grt` en pointillés ;
- bas : le plan (X,Y) avec les **contours des 9 canaux** (polygones ABCDEF) et les
  3 lignes horizontales aux coupes `ja`.
- Écrit `ACDF2.lis` puis `gv geo1.ps`.

### 7.2 `plotgeo2(xx, yy, xdel)` → `geo2.ps`
Zoom sur le **premier canal** (n=1) :
- lignes horizontales aux coupes `ja` ;
- points `abcdef` (nl=1..6) et `klmn` (nl=7..10) ;
- contours ABCDEF (nl=11..16) et **étiquettes de lettres** a–F ;
- verticales de repérage décalées de `±xdel`.

### 7.3 `plotgeo3(xx, yy)` → `geo3.ps` (ou `geo3b.ps` si `interp=0`)
6 panneaux (2 colonnes × 3 lignes) traçant les **fluctuations des dimensions** des
canaux (ΔAC, ΔDF, ΔAD, ΔCF) en **X** et en **Y**, en fonction du numéro de canal
(`ac: |A−C|`, `df: |D−F|`, `ad: |A−D|`, `cf: |C−F|`). Sert au contrôle de précision.

---

## 8. Notes et pièges

- **PGPLOT** est utilisé pour tout le tracé (`pgbegin`, `pgvport`, `pgbox`, ...).
  Sans bibliothèque PGPLOT, le programme ne peut pas tracer `geo*.ps`.
- Les appels `system('gv geoN.ps &')` ouvrent l'afficheur `gv` — à désactiver en
  mode automatique (voir `igeo`, paramètres de plots).
- `newgeom` utilise des dimensions **codées en dur** (1536, 1024, nm=9). Tout
  changement d'instrument impose de revoir ces constantes.
- La fonction `par1(' milangj')`, `normsq`, `norm`, `largrid` sont lus mais leur
  usage est en grande partie **commenté** dans cette version.
- C'est `meanflat` (flat − dark, transposé 1536×1024) qui sert de base à toute la
  détection `newgeom`.
- Le fichier `gname` (fichier moyen obs/flat) est passé mais surtout utilisé pour
  relire le dark ; le résultat final de "géométrie" est contenu dans tout le jeu
  de points `xx(nl,nc)/yy(nl,nc)` (nl=1..16, nc=1..nm).

---

## 9. Bogue corrigé — indexation d'array (`xx`/`yy`)

*Découvert et corrigé le 2026-09-04.*

### 9.1 Le symptôme

En observant `data/output/ms.lis` et `ACDF2.lis`, plusieurs canaux ressortaient
**invalides** alors que la détection semblait correcte :

- **canal 5** : colonne entièrement `0.00` dans `ACDF2.lis` ;
- **canal 4** : coin A nul (`0.00`) ;
- **canal 6** : valeurs `********` / `1.13E+33` (overflow) ;
- et paradoxalement les canaux **2, 3, 8, 9** semblaient "remplis" mais avec des
  valeurs **fausses** (reportées depuis d'autres points).

Simultanément, le **récapitulatif `nl=`** écrit dans `ms.lis` par `newgeom` montrait
une géométrie **valide pour les 9 canaux** (coin A du canal 5 = `714.21`, etc.).

→ Contradiction entre deux sorties du même calcul : le calcul était bon, seul
**l'écriture de sortie** était corrompue.

### 9.2 Le diagnostic (cause racine)

Deux déclarations **incompatibles** pour les mêmes tableaux `xx`/`yy` :

| Où | Déclaration |
|----|-------------|
| `newgeom` (qui **remplit** `xx`,`yy`) | `xx(24,24), yy(24,24)` |
| `plotgeo1/2/3` + `geom` (qui **lisent** via dummies) | `xx(20,9), yy(20,9)` |

En Fortran (stockage **column-major**), quand on passe un array `24×24` à un dummy
`20×9`, le sous-programme réinterprète la mémoire avec la nouvelle indexation :
`xx(i,j)` → `offset = i + (j−1)·20`.

- **canal 1** : `offset = 11` → tombe pile sur `xx(11,1)` → **bon** (d'où le canal 1 qui marchait).
- **canal 2** : lit `offset = 11+20 = 31` → dans le 24×24 c'est `xx(7,2)` (point **k**),
  pas le coin A → **faux** (d'où `283.00` au lieu de `261.05`).
- D'où les valeurs reportées d'un point à l'autre, les `0.00` et les NaN quand
  `intersec` divise par ~0 sur des points dégénérés.

### 9.3 La correction (1 ligne, dans `newgeom`)

```fortran
! AVANT  — newgeom
dimension ... xx(24,24), yy(24,24), distort(2,24)

! APRÈS  — aligné sur les dummy (20,9) des plotgeo*
dimension ... xx(20,9), yy(20,9), distort(2,24)
```

`newgeom` n'utilise que `nl ≤ 16` et `n ≤ 9`, donc `xx(20,9)` est largement
suffisant (`offset max = 16 + 8·20 = 176 < 180`). Aucun dépassement.

### 9.4 Vérification (avant / après)

| Canal | AVANT (run 001) | APRÈS (run 004) |
|-------|-----------------|-----------------|
| 1 | 113.59 / 80.87 / 235.57 / 202.53 | **identique** ✓ (ne bouge pas) |
| 5 | **0.00 ×8** (tout nul) | **714.21** 680.69 836.98 803.19 ✓ |
| 6 | 820.95 + NaN + 0 | **867.19** 833.68 990.35 956.30 ✓ |

- Le canal 1 (dont l'offset tombait juste) est **resté identique** → la correction
  ne casse rien.
- Les 9 canaux sont désormais **complets et cohérents** dans `ACDF2.lis` : plus aucun
  zéro, aucun NaN, doublets `A C D F` croissants.
- Les données `data/output/ACDF2_run_001.lis` (avant) et `ACDF2_run_004.lis` (après)
  servent de preuve ; les logs versionnés `ms_run_00X.lis` tracent toute l'itération.

### 9.5 Leçon / piège à retenir

- **Ne pas modifier `ms.par`** cherchait ce bug : un paramètre modifie les valeurs
  numériques mais **pas** le *motif* structurel des zéros/NaN (vérifié en changeant
  `interp` 1→0 : valeurs changées, motif d'échec identique). Ce type de signature
  (mêmes canaux ratés quelle que soit la config) pointe **toujours** vers un bug de
  code, pas un réglage.
- En Fortran 77, **l'indexation column-major** rend les déclarations d'array des
  procédures appelantes/calées sensibles : un mismatch de bornes produit un
  **aliasing mémoire silencieux** (pas d'erreur à la compilation ni à l'exécution).
- Si on retouche les dimensions, garder `xx(20,9)`/`yy(20,9)` **strictement
  identiques** entre `newgeom`, `geom`, et les `plotgeo*`.