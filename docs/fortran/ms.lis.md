# `ms.lis` — Journal (log) de traitement MSDP

> Chemin : `data/output/ms.lis` (751 lignes dans cet exemple, 38 Ko)
> Produit par : `ms1.f` / `ms2.f` — le programme écrit tout ce qui se passe dans le
> journal (unité logique **3**) via `write(3,...)`.
> Role : **trace complète** d'une exécution : paramètres lus, fichiers traités,
> moyennes, et surtout **détails de la géométrie des canaux** (détection des bords,
> distorsion, coins). C'est le premier fichier à consulter quand la géométrie s'écarte
> de l'attendu.

---

## 1. Les 3 grandes parties du fichier

Dans l'ordre, `ms.lis` contient :

| # | Lignes (approx.) | Contenu | Source (sous-programme) |
|---|------------------|---------|--------------------------|
| A | 1–138 | **Écho des paramètres** lus dans `ms.par` | `readpar` |
| B | 139–220 | **Moyennes** dark (`x`) et flat (`y`) + en-têtes FITS | programme principal + `readfits` |
| C | 221–751 | **Géométrie** : détection bords, distorsion, coins ABCDEF | `geom` → `SRECT` → `newgeom` |

> ⚠️ Partie C vu la plus importante : c'est elle qui permet de **valider la géométrie**
> et de **détecter les canaux mal détectés**.

---

## 2. Partie A — Écho des paramètres (lignes 1–138)

Le programme rejoue tout le contenu du fichier `ms.par` au démarrage, sous la forme
`<n°> <nom> <valeur>` (format `i3, a8, i8`). C'est **le paramètre réellement lu**
par `readpar` — pas celui que vous croyez avoir mis dans `ms.par`.

```text
          85  mingrad          18        ← paramètre mingrad = 18
```

Repérages utiles :
- **Ligne 11** : `ixy 1` → étape moyennes active.
- **Ligne 12** : `igeom 1` → géométrie active.
- **Ligne 58–59** : `is 1536`, `js 1024` → dimensions CCD (X, Y).
- **Ligne 65** : `nm 9` → nombre de canaux.
- **Ligne 85** : `mingrad 18` → seuil de gradient (critique pour `newgeom`).

> S'il y a un décalage entre ce que vous avez mis dans `ms.par` et ce qui est rejoué ici,
> c'est que le fichier a un problème de format (nom mal aligné sur 8 caractères,
> valeur inattendue). C'est la **1re section à contrôler** après chaque changement de
> paramètres. Elles s'arrêtent à la ligne `end`.

---

## 3. Partie B — Moyennes dark / flat (lignes 139–220)

### 3.1 Écho des lectures `par1`
Lignes 139–146 : rappel des paramètres extraits pour les bornes de fichiers
(`nfx1, nfx2, nfy1, nfy2, nfb1, nfb2, is, js`), format `par1 <nom> <nw> <valeur>`.

### 3.2 Boucle 1 : dark `x`
```text
Loop nxy            1
nf file(nxy,nf)    1  m010_b0101_ms_20170330_09564585_x1.fit
xname x170330_09564585_00000        ← nom du fichier dark moyen généré
nxy,  useful nfiles(nxy)=            1           1   (1 fichier utilisé)
Average nxy            1
from file            1  to            1
head: SIMPLE  = ... BITPIX = 16 NAXIS1=1536 NAXIS2=1024 ... END   ← en-tête FITS
readfits ipermu= 1
readfits iu,inbhead,is,js,ipermu,ktab   21   1  1536  1024  1  1
```
- `head:` = contenu de l'en-tête FITS du 1er fichier (dimensions, BITPIX).
- `denom= 1.000` : diviseur utilisé pour la moyenne (= nombre de fichiers).
- **Ensuite sont dumpées des lignes** `j   v(i1) v(i2) ...` : des **échantillons de
  l'image moyenne** (toutes les 100 pixels), utilisées pour du débogage visuel.
  Les valeurs `60` partout pour le dark = signal d'obscurité raisonnable (bruit ~0).

### 3.3 Boucle 2 : flat `y`
Même structure, mais le fichier `m011_stack20170330_y1.fit` donne un flat avec de la
**structure** (valeurs 200–2000 = signal réel des canaux, pics du champ plat) :
```text
 yname  30_y1.fit       00000
 30_y1.fit       00000    denom= 1.000  extreme points     67   70   98   68
```
- Les `extreme points` = les 4 coins de l'image moyenne (min/max).
- La grande variation 67→2000 confirme un **flat correctement exposé** avec les
  9 canaux visibles.

---

## 4. Partie C — Géométrie (lignes 221–751) ★ cœur à comprendre

### 4.1 En-tête
```text
 enter geom
debut geom: nw,gname           1 g30_y1.fit       00000
par1  igeo ... par1  distor ...       ← rappel des paramètres de géométrie lus
geom kz(2),kz(3)  (im,jm)        1024        1536   ← dimensions du flat
geom: i1,i2m,im, j1,j2m,jm  1 0 1024  1 0 1536
```

> 🔎 La géométrie travaille sur `ima` de dimensions **im=1024, jm=1536** — c'est
> le flat **permuté** (le rectangle CCD 1536×1024 devient 1024×1536).

### 4.2 Soustraction du dark
```text
par1     idc       1       1       ← dark soustrait (idc=1)
```
- `ima(i,j) = flat(i,j) − dark(i,j)` (bornée à ≥1).
- On dump `ima` de test (extrait 10%/90%).

### 4.3 `newgeom` — détection des bords sur 3 coupes
C'est le cœur. La coupe par défaut est `meanflat(i,512)`.

```text
newgeom: ja 1,2,3          151         501         851   ← 3 lignes en j (cut)
par1  mingrad       1      18
Newgeom   meanflat:          ← profil d'intensité de la coupe centrale (valeurs)
edges for 3 j-values
 zmax,zgmax     1986.0       665.0                      ← max intensité / gradient
```

**Chaque coupe** `ja(1..3)` produit 2 blocs "edges" (bord gauche, signe `+1`, puis
bord droit, signe `-1`) :

```text
 ja(           1 )=          151
edges: sig, l, n, iedge(n,is), zg(iedge-1/0/+1)  eps     XX      YY
      1.  1  1   111      26.  79.  72.   0.38  110.38  150.00
```

Décodage d'une ligne "edge" :

| Champ | Valeur | Explication |
|-------|--------|-------------|
| `sig` | `1.` (gauche) / `-1.` (droite) | signe du gradient détecté |
| `l` | `1..3` (coupes 1–3) / `4..6` (coupes 4–6) | identifiant du bord long (a,b,c = d,e,f) |
| `n` | `1..9` | numéro du **canal** |
| `iedge` | `111` | position entière du bord en i |
| `zg(edge-1/0/+1)` | `26. 79. 72.` | valeurs du gradient aux 3 pixels autour |
| `eps` | `0.38` | décalage parabolique d'interpolation (si `interp=1`) |
| `XX` | `110.38` | position finale affinée = `iedge + eps − 1` |
| `YY` | `150.00` | position en j de la coupe (`ja−1`) |

> ✅ **Lecture rapide** : pour une géométrie saine, chaque canal doit avoir une ligne
> dans chaque coupe, avec des `XX` qui croissent régulièrement d'un canal au suivant
> (**pas d'espacement chaotique**, pas de valeur aberrante).

### 4.4 Distorsion
```text
 distortion:  0.171  quadratic mean value ...   ← écart quadratique moyen des courbures
distortion: sig    n    distortion
             1.    1   -0.00
             1.    2   -0.11
             ...
            -1.    9    0.07
```
- Une distorsion lisse, **faible** (~0.1–0.4 px) et sans valeur énorme = bonne
  géométrie. Des valeurs de plusieurs dizaines indiquent un **bord mal détecté**.

### 4.5 Intersections → coins A–F
Pour chaque canal, `intersec` calcule les coins du quadrilatère :
```text
 A: interp,n, x1,x2,x2,x4, y1,y2,y3,y4, xres,yres
 1  1    97.4 110.4 135.0 207.0  500.0 150.0 59.9 58.2  113.6  60.4
```
- 2 premières lignes : les 4 points de départ (x puis y) de l'intersection.
- 3e ligne : `a,b,c,d` puis `xres,yres` (le coin calculé).
- Lettres : `A,C,D,F` sont calculés par intersection ; `B,E` sont recopiés.

### 4.6 Tableau récapitulatif des 16 points (lignes 572–623)
C'est la **vue d'ensemble la plus lisible** : pour chaque `nl` (1–16), deux lignes
donnent `xx` puis `yy` pour les 9 canaux.

| `nl` | Point | Signification |
|------|-------|---------------|
| 1–3 | `a,b,c` | bord **gauche** aux coupes j=150,500,850 |
| 4–6 | `d,e,f` | bord **droit** aux coupes j=150,500,850 |
| 7–10 | `k,l,m,n` | **bas/haut** (bords courts) |
| 11–16 | `A,B,C,D,E,F` | **coins** du quadrilatère du canal |

Exemple (lignes 577–578) pour `nl=1` :
```text
nl=           1
110.38  257.93  412.59  558.87  711.51  864.52 1012.56 1166.94 1315.50   ← xx (i)
150.00  150.00  150.00  150.00  150.00  150.00  150.00  150.00  150.00    ← yy (j=const)
```
→ le bord gauche a des positions i qui augmentent régulièrement (~147 px entre
canaux) : c'est le rythme attendu des canaux MSDP.

### 4.7 "dd" / `plotgeo*` / fin
Les lignes `nd, xdes, ydes` (départ ~664) sont les polygones des bords des canaux
passés à PGPLOT pour tracer `geo1.ps`. La série `plotgeo3:` (lignes 738–747) donne
les variations de dimensions ΔAC, ΔDF, ΔAD, ΔCF en X et Y par canal.

Le fichier se termine par `end geom`.

---

## 5. 🔴 Signaux d'alerte — comment repérer un canal raté

Le point **le plus important pour l'analyse** : dans cet exemple, les canaux **5 et 6**
(eu 4–5/6 selon indexation) montrent des valeurs **aberrantes** que vous devez savoir
reconnaître :

```text
   4       0.00     0.00   558.87   545.48   532.04   681.53   ← canal 4 partiel
   5       0.00     0.00     0.00     0.00     0.00     0.00    ← canal 5 : TOUT nul
   6     820.95   803.19*********  0.00*********  0.00         ← canal 6 : valeurs absurdes
```
et dans les polygones (`nd`) :
```text
 nd, xdes, ydes   3   1.13096469E+33   0.00000000      ← "infini" (pixel non défini)
```

Quand vous repérez ces motifs, cherchez la cause dans `ms.par` :

| Symptôme | Cause probable | Correctif |
|----------|----------------|-----------|
| Colonne entièrement `0.00` (canal non détecté) | seuil `mingrad` trop haut / canaux non exposés, ou `interc` incohérent | baisser `mingrad` (≈17–19), vérifier `si/sgi` |
| Valeurs `E+33` / `********` (débordement) | intersection avec des points de départ nuls → division par zéro dans `intersec` | correction manuelle `nleft`/`nright`, ou vérifier coupes `lip/leps` |
| Espacement irrégulier entre canaux | bord mal détecté sur une coupe | ajuster `jeps,intvi,leps,SMAX` |

> 🔎 Rappel projet : `mingrad=18` est le **bon** réglage (fenêtre 17–19) ; `15`
> produit un faux pic (canal 7) et `20+` décale les bords longs. Une colonne nulle
> ou `E+33` dans `ms.lis` = canal non résolu = **à corriger avant d'exploiter les
> cartes**.

---

## 6. Résumé de lecture en 60 secondes

1. **Vérifier les paramètres** (lignes 11–138) : `mingrad`, `is,js`, `nm` correspondent-ils ?
2. **Vérifier les moyennes** (139–220) : les fichiers `x`/`y` sont-ils les bons ? `denom` cohérent ?
3. **Scan rapide des `xx` récapitulatifs** (572–623) : chaque canal a-t-il des valeurs
   non-nulles et régulières ?
4. **Vérifier la distorsion** (351–371) : valeurs faibles et sans géant ?
5. **Contrôler les coins ABCDEF** (644–663) : aucun `0.00` ni `E+33` ?
6. Si tout est bon → la géométrie est valide, on peut utiliser les sorties `geo*.ps`
   et `ACDF2.lis`.

---

## 7. Fichiers associés dans `data/output/`
- `ACDF2.lis` : les 8 valeurs X/Y des coins (AX,CX,DX,FX,AY,CY,DY,FY) par canal.
- `geo1.ps` / `geo1.pdf` : tracé de contrôle (profil d'intensité, gradient, contours
  des 9 canaux) — le **pendant visuel** de `ms.lis`.
- `geo1_fortran_*.pdf` : captures de différentes runs pour comparaison.
- Les fichiers moyens `x..._00000`, `y..._00000` produits par la partie B.