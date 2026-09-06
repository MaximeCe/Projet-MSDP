# `ms1.f` — Programme principal et moyennes dark/flat

> Code **FORTRAN 77** — fichier source : `src/fortran/ms1.f` (677 lignes)
> Rôle : **orchestrateur de la chaîne MSDP** (multi-channel subtractive double pass).
> Il prépare les fichiers la base, gère les données d'entrée et appelle les
> sous-programmes de géométrie définis dans `ms2.f`.

---

## 1. Rôle général

`ms1.f` contient le **programme principal** mais aussi plusieurs **sous-programmes
utilitaires** (parsing du fichier de paramètres, lecture des fichiers FITS, swap
d'octets, comptage d'en-têtes, ouverture de fichiers, moindres carrés).

Le programme principal réalise, dans l'ordre :

1. **Lecture des paramètres** dans le fichier `ms.par` (via `readpar` / `par1`).
2. **Construction des listes de fichiers** dark (`x`), flat (`y`), field-stop (`z`),
   observations (`b`).
3. **Boucle sur les séquences x et y** :
   - lecture des fichiers individuels,
   - **permutation** des coordonnées CCD (fonction du `ipermu`),
   - **sommation** (`tabaver`) puis **moyenne** (`tab2`),
   - écriture des fichiers moyens `x...` (dark) et `y...` (flat).
4. Appel du **calcul de géométrie** `geom` (défini dans `ms2.f`).
5. (Commenté) appel des étapes `ms3.f` (canalisation et calibration).

> **Important** : dans cette version, les étapes aval (`channels`, `calib`, fichiers
> `c*`, `d*`) sont **commentées** — on s'arrête après le calcul de géométrie.

---

## 2. Conventions de nommage internes

| Nom | Lecture | Description |
|-----|---------|-------------|
| `x` | dark | courant d'obscurité (list/lecture/moyenne) |
| `y` | flat | champ plat (list/lecture/moyenne) |
| `z` | field-stop | géométrie du champ (list/lecture/moyenne) |
| `b` | obs | observations cibles (lecture seule) |
| `g` | — | fichier moyen field-stop / obs utilisé par la géométrie |

Coordonnées (d'après les commentaires en tête de fichier) :

| Étape | Fichier | Coordonnées CCD | Dimensions |
|-------|---------|-----------------|------------|
| CCD / `flat1.ps` | `imb, imima` | `1536` | `jmb, jmima` `1024` |
| XY permutation | `imc` | `1024` | `jmc` `1536` |
| canaux / `flat2.ps` | `imd` | `885` | `jmd` `123`, `nm=9` |
| filtregrammes & Doppler | `ime` | `< 885` | `jme` `< 123`, `nm=9` |

Avec une fenêtre `milsec=500` : la fenêtre vaut `li/1000 = 442″` et `lj/1000 = 61″`.

---

## 3. Programme principal (PROGRAM ms1)

### 3.1 Variables globales
- `file(4,200)` : noms des fichiers par séquence (4 types) et par numéro.
- `chead` : en-tête FITS (2880 octets) lu pour comptage.
- `tabpermu`, `tabaver`, `tab2`, `cymx`, `ima`, `cliss` : tableaux d'images 1536×1536,
  `integer*2` ou `integer*4` selon usage.
- `xname,yname,zname,bname,gname` : noms de fichiers moyens en sortie (22 caractères).
- `nfa(4), nfb(4), nfiles(4)` : bornes (début/fin) et nombre de fichiers par séquence.

### 3.2 Initialisation et lecture des paramètres
```fortran
xname = 'x000000_00000000_00000'
yname = 'y000000_00000000_00000'
...
call readpar                 ! lit ms.par en mode "balayage"
call par1('    nfx1',nw,nfx1) ! extrait chaque paramètre nommé
...
```
- Supprime `channel.lis` et `ms.lis`, les rouvre en écriture (unités 95 et 3).
- `sundec=0`, `uint=0`, `ipermu=1`, `ijcam=1536`.

### 3.3 Bornes des séquences (dark/flat)
Les paramètres `nfx1/nfx2` (dark), `nfy1/nfy2` (flat), `nfz1/nfz2` (field-stop),
`nfb1/nfb2` (obs) sont lus, puis :
```fortran
nfa(1)=nfx1 ; nfb(1)=nfx2 ; nfiles(1)=nfx2-nfx1+1   ! dark
nfa(2)=nfy1 ; nfb(2)=nfy2 ; nfiles(2)=nfy2-nfy1+1   ! flat
...
```
`is`, `js` (dimensions CCD) sont lus ; avec `ipermu=1`, les dimensions de travail
sont **échangées** : `isp=js`, `jsp=is`.

### 3.4 Boucle principale `nxy` (moyennes dark puis flat)
```
do 300 nxy=1,2
```
Pour `nxy=1` (dark) : `ls m*x1.fit > xtab.lis`, ouverture unité 11.
Pour `nxy=2` (flat) : `ls m*y1.fit > ytab.lis`, ouverture unité 12.

**Boucle de lecture** `do 100 nf=1,nfb(nxy)` : lit le nom de fichier depuis la liste,
puis :

```fortran
call openold38(file(nxy,nf),sundec,iu)   ! ouverture directe
call counthead(iu,nhead,chead)           ! comptage du nombre de blocs d'en-tête
call readfits(iu,nhead,iswap,is,js,ipermu,tab2,tabpermu,ktab)
```
- Le tableau `tab2` (avant permutation) est converti en `tabpermu` (permuté)
  directement dans `readfits`.
- **Sommation** : `tabaver(ip,jp) += tabpermu(ip,jp)`.
- En-tête et coins (`tab2(1,1)`, …) sont journalisés dans `ms.lis`.

**Moyenne** : après la boucle,
```fortran
kpiv = nfb(nxy)-nfa(nxy)
denom(nxy) = float(kpiv)+1.
tab2(ip,jp) = float(tabaver(ip,jp))/denom(nxy) + 0.5
```

**Écriture du fichier moyen** (unité `iut = 30+nxy`) :
- en-tête `head(1)=3, head(2)=isp, head(3)=jsp, head(4)=1`, le reste à 0 ;
- puis chaque ligne du tableau `tab2` en binaire non formaté.

> Le fichier moyen porte le nom `name(nxy)` extrait du dernier fichier de la séquence
> (positions 17–33 du nom). Pour `nxy=2`, on en dérive aussi `yname`, `zname` et
> `gname` (le flat remplace alors le field-stop).

### 3.5 Calcul de la géométrie
```fortran
nw=1 ; win(1)=1 ; win(2)=0 ; nm=9
ima = ...          ! image flat-donc-dark
imb=1536 ; jmb=1024 ; imc=1024 ; jmc=1536

call geom(nw,win,nm,31,32,32,gname,istop,ima,ijcam,imima,jmima,
    1                                              xr,yr,imc,jmc)
```
`geom` est défini dans `ms2.f`. Les unités 31/32 correspondent aux fichiers
moyens (obs/flat). Le reste de la fin du programme (`channels`, `calib`, PGPLOT)
est **commenté**.

---

## 4. Sous-programmes

### 4.1 `readpar` — lecture "balayage" de `ms.par`
- Ouvre `ms.par` (unité 96), lit par paires `(nom, nombre)` au format `(a8,i8)`.
- S'arrête quand `nom == 'end     '`.
- **Ne stocke rien** : sert à vérifier/formater la lecture.

### 4.2 `par1(name, nw, nombre)` — extraction d'un paramètre nommé
- Relit `ms.par` de façon répétée pour trouver la ligne dont le nom de 8 caractères
  correspond à `name`.
- Remplit `nombre` (sortie) et journalise `nom, nw, nombre` dans `ms.lis`.
- Retourne sans valeur si `end` est atteint sans correspondance.
- **C'est l'accès standard aux paramètres** utilisé partout dans `ms1.f`/`ms2.f`.

### 4.3 `readfits(iu, inbhead, iswap, is, js, iperm, tab2, tabpermu, ktab)`
Lecture d'une image FITS brute et permutation.

Arguments (entrées) : `iu` unité, `inbhead` nb de blocs d'en-tête, `iswap`
(1 = swap d'octets LINUX), `is`,`js` dimensions CCD, `ipermu` (1 = permuter),
`ktab` (1 = journaliser le tableau).
Sorties : `tab2` (avant permut), `tabpermu` (permuté).

Algorithme :
- Lecture par bloc de 1440 entiers `*2` (records directs) à partir du 1er bloc de
  données (`n=inbhead`).
- Remplissage ligne par ligne de `tab2(i,j)`, en appliquant `swap` sur chaque pixel
  si `iswap=1` (et `lswap=1` au pixel central 512,512 pour diagnostics).
- Boucle sur `i` puis passage à la ligne `j` suivante ; fin (`ios<0`) → `1000`.
- **Permutation** si `ipermu=1` :
  ```fortran
  tabpermu(ip,jp) = tab2(i,j)  avec ip = js+1-j, jp = i
  ```
  (transposition + renversement : logicien 1536→1024, 1024→1536).
- Si `ktab=1`, imprime des extraits de `tab2` et `tabpermu`.

### 4.4 `swap(in, ncar, out, lswap)` — swap d'octets
- Inverse l'ordre des octets de chaque entier `integer*2` (big-endian ↔ little-endian,
  nécessaire sous LINUX).
- Utilise une `equivalence` entre un `integer*2` et 2 `logical*1`.

### 4.5 `counthead(iu, nb, chead)` — comptage des blocs d'en-tête
- Lit les 10 premiers records de 2880 octets ; parcourt par blocs de 80 octets en
  cherchant le mot `'END '` (fin de l'en-tête FITS).
- Rend le nombre de blocs d'en-tête `nb`.

### 4.6 `openold38(name1, sundec, iu)` — ouverture directe
- Ouvre un fichier en **accès direct non formaté**, longueur de record :
  `720` (16 bits, `sundec=1`) **ou** `2880` octets (sinon).

### 4.7 Sous-programmes d'ouverture `*22`
- `opennew22(xyname, iu)` : ouverture séquentielle nouvelle (binaire) 22 caractères.
- `openold22(name, sundec, iu)` : ouverture séquentielle lecture (binaire).
- `opennew22sf` / `openold22sf` : variantes **formatées** (`sf` = with format).

### 4.8 `comptehead(iu, nb)` — variante de `counthead`
- Même principe mais journalise `buf(1:600)` de chaque record dans `ms.lis`.

### 4.9 `DPMCAR(X, Y, P, ND, NT, COEF)` — moindres carrés (Schneider 75)
- Calcule, en **double précision**, le polynôme de `NT` termes (max 10) `Y=F(X)`
  associé à `ND` mesures, avec poids `P` (accès discuté d'après l'en-tête).
- Résolution par pivot de Gauss sur la matrice C(11,11) (normales).
- `COEF(i)` rend les coefficients, avec changement d'échelle `X/1000`.

---

## 5. Notes et pièges

- Le programme s'appuie sur des **unités logiques non standard** (3, 7, 11, 12, 31,
  32, 95, 96). Modifier l'ordre d'ouverture peut casser les correspondances.
- Les séquences `z` (field-stop) et `b` (obs) sont **prévues mais commentées** dans
  cette version : on ne traite que **dark (x)** et **flat (y)**.
- Le calcul de géométrie (`geometrie ms2.f`) est appelé avec le fichier moyen `g`
  (flat) en sorte que le dark y soit déjà retranché.
- `ipermu=1` impose la permutation CCD → cartes (1024×1536). Si `ipermu=0`,
  `tabpermu = tab2` (aucune permutation).