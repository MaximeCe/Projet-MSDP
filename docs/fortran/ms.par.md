# `ms.par` — Fichier de paramètres MSDP

> Format : texte de type `"disk.par"` — chaque ligne : **nom (8 car.) + valeur (i8)**,
> soit `(a8,i8)`. Fichier source : `src/fortran/ms.par` (137 lignes).
> Utilisé par `ms1.f` (`readpar`, `par1`) et `ms2.f` (`par1`).

---

## 1. Généralités

- Les **caractères `*`** (position 18) marquent les paramètres **souvent modifiés**.
- Convention générale : **`1` = action active**, `0` = inopérant, sauf indication.
- C'est **`par1('   nom  ', nw, variable)`** qui lit chaque paramètre à la demande,
  en re-ouvrant `ms.par` à chaque appel (lecture répétée).

---

## 2. Étapes successives (successive steps)

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `ixy` | 1 * | Moyennes des fichiers dark, flat, field-stop |
| `igeom` | 1 * | Géométrie |
| `iflat` | 0 * | Calibration |
| `ibmc` | 0 | Fichiers c élémentaires calibrés (1 fichier/temps) |
| `icmd` | 0 | Fichiers d (spectrohéliogrammes et cartes I/V) |
| `ides` | 0 | Dessins (`1` ; `0` = pas de tracé) |
| `iquick` | 0 | q-fichiers pour cibles "full scanned" |
| `igrayq` | 0 | Tracés des q-fichiers |

Tracés automatiques après calcul (`0` = plots disponibles après seulement avec `gv`) :

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `igeo` | 1 | Plot `geo.ps` |
| `iflat1` | 0 | `flat1.ps` |
| `iflat2` | 0 | `flat2.ps` |
| `ical` | 0 | `cal.ps` |

---

## 3. Observations sélectionnées

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `nfx1` | 1 | 1er fichier dark utilisé |
| `nfx2` | 1 | dernier fichier dark utilisé |
| `nfy1` | 1 | 1er fichier flat utilisé |
| `nfy2` | 1 | dernier fichier flat utilisé |
| `nfb1` | 0 | 1er fichier obs utilisé |
| `nfb2` | 0 | dernier fichier obs utilisé |
| `nob1` | 1 | 1re image à traiter |
| `nob2` | 5 | dernière image à traiter |
| `ntmax` | 5 | nb. d'images par scan |
| `priscan` | 4 | = 4 pour Meudon (5 images), scan par prismes (2,3,1,5,4) |
| `nobstep` | 5 | pas entre les 1res images de 2 scans |
| `dob` | 20170330 | date d'observation (optionnelle) |
| `tob1` | 10200351 * | 1er temps d'observation traité |
| `tob2` | 24000000 | dernier temps |
| `tdc1` | 0 | idem pour dark |
| `tdc2` | 24000000 | — |
| `tfs1` | 0 | idem pour field-stop |
| `tfs2` | 0 | — |
| `calfs` | -1 | -1 si le flat Y remplace le field-stop Z |
| `tff1` | 0959000 | idem pour flat field |
| `tff2` | 2400000 | — |
| `nff` | 1 | nb. de flats |

---

## 4. Géométrie

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `kold` | 0 | résultats avec ancienne géométrie |
| `knew` | 1 | résultats avec nouvelle géométrie (les deux possible en série) |
| `is` | 1536 | dimension X du CCD |
| `js` | 1024 | dimension Y du CCD |
| `li` | 442000 | longueur du field-stop (arcsec/1000) |
| `lj` | 61000 | largeur du field-stop |
| `jypas` | 45000 | translation entre images (pour corrélations) |
| `nline` | 1 | index de la raie spectrale |
| `ncam1` | 1 | index du détecteur |
| `nm` | 9 | nb. de canaux (paramétrable — le pipeline s'adapte, cf. nouveaugeom) |
| `ja1` | 151 | position de la 1re coupe de détection (j, 1-based) |
| `ja2` | 501 | position de la 2e coupe de détection (j, 1-based) |
| `ja3` | 851 | position de la 3e coupe de détection (j, 1-based) |
| `xdel` | 25 | décalage latéral des colonnes de détection k,l,m,n |
| `jtriple` | 1 | 1 = moyenner 3 lignes autour de chaque coupe |
| `lbda` | 6563 | longueur d'onde de la raie (Å) |
| `dlbd` | 300 | distance en longueur d'onde entre canaux (mÅ) |
| `mupris` | 9000 | translation entre canaux de sortie (microns) |
| `mustep` | 2500 | distance entre fentes successives (microns) |
| `nwinp` | 1 | nb. de détecteurs simultanés |
| `interc` | 15 | distance approx. entre bord droit d'un canal et bord gauche du suivant (pixel CCD) |
| `nbcln` | 1024 | nb. final de pixels X-CCD |
| `nblgn` | 1536 | nb. final de pixels Y-CCD |
| `invern` | 0 | 0 : j du même λ diminue avec les canaux ; 1 : augmente |
| `idc` | 1 | 1 = dark soustrait ; 0 = non soustrait ; -1 = dark indisponible |

### Seuils de détection

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `si` | 15 | seuil d'intensité pour détection des bords de canaux vs X ; 0 = auto (si,sgi,sj,sgj) |
| `mingrad` | 18 | seuil min du gradient d'intensité dans `newgeom` |
| `interp` | 1 | interpolation parabolique des gradients d'intensité |
| `sgi` | 8 | seuil d'intensité des gradients vs X |
| `sj` | 15 | le même pour l'intensité vs Y |
| `sgj` | 5 | le même pour gradients vs Y |
| `iadd` | 0 | pour z(j) ajoute les données ii-iadd … ii+iadd |

### Angles et précisions

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `milangi` | -40 | angle approximé entre le bord long des canaux et le CCD (radian/1000) |
| `milgeo` | 2000 | seuil de précision géométrique : dép. max entre valeurs et droites de régression (plot geo.ps, unit pixel/1000) |
| `nleft` | 0 | approx. interpolées pour canaux défectueux (gauche) |
| `nright` | 0 | le même (droite) |

### Pixels utiles

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `i1` | 1 | 1er pixel utile en i |
| `i2m` | 0 | dernier pixel utile en i : `i2 = im - i2m` (im = nb total de pixels) |
| `j1` | 1 | définitions identiques pour j |
| `j2m` | 0 | … |

### Courbure / détection précise

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `lip` | 40 | courbure des canaux déterminée par 3 intervalles autour de 3 points du bord long ; si L longueur du bord, points à `L*(0.5−lip/100)` et `L*(0.5+lip/100)` |
| `jeps` | 20 | détermination précise des bords les plus longs recherchée autour des valeurs approx ± `jeps` pixels |
| `intvi` | 60 | bords ∥ i déterminés par coupes le long de j, moyennées sur ± `intvi` autour des 3 points (voir `lip`) |
| `intvj` | 30 | définition similaire pour bords courts ∥ j, avec intervalles : extrémité gauche − `intvj` … gauche, droite … droite + `intvj` |
| `leps` | 50 | détection des points de gradient max en 2 étapes : valeurs approx (signal = seuil d'intensité), puis recherche du gradient max dans ± `leps` |
| `n1` | 1 | canaux utiles : `n1 ≤ n ≤ (nm−n1+1)` où nm = nb total de canaux observés |
| `distor` | 1 | 1 = courbure prise en compte pour la détection (vérif. exclue du calcul) ; 0 = courbure non prise en compte |

---

## 5. Divers (Linux)

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `calfs` | -1 | -1 = le field-stop est remplacé par le flat pour la géométrie |
| `iswap` | 1 | 1 pour LINUX (swap d'octets big-endian → little-endian) |
| `milsec` | 500 | pixel de sortie (unit : arcsec/1000) |

---

## 6. Notes

- Le fichier se termine par la ligne **`end`** (mot-clé qui arrête la lecture de
  `readpar` / `par1`).
- Certains paramètres sont **hermétiques / peu commentés** dans le code (`normsq`,
  `norm`, `largrid`, `kdangle`) et non utilisés dans cette version "newgeom".
- `mingrad` (18) est le paramètre **critique** de la nouvelle géométrie : fenêtre
  recommandée 17–19 (voir discussions projet : `15` donne un faux pic sur le canal 7,
  `20+` décale les bords longs).
- Les valeurs `*` sont celles qu'on modifie habituellement d'une observation à
  l'autre (`ixy, igeom, ...` et les temps `tob*`).