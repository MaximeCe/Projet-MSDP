# Sorties du pipeline MSDP — à quoi correspond chaque fichier

Guide limité aux sorties réellement produites par `run_pipeline.sh` (un run complet = dossier
`data/output/run_NNN/`). Pour chaque sortie : ce qu'elle représente physiquement et comment la lire.

---

## Les données numériques

| Fichier | À quoi ça correspond | Comment le lire |
|---|---|---|
| `ACDF2_run_NNN.lis` | **Géométrie spectrographique des 9 canaux.** Les coordonnées des 4 coins (A, C, D, F) de chaque canal MSDP, en pixels. C'est LA sortie qui décrit la forme de chaque canal dans le champ. | 9 lignes × 8 colonnes : `Aₓ Aᵧ Cₓ Cᵧ Dₓ Dᵧ Fₓ Fᵧ`. X croissants (150 px/canal), Y ≈ constant → canaux rectilignes parallèles. Validé par le checker (0 `0.00`/`****`). |
| `miv_run_NNN.lis` | **Vitesses de décalage Doppler (bisector)** en chaque point de mesure — la carte des vitesses radiales solaires déduites du déplacement du profil de raie Hα. | Une série de blocs `intvel`, un par position : `lbdvel` (largeur utilisée en px), `xv2`, `yv2` d'abord en pixels puis en **km/s**. |
| `ms_run_NNN.lis` | **Journal complet** de l'exécution : paramètres lus, étapes franchies (`enter geom`, `entree calib`, `solarobs`, `ivmap`), et les tableaux intermédiaires imprimés. | Les lignes `intvel3 diff`, `cal n,i,j,…`, `profnf` etc. sont des débug de valeurs à estaver. Chercher les marqueurs d'étape pour vérifier qu'un run est allé au bout (`fin calib` en dernier). |
| `ms_par_run_NNN.par` | **Snapshot des paramètres** utilisés pour ce run (mingrad, ja1/2/3, lbdvel, nfb, seuils de détection…). | Copie fidèle du `ms.par` au moment du run — indispensable pour rejouer/reproduire. |

---

## Les plots de géométrie (coordonnées d'un scan MSDP)

Les 4 plots montrent la couverture spatiale du spectrographe : comment les **9 canaux** se
répartissent dans le champ 1536×1024 et comment ce pavage est déformé.

| Fichier | À quoi ça correspond |
|---|---|
| `geo1_fortran_NN.pdf` | **Détection des bords** : 3 coupes de l'image (haut/milieu/bas) sur lesquelles les sauts d'intensité des inter-canaux sont repérés. C'est la matière première — les traits verticaux = les bords trouvés. |
| `geo2_fortran_NN.pdf` | **Coins ABCDEF** : le pavage complet des 9 canaux tracé à partir des coins (même info que ACDF2.lis, en version graphique). |
| `geo3_fortran_NN.pdf` | **Distorsions AC/DF/CF** : l'écart entre bord attendu (droite) et bord réellement mesuré, le long du champ. Révèle les courbures/deformations optiques. |
| `geo4_fortran_NN.pdf` | **Bords interpolés** : le pavage lissé/reconstruit après interpolation, prêt à être utilisé pour calibrer. |

> À noter : de petites distorsions sont normales (astigmatisme du spectrographe type Meudon).
> Une géométrie "saine" = coins X réguliers, Y quasi-alignés.

---

## Les plots de calibration (intensité)

| Fichier | À quoi ça correspond |
|---|---|
| `calib_fortran_NN.pdf` | **Profils de raie Hα centrés** : pour chaque canal, le profil spectral (intensité vs λ) avec le centre de raie repéré. Sert à vérifier que les canaux sont bien au même λ0 ≈ 6563 Å après calibration. |
| `cal_fortran_NN.pdf` | **Maps de calibration photométrique par canal (__les plus lourdes, ~2 Mo__)** : la correction d'intensité à appliquer en chaque pixel — montre où le détecteur est hétérogène (poussières, vignettage). |

---

## Les plots d'observations (le Soleil lui-même)

Ce sont les reconstructions du disque solaire à partir des données calibrées, dans l'ordre de
traitement croissant :

| Fichier | À quoi ça correspond |
|---|---|
| `obs_fortran_NN.pdf` | Observation brute normalisée (image solaire telle que sortie de la caméra, avant calibration photométrique). |
| `obsD1_fortran_NN.pdf` | Observation **non calibrée** en longueur d'onde (position spectrale telle quelle, étape intermédiaire). |
| `obsD2_fortran_NN.pdf` | Observation **calibrée** : image solaire où l'axe spectral (λ → vitesse Doppler) est correctement étalonné. C'est la reconstruction la plus aboutie des observations. |

> Échelle visuelle : dessiner 1 pixel = 1 colonne spectrale ; la carte I/V se lit canal par canal.

---

## Les plots de profils I/V (physique de la raie Hα)

Trois variantes du même calcul : l'interpolation de la forme de la raie (intensité I et polarisation V)
pour reconstruire des cartes doppler/profilés propres :

| Fichier | À quoi ça correspond |
|---|---|
| `ivprof1_fortran_NN.pdf` | Profils I/V — **interpolation polynomiale de degré 3**. |
| `ivprof2_fortran_NN.pdf` | Profils I/V — **interpolation de degré 4**. |
| `ivprof3_fortran_NN.pdf` | Profils I/V — **mixte** (comparaison degré 3 vs degré 4 sur le même profil). |

> Usage : comparer ivprof1 (lisse, robuste) vs ivprof2 (plus fidèle aux formes raides) ;
> ivprof3 permet de juger si le degré change la mesure (=> choix du degré par dataset).

---

## Arborescence d'un run complet (run_009, 16 fichiers)

```
data/output/run_009/
├── ACDF2_run_009.lis         géométrie (chiffres)
├── ms_run_009.lis            journal
├── miv_run_009.lis           vitesses Doppler (km/s)
├── ms_par_run_009.par        paramètres
├── geo1_fortran_09.pdf       bords
├── geo2_fortran_09.pdf       coins
├── geo3_fortran_09.pdf       distorsions
├── geo4_fortran_09.pdf       bords interpolés
├── calib_fortran_09.pdf      profils de raie
├── cal_fortran_09.pdf        maps de calibration
├── obs_fortran_09.pdf        obs brute
├── obsD1_fortran_09.pdf      obs non calibrée
├── obsD2_fortran_09.pdf      obs calibrée
└── ivprof1/2/3_fortran_09.pdf  profils I/V (deg 3, deg 4, mixte)
```

---

## Sorties annoncées mais non produites

| Sortie | Signification | Statut |
|---|---|---|
| c-files | Fichiers calibrés élémentaires (un par temps) | flag `ibmc=0` |
| d-files | Spectrohéliogrammes + cartes I/V | flag `icmd=0` |
| q-files | Scans complets (étapes 5–7 de la méthode Mein) | non implémenté |
| `flat2` | Canaux extraits de l'image flat | `map3` non appelé |