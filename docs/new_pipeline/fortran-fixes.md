# Corrections Fortran nécessaires — ms3.f / ms4.f

## Problème
Le pipeline Fortran complet (étapes 1-4) crash à cause de deux bugs dans le code `new/` :
1. **`ms3.f` / `intvel3`** : variables `diff(nla)`/`diff(nlb)` non définies → segfault
2. **`ms4.f` / `ivmap4`** : appel inconditionnel de `intvel4` + débordement d'indices

---

## Correction 1 : `ms3.f` ligne 333 — Bug `intvel3`

**Fichier** : `/home/max/nextcloud/Workspace/Projet-MSDP/src/fortran/new/ms3.f`

**Ligne 333 actuelle** :
```fortran
write(3,*)' diff(nla) diff(nlb) dn xl yl ',
     1           diff(nla),diff(nlb),dn,xl,yl
```

**À remplacer par** :
```fortran
write(3,*)' diff(nla1) diff(nlb1) dn xl yl ',
     1           diff(nla1),diff(nlb1),dn,xl,yl
```

**Pourquoi** : Les variables `nla` et `nlb` n'existent pas — seules `nla1`/`nla2` et `nlb1`/`nlb2` sont définies (lignes 318-321). Le code utilise des variables non initialisées → segfault.

---

## Correction 2 : `ms4.f` lignes 262-283 — Appel inconditionnel de `intvel4`

**Fichier** : `/home/max/nextcloud/Workspace/Projet-MSDP/src/fortran/new/ms4.f`

**Bloc actuel (lignes 262-283)** :
```fortran
      enddo
c      write(3,*)'intvel4 iic jj nm nfb1 lbdvel nlm ',
c     1     iic,jj,nm,nfb1,lbdvel,nlm
c     if(lbdvel.ne.0)then
     
c      goto 198
      call pgwindow(80.,0.,0.,2000.)
      lbdvel=lbdvel1
      call pgslw(4)
      call intvel4(jj,pro,n0,lbdvel,nff1,nff2,nlm)   !,xv2,yv2)
c           write(4,4) 4,lbdvel1,xv2,yv2
      call pgslw(3)
      lbdvel=lbdvel2
      call pgsls(4)
      call pgslw(6)
      call intvel4(jj,pro,n0,lbdvel,nff1,nff2,nlm)   !,xv2,yv2)
c              write(4,4) 4,lbdvel2,xv2,yv2
      call pgsls(1)
      call pgslw(3)
c         call intvel5(jj,prof5,320,nf1,nf2,nlm)        
c     endif
c 198  continue
 200  continue                          !  j
      call pgend
```

**À remplacer par** :
```fortran
      enddo
      if(lbdvel1.ne.0)then
         call pgwindow(80.,0.,0.,2000.)
         lbdvel=lbdvel1
         call pgslw(4)
         call intvel4(jj,pro,n0,lbdvel,nff1,nff2,nlm)
         call pgslw(3)
      endif
      if(lbdvel2.ne.0)then
         call pgsls(4)
         call pgslw(6)
         lbdvel=lbdvel2
         call intvel4(jj,pro,n0,lbdvel,nff1,nff2,nlm)
         call pgsls(1)
         call pgslw(3)
      endif
      call pgend
```

**Pourquoi** : Le code original a les `if` commentés et un `goto` mal placé → `intvel4` est **toujours appelé** même si `lbdvel=0`. De plus `intvel4` accède à `prof(10,100)` avec des indices `nl+lbdvel` qui débordent quand `lbdvel>0`.

---

## Correction 3 : `ms4.f` ligne ~423 — Protection indices `intvel4`

**Fichier** : `/home/max/nextcloud/Workspace/Projet-MSDP/src/fortran/new/ms4.f`

**Ligne 423 actuelle** :
```fortran
      do 4 nl=nf1,nlmb
         diff(nl)=prof(nl+lbdv)-prof(nl)
```

**À protéger** :
```fortran
      do 4 nl=nf1,nlmb
         if (nl+lbdv .le. nf2) then
            diff(nl)=prof(nl+lbdv)-prof(nl)
         else
            diff(nl)=0.0
         endif
```

**Pourquoi** : `nlmb = nf2 - lbdvel` mais la boucle va jusqu'à `nlmb`, donc `nl+lbdv` peut atteindre `nf2` (OK) mais si `lbdvel` mal calculé ou `nf1>1`, débordement possible. Le tableau `prof(10,100)` n'a que 100 éléments par `n0`.

---

## Correction 4 : `ms.par` — Borner `lbdvel`

```par
lbdvel1      10    fist width of profile for veocity measurment
lbdvel2      10    second ...
lbdvel3      10    3    (0 = no lbdvel)
```

**Pourquoi** : `nlm=81` (points λ interpolés), `nf2 ≤ 71` (fenêtre 41 pts). `lbdvel > 10` fait sortir des bornes.

---

## Résumé des 4 changements

| # | Fichier | Ligne(s) | Type | Impact |
|---|---------|----------|------|--------|
| 1 | `ms3.f` | 333 | Fix typo variable | Répare segfault `intvel3` |
| 2 | `ms4.f` | 262-283 | Rétablir `if` conditionnel | `intvel4` appelé seulement si `lbdvel≠0` |
| 3 | `ms4.f` | 423 | Borne indice | Protège `prof(10,100)` |
| 4 | `ms.par` | 40-43 | Valeurs ≤10 | Évite débordements |

---

## Procédure d'application

```bash
cd /home/max/nextcloud/Workspace/Projet-MSDP/src/fortran/new

# 1. ms3.f - ligne 333
sed -i "s/diff(nla),diff(nlb)/diff(nla1),diff(nlb1)/" ms3.f

# 2. ms4.f - bloc 262-283 (édition manuelle requise - voir bloc complet ci-dessus)
# 3. ms4.f - ligne 423 (bounds check)
# 4. ms.par - valeurs lbdvel

# Recompiler
gfortran -g -o msdp ms1.f ms2.f ms3.f ms4.f \
    -lpgplot -L/usr/lib/x86_64-linux-gnu -lX11 \
    -lgfortran -lquadmath

# Tester
./run_pipeline.sh
```

---

## Après corrections

Le pipeline devrait :
- ✅ Compiler sans erreur
- ✅ Exécuter étapes 1-4 sans segfault
- ✅ Produire `ivprof1/2/3.ps` + vitesses bisector dans `miv.lis`
- ✅ Passer `check_fortran_pipeline.py`