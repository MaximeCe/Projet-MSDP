# Preuve de faisabilité — estimation automatique du nombre de canaux (nm)

> **Objectif (option A3)** : prouver qu'on peut déterminer automatiquement le
> nombre de canaux MSDP (`nm`) et l'espacement depuis le flat field, sans saisie
> manuelle, pour automatiser la boucle d'ingestion multi-instruments.
>
> Prototype **jetable** dans `/tmp/msdp_proto/` (ne touche pas au pipeline).
> Données de validation : flat Meudon `m011_stack20170330_y1.fit` (nm attendu = **9**).

---

## 1. Conclusion

**FAISABLE, avec une nuance à connaître.**

- ✅ **`nm` est détectable automatiquement** par la méthode des **maxima locaux
  sur une coupe 1D à Y fixe** (pas la moyenne sur Y — qui brouille la structure).
- ✅ **Espacement inter-canaux** mesuré ~150 px (médiane robuste).
- ⚠️ La **robustesse n'est pas de 100 %** : sur la grille d'hyperparamètres, la
  détection donne tantôt 9, tantôt 8 (le canal central a un double-max).
  Le **vote multi-lignes** fait pencher vers 9 (62 % des votes).

Ceci **confirme que la boucle de correction automatique prévue est nécessaire**
(voir le plan d'ensemble) : l'estimation doit être suivie d'un run + check, et
d'une itération si le check échoue.

---

## 2. La méthode qui marche

### Le piège évité
Moyenner le flat sur **toutes les lignes Y** brouille la structure (les canaux
sont des bandes verticales qui varient le long de Y). Résultat : 7-8 pics au lieu
de 9. → **Travailler sur des coupes 1D à Y fixe.**

### La méthode retenue
Pour chaque ligne Y (dans la zone utile ~15-85 % de la hauteur) :
1. **Lisser** la coupe le long de X (fenêtre ~21-31).
2. **Détecter les maxima locaux** (`scipy.signal.find_peaks`, distance ≥ 15,
   prominence ≥ 30 % de l'amplitude).
3. **Compter les pics** = candidats canaux.
4. **Voter** sur plusieurs lignes Y × plusieurs fenêtres de lissage → `nm` =
   valeur la plus fréquente.

### Résultats observés (flat Meudon, nm=9 attendu)

| Méthode | Résultat |
|---|---|
| Pics sur coupe Y fixe (bon hyperparam) | 9 canaux sur **21/21 lignes** Y testées |
| Vote multi-lignes × lissages | **nm=9 sur 62 %** des couples, 8 sur 38 % |
| Seuil sur gradient (sauts) | ~19-21 frontières → **surestime** (flat sans paliers nets) |
| Moyenne sur Y | 7-8 pics → **sous-estime** |

---

## 3. Points clés de robustesse

1. **Coupe Y fixe > moyenne sur Y.** C'est la découverte décisive.
2. **`find_peaks` (prominence)** plus fiable que le seuil sur le gradient —
   le flat a des pentes douces, pas de paliers franches.
3. **Le dernier canal / double-max** cause l'ambiguïté 8-vs-9. Un critère
   d'**espacement régulier** (rejeter un pic trop proche du précédent, < 0.5×
   la médiane) stabilise.
4. **La région Y utile** importe : loin des bords mal éclairés.

---

## 4. Ce qu'il faudra dans le module d'auto-param (A-volets)

```python
# Pseudo-API (à implémenter dans src/python/auto_params.py)
def estimate_nm(flat) -> (int nm, float spacing):
    votes = []
    for Y in y_lines():
        for w in lissages:
            peaks = find_peaks(smooth(flat[Y,:], w), distance=15, prominence=0.3*(range))
            votes.append(len(filter_regular_spacing(peaks)))
    return mode(votes), median_spacing
```

Le module final devra **combiner** :
- `im`, `jm` : lus du header FITS (trivial).
- `nm` : vote par pics (cette méthode).
- `mingrad` / autres seuils : boucle de correction après check (voir plan).

---

## 5. Limites de cette preuve

- Validé sur **un seul flat Meudon** (9 canaux). La robustesse réelle
  multi-instruments (7, 12 canaux, capteurs différents) **reste à tester** sur
  les données BASS2000 téléchargées.
- Un canal **saturé, éteint ou défectueux** produira un faux compte → la boucle
  de correction (run + check + ré-estimation) est **indispensable**, pas
  optionnelle.
- L'espacement est secondaire (il dérive de `nm` et de `im`) ; le livrable
  critique est `nm`.

---

## 6. Fichiers (prototype jetable)

Dans `/tmp/msdp_proto/` :
| Fichier | Contenu |
|---|---|
| `proto_inspect.py` | charge le flat, profils X/Y |
| `proto_cuts.py` | **coupes à Y fixe → 9 pics** (clé) |
| `proto_peaks*.py` | détection pics, robustesse hyperparams |
| `proto_edges*.py` | tentative par sauts (échoue → écarté) |
| `proto_nm.py` | compte sur 21 lignes Y → nm=9 (100 %) |
| `proto_vote.py` | **vote multi-lignes/lissages → nm=9 (62 %)** |

---

## 7. Recommandation

La **faisabilité est prouvée** pour l'estimation de `nm`/espacement. Je recommande
de poursuivre avec :
1. **A1** : figer ces résultats dans une spécification (ce doc fait office de
   brouillon).
2. **A2** : refactoriser `ms2.f`/`ms2.py` (extraire `im, jm, nm, ja` des constantes
   en dur) — condition nécessaire pour que `nm` estimé serve réellement.
3. **Module auto_params** : implémenter `estimate_nm` + correction itérative
   (run → check → ré-estimer si échec).

Le prototype confirme que le point le plus risqué (compter les canaux) est
**maîtrisable** — à condition de l'intégrer dans une boucle de validation.