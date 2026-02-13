# GUIDE RAPIDE - Démarrage en 5 minutes

## 🚀 Installation rapide

### Étape 1 : Installer Docker Desktop
1. Télécharger : https://www.docker.com/products/docker-desktop
2. Installer et redémarrer votre PC
3. Lancer Docker Desktop

### Étape 2 : Préparer vos fichiers
Créez un dossier (ex: `C:\fortran-project`) et copiez-y :
- ✅ Dockerfile
- ✅ ms.par
- ✅ ms1.f
- ✅ ms2.f
- ✅ run_fortran.bat (optionnel)
- ✅ menu_fortran.bat (optionnel)

### Étape 3 : Exécuter

#### Option A : Script automatique (le plus simple)
Double-cliquez sur `run_fortran.bat`

#### Option B : Menu interactif
Double-cliquez sur `menu_fortran.bat` et suivez les instructions

#### Option C : Ligne de commande
```bash
# Ouvrir PowerShell dans le dossier
cd C:\fortran-project

# Tout en une commande
docker build -t fortran-compiler . && docker run --rm -v "%cd%":/app fortran-compiler bash -c "./compile.sh && ./run.sh"
```

## 📋 Commandes essentielles

### Construction
```bash
docker build -t fortran-compiler .
```

### Compilation
```bash
docker run --rm -v "%cd%":/app fortran-compiler ./compile.sh
```

### Exécution
```bash
docker run --rm -v "%cd%":/app fortran-compiler ./run.sh
```

### Shell interactif
```bash
docker run -it --rm -v "%cd%":/app fortran-compiler
```

## 🔍 Vérification

Après exécution, vous devriez voir dans votre dossier :
- `ms.lis` - fichier log principal
- `channel.lis` - informations sur les canaux
- Autres fichiers selon la configuration

## ❓ Problèmes courants

### Docker ne démarre pas
→ Vérifiez que la virtualisation est activée dans le BIOS

### "docker: command not found"
→ Redémarrez votre PC après l'installation de Docker

### Erreur de compilation
→ Vérifiez que tous les fichiers .f et .par sont présents

### Fichiers manquants après exécution
→ Utilisez l'option `-v "%cd%":/app` pour monter le volume

## 📚 Fichiers inclus

| Fichier | Description |
|---------|-------------|
| Dockerfile | Configuration de l'environnement Docker |
| README_FR.md | Documentation complète |
| run_fortran.bat | Script d'exécution automatique |
| menu_fortran.bat | Menu interactif |
| ms.par | Paramètres du programme |
| ms1.f | Programme principal Fortran |
| ms2.f | Module de géométrie Fortran |

## 💡 Conseils

1. **Première utilisation** : La construction de l'image prend 1-2 minutes
2. **Modifications** : Après avoir modifié les fichiers .f, reconstruisez l'image
3. **Résultats** : Utilisez `-v "%cd%":/app` pour que les fichiers restent sur Windows
4. **Débogage** : Utilisez le shell interactif pour explorer

## 🆘 Besoin d'aide ?

Consultez le fichier **README_FR.md** pour :
- Instructions détaillées
- Explication des paramètres
- Résolution de problèmes
- Personnalisation avancée

---

**Version simplifiée** - Pour la documentation complète, voir README_FR.md
