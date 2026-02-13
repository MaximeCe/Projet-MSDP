# Guide d'utilisation Docker pour programmes Fortran

## Description
Cette image Docker vous permet de compiler et exécuter vos programmes Fortran (ms1.f et ms2.f) avec le fichier de paramètres ms.par dans un environnement Linux isolé, même sur Windows 10.

## Prérequis

### Installation de Docker Desktop sur Windows 10

1. **Télécharger Docker Desktop**
   - Visitez : https://www.docker.com/products/docker-desktop
   - Téléchargez la version pour Windows
   - Version minimum requise : Windows 10 64-bit Pro, Enterprise, ou Education

2. **Installer Docker Desktop**
   - Exécutez le fichier d'installation téléchargé
   - Suivez les instructions à l'écran
   - Redémarrez votre ordinateur si demandé

3. **Vérifier l'installation**
   - Ouvrez PowerShell ou l'invite de commandes
   - Tapez : `docker --version`
   - Vous devriez voir la version de Docker installée

## Structure des fichiers

Créez un dossier sur votre PC (par exemple `C:\fortran-project`) et placez-y les fichiers suivants :
```
fortran-project/
├── Dockerfile          (ce fichier configure l'image Docker)
├── ms.par             (fichier de paramètres)
├── ms1.f              (premier fichier source Fortran)
└── ms2.f              (second fichier source Fortran)
```

## Construction de l'image Docker

1. **Ouvrir PowerShell ou l'invite de commandes**
   - Clic droit sur le menu Démarrer > Windows PowerShell
   - Ou cherchez "cmd" dans le menu Démarrer

2. **Naviguer vers votre dossier**
   ```bash
   cd C:\fortran-project
   ```

3. **Construire l'image Docker**
   ```bash
   docker build -t fortran-compiler .
   ```
   
   Cette commande :
   - `-t fortran-compiler` : donne le nom "fortran-compiler" à l'image
   - `.` : utilise le Dockerfile dans le dossier actuel
   - Durée : environ 1-2 minutes selon votre connexion internet

## Utilisation

### Méthode 1 : Compilation et exécution séparées

1. **Démarrer le conteneur**
   ```bash
   docker run -it --name fortran-container fortran-compiler
   ```
   
   Vous êtes maintenant dans le conteneur Linux (vous verrez un prompt comme `root@abc123:/app#`)

2. **Compiler les programmes**
   ```bash
   ./compile.sh
   ```
   
   Cette commande compile ms1.f et ms2.f ensemble pour créer l'exécutable `ms_program`

3. **Exécuter le programme**
   ```bash
   ./run.sh
   ```

4. **Quitter le conteneur**
   ```bash
   exit
   ```

### Méthode 2 : Compilation directe

Pour compiler sans entrer dans le conteneur :
```bash
docker run --rm -v "%cd%":/app fortran-compiler ./compile.sh
```

Pour exécuter après compilation :
```bash
docker run --rm -v "%cd%":/app fortran-compiler ./run.sh
```

### Méthode 3 : Tout en une commande

```bash
docker run --rm -v "%cd%":/app fortran-compiler bash -c "./compile.sh && ./run.sh"
```

## Gestion des fichiers de sortie

Le programme génère plusieurs fichiers de sortie (ms.lis, channel.lis, etc.). Pour les récupérer :

1. **Avec montage de volume** (recommandé)
   ```bash
   docker run -it -v "%cd%":/app fortran-compiler
   ```
   
   Les fichiers créés dans `/app` seront automatiquement disponibles dans votre dossier Windows

2. **Copier depuis le conteneur**
   ```bash
   # Démarrer le conteneur avec un nom
   docker run -it --name fortran-work fortran-compiler
   
   # (Après exécution, dans une autre fenêtre PowerShell)
   docker cp fortran-work:/app/ms.lis C:\fortran-project\
   docker cp fortran-work:/app/channel.lis C:\fortran-project\
   ```

## Commandes Docker utiles

### Gestion des conteneurs

```bash
# Lister les conteneurs actifs
docker ps

# Lister tous les conteneurs (même arrêtés)
docker ps -a

# Arrêter un conteneur
docker stop fortran-container

# Redémarrer un conteneur existant
docker start -i fortran-container

# Supprimer un conteneur
docker rm fortran-container

# Supprimer tous les conteneurs arrêtés
docker container prune
```

### Gestion des images

```bash
# Lister les images
docker images

# Supprimer une image
docker rmi fortran-compiler

# Supprimer les images non utilisées
docker image prune
```

### Accéder à un conteneur en cours d'exécution

```bash
docker exec -it fortran-container /bin/bash
```

## Modification des fichiers source

### Option 1 : Modifier sur Windows puis reconstruire

1. Modifiez ms1.f, ms2.f ou ms.par avec votre éditeur préféré sous Windows
2. Reconstruisez l'image :
   ```bash
   docker build -t fortran-compiler .
   ```

### Option 2 : Modifier dans le conteneur

1. Démarrez le conteneur avec un éditeur :
   ```bash
   docker run -it fortran-compiler /bin/bash
   ```

2. Utilisez nano ou vim pour éditer :
   ```bash
   nano ms1.f
   # ou
   vim ms1.f
   ```

3. Sauvegardez et compilez :
   ```bash
   ./compile.sh
   ```

## Compilation personnalisée

Si vous voulez compiler avec des options spécifiques, dans le conteneur :

```bash
# Compilation standard
gfortran -o ms_program ms1.f ms2.f -std=legacy

# Avec optimisation
gfortran -o ms_program ms1.f ms2.f -std=legacy -O3

# Avec informations de débogage
gfortran -o ms_program ms1.f ms2.f -std=legacy -g

# Avec affichage de tous les avertissements
gfortran -o ms_program ms1.f ms2.f -std=legacy -Wall
```

## Explication des options du compilateur

- `-std=legacy` : Permet la compilation de code Fortran ancien (77/90)
- `-w` : Supprime les avertissements
- `-O3` : Optimisation maximale
- `-g` : Inclut les informations de débogage
- `-Wall` : Affiche tous les avertissements

## Résolution des problèmes

### Docker Desktop ne démarre pas
- Vérifiez que la virtualisation est activée dans le BIOS
- Vérifiez que WSL 2 est installé (Windows Subsystem for Linux)

### Erreur "docker: command not found"
- Redémarrez votre ordinateur après l'installation de Docker
- Vérifiez que Docker Desktop est en cours d'exécution

### Erreur de compilation
- Vérifiez que tous les fichiers (ms1.f, ms2.f, ms.par) sont présents
- Consultez les messages d'erreur détaillés
- Les fichiers peuvent nécessiter des bibliothèques supplémentaires

### Permission denied sur les fichiers
```bash
# Dans le conteneur
chmod +x compile.sh run.sh
```

### Le programme ne trouve pas ms.par
- Assurez-vous que ms.par est dans le même dossier que le Dockerfile
- Vérifiez que le fichier a été copié dans l'image :
  ```bash
  docker run fortran-compiler ls -la /app
  ```

## Architecture du programme

### ms1.f
Programme principal qui :
- Lit le fichier de paramètres ms.par
- Gère les fichiers d'entrée (dark current, flat field, observations)
- Coordonne le traitement des données

### ms2.f
Module de traitement géométrique qui :
- Effectue les calculs de géométrie pour les observations solaires
- Traite les canaux spectraux
- Génère les calibrations

### ms.par
Fichier de paramètres contrôlant :
- Les étapes de traitement (calculs, calibration, plots)
- La sélection des observations
- Les paramètres géométriques du CCD et du spectromètre
- Les seuils de détection

## Notes importantes

1. **Fichiers d'entrée** : Le programme s'attend à trouver des fichiers FITS d'observations. Vous devrez les ajouter au conteneur ou monter un volume contenant ces données.

2. **Résultats** : Les fichiers de sortie (ms.lis, channel.lis, fichiers PostScript) seront créés dans /app

3. **Performance** : La première exécution peut être lente car Docker doit télécharger l'image de base Ubuntu.

4. **Espace disque** : L'image Docker occupe environ 500 Mo.

## Support et ressources

- Documentation Docker : https://docs.docker.com/
- Documentation gfortran : https://gcc.gnu.org/onlinedocs/gfortran/
- Pour des problèmes spécifiques, conservez les messages d'erreur complets

## Exemple complet de session

```powershell
# 1. Navigation vers le dossier
cd C:\fortran-project

# 2. Construction de l'image
docker build -t fortran-compiler .

# 3. Exécution avec montage de volume (pour récupérer les résultats)
docker run -it -v "%cd%":/app --name fortran-work fortran-compiler

# Dans le conteneur :
# 4. Compilation
./compile.sh

# 5. Exécution
./run.sh

# 6. Vérifier les fichiers créés
ls -la

# 7. Sortir
exit

# 8. Nettoyer (si nécessaire)
docker rm fortran-work
```

## Automatisation avec un script batch

Créez un fichier `run_fortran.bat` dans votre dossier :

```batch
@echo off
echo Construction de l'image Docker...
docker build -t fortran-compiler .

echo.
echo Compilation et execution...
docker run --rm -v "%cd%":/app fortran-compiler bash -c "./compile.sh && ./run.sh"

echo.
echo Termine!
pause
```

Double-cliquez sur ce fichier pour tout exécuter automatiquement.

---

**Créé le :** 13 février 2026
**Version Docker :** Compatible avec Docker Desktop pour Windows 10+
**Compilateur :** gfortran (GNU Fortran) 11.x
