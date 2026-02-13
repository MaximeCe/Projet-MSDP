@echo off
REM Script d'automatisation pour compilation et execution Fortran (VERSION CORRIGEE)
REM ================================================================================

echo ========================================
echo Compilation et Execution Fortran Docker
echo ========================================
echo.

REM Verifier si Docker est installe
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Docker n'est pas installe ou n'est pas dans le PATH
    echo Veuillez installer Docker Desktop depuis https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo Docker detecte: OK
echo.

REM Construire l'image Docker
echo [1/3] Construction de l'image Docker...
docker build -t fortran-compiler .
if errorlevel 1 (
    echo ERREUR: La construction de l'image a echoue
    pause
    exit /b 1
)
echo Construction reussie!
echo.

REM Compiler le programme (DANS le conteneur, sans monter de volume)
echo [2/3] Compilation du code Fortran...
docker run --rm --name fortran-compile fortran-compiler ./compile.sh
if errorlevel 1 (
    echo ERREUR: La compilation a echoue
    pause
    exit /b 1
)
echo Compilation reussie!
echo.

REM Executer le programme et recuperer les resultats
echo [3/3] Execution du programme...
docker run --rm -v "%cd%":/app/output --name fortran-run fortran-compiler bash -c "cd /app && ./ms_program && cp *.lis /app/output/ 2>/dev/null || true"
if errorlevel 1 (
    echo ATTENTION: L'execution a rencontre des problemes
    echo Consultez les messages ci-dessus pour plus de details
)

echo.
echo ========================================
echo Termine!
echo ========================================
echo.
echo Fichiers de sortie dans le dossier actuel:
dir /b *.lis 2>nul
if errorlevel 1 (
    echo Aucun fichier .lis trouve
    echo Le programme peut necessiter des fichiers d'entree supplementaires
)
echo.
pause
