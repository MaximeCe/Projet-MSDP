@echo off
REM Menu interactif pour gestion Docker Fortran
REM ===========================================

:menu
cls
echo ========================================
echo  MENU DOCKER FORTRAN
echo ========================================
echo.
echo 1. Construire l'image Docker
echo 2. Compiler le code
echo 3. Executer le programme
echo 4. Compiler ET executer
echo 5. Ouvrir un shell dans le conteneur
echo 6. Voir les fichiers de sortie
echo 7. Nettoyer (supprimer conteneurs et images)
echo 8. Quitter
echo.
set /p choice="Choisissez une option (1-8): "

if "%choice%"=="1" goto build
if "%choice%"=="2" goto compile
if "%choice%"=="3" goto run
if "%choice%"=="4" goto compile_and_run
if "%choice%"=="5" goto shell
if "%choice%"=="6" goto list_files
if "%choice%"=="7" goto clean
if "%choice%"=="8" goto end
echo Choix invalide!
pause
goto menu

:build
echo.
echo Construction de l'image Docker...
docker build -t fortran-compiler .
pause
goto menu

:compile
echo.
echo Compilation du code...
docker run --rm -v "%cd%":/app fortran-compiler ./compile.sh
pause
goto menu

:run
echo.
echo Execution du programme...
docker run --rm -v "%cd%":/app fortran-compiler ./run.sh
pause
goto menu

:compile_and_run
echo.
echo Compilation et execution...
docker run --rm -v "%cd%":/app fortran-compiler bash -c "./compile.sh && ./run.sh"
pause
goto menu

:shell
echo.
echo Ouverture du shell (tapez 'exit' pour sortir)...
docker run -it --rm -v "%cd%":/app fortran-compiler /bin/bash
pause
goto menu

:list_files
echo.
echo Fichiers de sortie:
dir /b *.lis *.ps 2>nul
if errorlevel 1 echo Aucun fichier de sortie trouve
pause
goto menu

:clean
echo.
echo Nettoyage...
docker container prune -f
docker image rm fortran-compiler 2>nul
echo Nettoyage termine!
pause
goto menu

:end
echo.
echo Au revoir!
exit
