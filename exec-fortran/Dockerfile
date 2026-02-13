# Image Docker pour compilation et exécution de programmes Fortran avec PGPLOT
FROM ubuntu:20.04

# Éviter les prompts interactifs pendant l'installation
ENV DEBIAN_FRONTEND=noninteractive

# Installation des outils de compilation Fortran et dépendances PGPLOT
RUN apt-get update && apt-get install -y \
    gfortran \
    g++ \
    gcc \
    make \
    vim \
    nano \
    wget \
    libx11-dev \
    libpng-dev \
    x11proto-core-dev \
    && rm -rf /var/lib/apt/lists/*

# Installation de PGPLOT depuis les dépôts Ubuntu 20.04
RUN apt-get update && apt-get install -y \
    pgplot5 \
    libpgplot0 \
    && rm -rf /var/lib/apt/lists/*

# Configuration des variables d'environnement pour PGPLOT
ENV PGPLOT_DIR=/usr/lib/pgplot5
ENV PGPLOT_DEV=/NULL

# Création du répertoire de travail
WORKDIR /app

# Copie des fichiers source
COPY ms.par /app/
COPY ms1.f /app/
COPY ms2.f /app/

# Configuration pour les fichiers de sortie
RUN mkdir -p /app/output

# Script de compilation avec PGPLOT dans /usr/local/bin
RUN echo '#!/bin/bash' > /usr/local/bin/compile-fortran && \
    echo 'cd /app' >> /usr/local/bin/compile-fortran && \
    echo 'echo "Compilation des fichiers Fortran avec PGPLOT..."' >> /usr/local/bin/compile-fortran && \
    echo 'gfortran -o ms_program ms1.f ms2.f -std=legacy -w -L/usr/lib -lpgplot -lX11 -lpng -lz' >> /usr/local/bin/compile-fortran && \
    echo 'if [ $? -eq 0 ]; then' >> /usr/local/bin/compile-fortran && \
    echo '    echo "Compilation reussie!"' >> /usr/local/bin/compile-fortran && \
    echo '    echo "Executable cree: ms_program"' >> /usr/local/bin/compile-fortran && \
    echo '    ls -lh ms_program' >> /usr/local/bin/compile-fortran && \
    echo 'else' >> /usr/local/bin/compile-fortran && \
    echo '    echo "Erreur de compilation"' >> /usr/local/bin/compile-fortran && \
    echo '    exit 1' >> /usr/local/bin/compile-fortran && \
    echo 'fi' >> /usr/local/bin/compile-fortran && \
    chmod +x /usr/local/bin/compile-fortran

# Script d'exécution dans /usr/local/bin
RUN echo '#!/bin/bash' > /usr/local/bin/run-fortran && \
    echo 'cd /app' >> /usr/local/bin/run-fortran && \
    echo 'if [ -f /app/ms_program ]; then' >> /usr/local/bin/run-fortran && \
    echo '    echo "Execution du programme..."' >> /usr/local/bin/run-fortran && \
    echo '    ./ms_program' >> /usr/local/bin/run-fortran && \
    echo '    echo ""' >> /usr/local/bin/run-fortran && \
    echo '    echo "Fichiers generes:"' >> /usr/local/bin/run-fortran && \
    echo '    ls -lh *.lis *.ps 2>/dev/null || echo "Aucun fichier de sortie genere"' >> /usr/local/bin/run-fortran && \
    echo 'else' >> /usr/local/bin/run-fortran && \
    echo '    echo "Erreur: le programme n a pas ete compile."' >> /usr/local/bin/run-fortran && \
    echo '    echo "Executez d abord: compile-fortran"' >> /usr/local/bin/run-fortran && \
    echo '    exit 1' >> /usr/local/bin/run-fortran && \
    echo 'fi' >> /usr/local/bin/run-fortran && \
    chmod +x /usr/local/bin/run-fortran

# Créer aussi les scripts dans /app pour compatibilité
RUN ln -s /usr/local/bin/compile-fortran /app/compile.sh && \
    ln -s /usr/local/bin/run-fortran /app/run.sh

# Commande par défaut: ouvrir un shell bash
CMD ["/bin/bash"]
