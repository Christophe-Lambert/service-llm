FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Mettre à jour les paquets
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    gcc g++ \
    libopenblas-dev liblapack-dev libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

# Vérifier que Python 3 est installé et configuré comme python par défaut
RUN ln -s /usr/bin/python3 /usr/bin/python


# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier uniquement le fichier requirements.txt (pour maximiser l'utilisation du cache)
COPY service/requirements.txt /app/requirements.txt

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt && pip install protobuf
RUN pip install accelerate
# Installer bitsandbytes
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install bitsandbytes --index-url https://pypi.ngc.nvidia.com

# Copier les fichiers de votre application dans le conteneur
COPY service /app

# Exposer le port que Flask va utiliser
EXPOSE 5000

# Démarrer le serveur Flask
CMD ["python", "app.py"]
