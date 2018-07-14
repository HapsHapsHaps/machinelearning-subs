# Docker-TensorFlow-Object-Detection
TensorFlow Docker containers with Improvements and fixes for training Objekt Detection models

## Build

### CPU
docker build -t tensorflow-improved-cpu -f Dockerfile.CPU .

### GPU
nvidia-docker build -t tensorflow-improved-gpu -f Dockerfile.GPU .

#### Building on an Optimus laptop:
optirun nvidia-docker build -t tensorflow-improved-gpu -f Dockerfile.GPU .

