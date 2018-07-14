# Docker-TensorFlow-Object-Detection
TensorFlow Docker containers with Improvements and fixes for training Objekt Detection models


Prebuild containers can be retrieved from here: https://hub.docker.com/r/jacobpeddk/tensorflow-improved/

## Build

### CPU
docker build -t tensorflow-improved -f Dockerfile .

### GPU
nvidia-docker build -t tensorflow-improved-gpu -f Dockerfile.gpu .

#### Building on an Optimus laptop:
optirun nvidia-docker build -t tensorflow-improved-gpu -f Dockerfile.gpu .

