# Training models for object-detection and classification

## [Object Detection](training-object-detection.md)
Training a model for object-detection is quite a bit more convoluted than just classification.
But the advantage is knowing exactly where and what is in the image.

The process for training a model for object-detection is described in the file:  
[training-object-detection](training-object-detection.md)

## Classification (WIP)
Todo..

I made a fairly good guide a while ago [in another repository.](https://github.com/HapsHapsHaps/Training-examples/tree/master/Classification/flower-image-classifier)

Just one one of these Docker containers instead:

**Using CPU**  
```
docker run --rm -it -p 8001:8888 -p 6006:6006 -v $HOME/andet/training/docker-training-shared/object-training:/root/sharedfolder:Z jacobpeddk/tensorflow-improved:latest
```

**Using GPU**  
```
nvidia-docker run --rm -it -p 8001:8888 -p 6006:6006 -v $HOME/andet/training/docker-training-shared/object-training:/root/sharedfolder:Z jacobpeddk/tensorflow-improved:latest-gpu
```
