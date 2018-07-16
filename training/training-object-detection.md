# How to train a model for object-detection
This is a quick run down on how to train a TensorFlow model for object classifiaction with images.
This example focuses on training with two object types.

This project requires docker to be installed, and using the specifically customized containers.  
You can read more about this in [the readme file in the docker folder.](../docker/README.md)

These are the different topics that will be covered
- Setup
    1. Preparing training data
    2. Process annotations
    3. Prepare configs
    4. Download the TensorFlow research files
    5. Setup docker container
- Training with the dataset
    1. Run preparation script
    2. The actual training
    3. Follow along with the training process (Optional)
    4. Process trained result
    5. Copying the trained results
- Extras
    1. Commands from within the docker container
    2. Training with GPU
    3. Evaluate training process with live testing




## Setup

### Working directory for storing the data
This is the folder that you attach to you docker container, for which everything that needs to be kept will reside within.
It's the workdir, where everything will happen within.

The workdir path for me is: `$HOME/andet/training/docker-training-shared/object-training`  
This is the folder, that will be the working directory for this tutorial.

Inside the docker container, the `object-training` folder will reside at:  
`$HOME/sharedfolder`

And on the host, the same folder will reside at:  
`$HOME/andet/training/docker-training-shared/object-training`

In this folder, there is a few other folders that should be created.  
These are: `data`, `images`, `training`, `trainingReult` and `content`.

The `content` folder is specific for this guide, and is where the original images and label files can be placed. You can easily have this folder elsewhere on your system if you want that.

So a nice little tree-view of the folder structure:
- object-training
    - data
    - images
    - training
    - trainingReult
    - content

From now on, this guide will assume that you have created the working directory and have a terminal open at its path.  
Remember, the docker attached folder and thereby also the working directory, will be accessible from inside the docker container.  

### Using the docker containers
As described earlier, this guide requires the use of the specialized docker containers.  
With that, there's therefor also many of the steps below that requires you be executing the commands from within the container.

So to launch and use one of the containers, execute one of the two long commands below. But remember to adapt the path to the working directory to where yours is.

**Using CPU**  
`docker run --rm -it -p 8001:8888 -p 6006:6006 -v $HOME/andet/training/docker-training-shared/object-training:/root/sharedfolder:Z jacobpeddk/tensorflow-improved:latest`

**Using GPU**  
`nvidia-docker run --rm -it -p 8001:8888 -p 6006:6006 -v $HOME/andet/training/docker-training-shared/object-training:/root/sharedfolder:Z jacobpeddk/tensorflow-improved:latest-gpu`



### 1. Preparing training data (images and object annotations)
This guide will assume that you already have prepared a dataset to train upon, as described in the guide: [label-dataset](label-dataset.md).

If not, I'll quickly show you how to get a hold of one to test with. (WIP - NOT YET DONE)

#### If you prepared the dataset yourself
If you prepared it yourself in accordance with [the related guide](label-dataset.md), it should consist of a folder called images and another called annotations.  
So lets move on to the next part: 2. Process annotations.

#### If you want to download a dataset  (WIP)
Luckily there's predefined and prepared dateset available, that have everything we need to test this training process with.  
Everything related to this process is described in the file: [object-detection-premade-dataset](object-detection-premade-dataset.md)

### 2. Process annotations

#### Enter 

#### Step 1. CSV file
With all the images and their annotation files prepared, we will now have to convert the annotations into a csv-file.
So copy all the images and their related annotation files into the folder called `images`.

Then run the `xml_to_csv.py` script, by executing it in your terminal.
  
That should give you a new file inside the `data` directory called `labels.csv`.  
Quickly open this file and make sure it actually contains a bunch of entries. If something went wrong. It'll be empty.

#### Step 2. TFRecord file
With the `labels.csv` file now created, we can convert the labelled annotations into the proper format that the training scripts can understand.



### 3. Prepare configs

### 4. Download the TensorFlow research files

### 5. Setup docker container

#### Copy files to workdir

`scripts` directory.

#### Start docker container (CPU)
This will start the cotnainer and attach your terminal to the container, so every command will be executed inside of it.  

So execute the following command:  
`docker run --rm -it -name tsCPU -p 8000:8888 -p 6005:6006 -v $HOME/andet/training/docker-training-shared:/root/sharedfolder:Z tensorflow/tensorflow:latest-devel`  

The `--rm` part of the command means it will be deleted the moment you exit the container. Makes it easier for rapid testing, as it will just create a new container the next time you run the command.  

The `:Z` part of the command, is to stop selinux on the host machine from blocking every single interraction with the attached folder from inside the container.  

With the docker container running, and your terminal attached to it. It's time to train with the dataset.

#### Run setup script
From inside the docker container, run the following command to execute the setup script. 
The purpose of this script is to perform all the changes needed, so the tensorflow docker image can be used for training object detection with classification of images.   
Without this, the training will simply not work.  

So execute the following command to run the script:
`$HOME/sharedfolder/scripts/setup_docker_object_detection.sh`

Now also inside the container, execute the following command:  
``export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim``

## Training with the dataset

### 1. Run preparation script


### 2. The actual training
To train with the dataset you run the following command inside the container, which easily can take 30 minutes to complete.  
`python train.py --logtostderr --pipeline_config_path=training/ssd_mobilenet_v1_coco.config --train_dir=training/`  

### 3. Follow along with the training process (Optional)
Using Tensorboard, this will give you a statistic representation of how the training process is going.

From another terminal instance, execute the following command to attach a second terminal to the docker container:  
`docker exec -it tsCPU bash`  

Then to start tensorboard, execute the following command, which will keep and eye on the training results and give an overview of how it improves over time:  
`tensorboard --logdir=/tensorflow/tensorflow/models/research/object_detection/training/`

It should start up within 3 seconds, and you can now with your webbrowser go to:  
`http://localhost:6005`

### 4. Process trained result
From the training training snapshots and more will have been generated. This is not the directly usefull result, as these files will first have to be processed.  
So lets go ahead and process them so we can get the final trained model, which is what we want and need for actually testing and using the trained model.

### 5. Copying the trained results
To keep the trained model saved.
 We will copy it from the container, into the attached sharedfolder so it will be kept and can be used to detect and classify objects in images.

## The final piece. Classifying an image
If everything until now has worked. 

## The end.