# How to train a model for object-detection
This is a quick run down on how to train a TensorFlow model for object classifiaction with images.
This example focuses on training with two object types.

This project requires docker to be installed, and using the specifically customized containers.  
You can read more about this in [the readme file in the docker folder.](../docker/README.md)

These are the different topics that will be covered
- Prepare
    - Working directory for storing the data
    - Required files
    - Using the docker containers
- Setup
    1. Preparing training data
    2. Process annotations
    3. Prepare configs
    4. Checkup
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


## Prepare
Before you can begin setting up for the training process. There's a few things that need to be understood first.

### Working directory for storing the data
This is the folder that you attach to you docker container, for which everything that needs to be kept will reside within.
It's the workdir, where everything will happen within.

The workdir path for me is: `$HOME/andet/training/docker-training-shared/object-training`  
This is the path, that will be the working directory for this tutorial.
Remember to adapt it to your setup.

Inside the docker container, the `object-training` folder will be at the following path:  
`$HOME/sharedfolder`

And on the host, the same `object-training` folder will be at the following path:  
`$HOME/andet/training/docker-training-shared/object-training`

In this folder, there is a few other folders that should be created.  
These are: `data`, `images`, `training`, `trainingResult` and `content`.

The `content` folder is specific for this guide, and is where the original images and label files can be placed. You can easily have this folder elsewhere on your system if you want that.

So a nice little tree-view of the folder structure:
- object-training
    - data
    - images
    - training
    - trainingResult
    - content

From now on, this guide will assume that you have created the working directory and have a terminal open at its path.  
Remember, the docker attached folder and thereby also the working directory, will be accessible from inside the docker container.  

### Required files
In the repository where this guide resides, there's a folder named `object-training` with some files inside it.
These are required for this guide.
So please copy its content into your workdir, or nothing will work.

With this added, there should now be two python scripts in your workdir, as-well as two files and a zip-file in the `data` folder.  
The zip file also needs to be extracted, so there will be a `ssd_mobilenet_v1_coco` folder inside the `data` folder.

### Using the docker containers
As described earlier, this guide requires the use of the specialized docker containers.  
With that, there's therefor also many of the steps below that requires you be executing the commands from within the container.

So to launch and use one of the containers, execute one of the two long commands below in a new terminal session. But remember to adapt the path to the working directory to where yours is.

**Using CPU**  
```
docker run --rm -it -p 8001:8888 -p 6006:6006 -v $HOME/andet/training/docker-training-shared/object-training:/root/sharedfolder:Z jacobpeddk/tensorflow-improved:latest
```

**Using GPU**  
```
nvidia-docker run --rm -it -p 8001:8888 -p 6006:6006 -v $HOME/andet/training/docker-training-shared/object-training:/root/sharedfolder:Z jacobpeddk/tensorflow-improved:latest-gpu
```

That should launch the container with your terminal ready inside the working directory, within the container itself.
So now you're ready to continue with the guide.

## Setup

### 1. Preparing training data (images and object annotations)
This guide will assume that you already have prepared a dataset to train upon, as described in the guide: [label-dataset](label-dataset.md).

If not, I'll quickly show you how to get a hold of one to test with. (WIP - NOT YET DONE)

#### If you prepared the dataset yourself
If you prepared it yourself in accordance with [the related guide](label-dataset.md), it should consist of a folder called images and another called annotations.  
So lets move on to the next part: 2. Process annotations.

#### If you want to download a dataset  (WIP)
Luckily there's predefined and prepared dataset available, that have everything we need to test this training process with.  
Everything related to this process is described in the file: [object-detection-premade-dataset](object-detection-premade-dataset.md)

### 2. Process annotations

#### Step 1. Create CSV file
With all the images and their annotation files prepared, we will now have to convert the annotations into a csv-file.
So copy all the images and their related annotation files into the folder called `images`.

Then within the container, execute the following command to process the annotations to a csv:  
```
xml_to_csv.py
```

That should give you a new file inside the `data` directory called `labels.csv`.  
Quickly open this file and make sure it actually contains a bunch of entries. If something went wrong. It'll be empty.

#### Step 2. Before TFRecord file
With the `labels.csv` file now created, we can convert the labelled annotations into the proper format that the training scripts can understand.

This will be done with the script called `generate_tfrecord.py`, but it must be edited first!  
So open the file in your favorite text editor.
At line 30 and below, there is two row labels defined:
```
def class_text_to_int(row_label):
    if row_label == 'sub':
        return 1
    elif row_label == 'nosub':
        return 2
    else:
        None
```
Replace these with the labels that you have defined for your dataset in the file `object-detection.pbtxt`.
If you need more labels, just add another set of:
```
elif row_label == 'thirdlabel':
        return 3
```
And now you're ready to create the final annotations file.
#### Step 3. Create TFRecord file
To create the final annotations file, all that will have to be done, is to execute the following line from within the container:  
```
python generate_tfrecord.py --csv_input=data/labels.csv  --output_path=data/train.record
```
That wil create the resulting TFRecord `train.record` file inside the `data`, which we needed to begin the actual training process.

### 3. Prepare configs
The last step in the setup process, is defining the right values in the training configuration file.  
This guide is based on the `ssd_mobilenet_v1_coco` model and config. Others might work as-well, but this is the one that is known to work with the created java program.

So, in the `data` folder, open the file `ssd_mobilenet_v1_coco.config` in your favorite text editor, and adapt the following lines as needed:
- Line 9, `num_classes: 2` shall be changed to match the number of types you have defined in the file `object-detection.pbtxt`.
- Line 143, `batch_size: 15` will have to be fine tuned depending on the hardware and available memory in your machine.
    - I used the batch size 15, on my Desktop GPU which is a GTX 1080TI with 11GB of video memory.
- Line 168, `num_steps: 500` completely depends on the amount of images that your dataset contains. Read the text above it for an explanation.

You can also modify the `eval_config` section. But that's only if you want to evaluate the process.  
Be aware that evaluating the training process, is separate from the training process, and have no direction relations between them.

### 4. Checkup
With all the steps done, lets make sure everything is ready.

- The `data` folder should at a minimum container the following files and folders:
    - file: `object-detection.pbtxt`
    - file: `train.record`
    - file: `ssd_mobilenet_v1_coco.config`
    - folder: `ssd_mobilenet_v1_coco`
- The `images` folder should at a minimum contain all the images, that had an annotation file.
- All images must be of the type jpg.
- The following folders also exists in the workdir: (easiest if they're empty as-well)
    - `training`
    - `trainingResults`
- You've modified the `ssd_mobilenet_v1_coco.config` file to fit your training data and hardware.

## Training with the dataset
Phew, that was a lot of stuff to figure out and setup. 
But now you've come to the part which i suppose is what you actually wanted to try to begin with. 

So lets go!

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