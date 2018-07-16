# How to train a model for object-detection
This is a faily comprehensive run down on how to train a TensorFlow model for object detection in images.
This example focuses on training with two object types, but is very easy to use for even more types.

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
    1. The actual training
    2. Follow along with the training process (Optional)
    3. Process trained result
    4. Copying the trained results
- Time to try it out! - Use your trained model and see if your hard work has payed off..

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

#### If you prepared the dataset yourself (WIP)
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
python generate_tfrecord.py --csv_input=data/labels.csv --output_path=data/train.record
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

### 1. The actual training
To train with the dataset you run the following command inside the container, which easily can take mre than 30 minutes to complete:
```
python /tensorflow/tensorflow/models/research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/root/sharedfolder/data/ssd_mobilenet_v1_coco.config \
    --train_dir=/root/sharedfolder/training/
```
When a checkpoint is created in the training folder, that means it have reached an epoch, which is a complete run trough all the images.
For testing 3 or 5 epochs is more than enough. So just hit `Ctrl + c`, when you want to stop it.

This training process should take a toll on your machines hardware. If the process crashes or gives a ton of errors, you will have to decrease the batch_size in the config.

### 2. Follow along with the training process (Optional)
Using Tensorboard, this will give you a statistic representation of how the training process is going.

In my experience, it's best to to train on the GPU and perform the evaluation on the CPU. 
If you do both on the CPU, you will probably have to decrease your batch_size a little bit. 
Or the evaluation will cause issues for the training process.

Start a new terminal instance.

#### Same container
If you want to do both from the same container, find the container id in the list of active containers, by executing the following command:
```
docker -ps
```
Then enter it by executing the following command to attach a second terminal to the same docker container:
``` 
docker exec -it <container id> bash
```
Remember to replace `<container id>` with the actual container id.

#### Separate container (Recommended)  
Just as described under the section `Using the docker containers`, you launch a CPU container, but with the `6006` port mapped differently.
Else you wont be able to start it, as the other container have reserved that port.  
I've changed the port to `6007` in the command below and removed the first port as it's not needed:
```
docker run --rm -it -p 6007:6006 -v $HOME/andet/training/docker-training-shared/object-training:/root/sharedfolder:Z jacobpeddk/tensorflow-improved:latest
```

#### Evaluating
**Start TensorBoard**   
Then to start Tensorboard, execute the following command inside the container, which will continuously perform detection on some test files and give an overview of how it improves over time.
All depending on your training configuration in the config file:  
`tensorboard --logdir=/root/sharedfolder/training/`

It should start up within 3 seconds, and you can now with your webbrowser go to the following url:  
`http://localhost:6007`

**Start evaluation**  
Now you're ready to see the evaluation process.

So just like described under the `Same container` section above, attach yet another terminal to the docker container that Tensorboard runs from as follows:
``` 
docker exec -it <container id> bash
```
Remember to replace `<container id>` with the actual container id.

And now execute the following command, which will start the evaluation:
```
python /tensorflow/tensorflow/models/research/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=/root/sharedfolder/data/ssd_mobilenet_v1_coco.config \
    --checkpoint_dir=/root/sharedfolder/containerOutput/training/ \
    --eval_dir=/root/sharedfolder/training/eval/
```

### 3. Process trained result
The training process should have generated a bunch of snapshots during its training process. 
These snapshots can't be used directly.
To get the actual trained model, that we can use, these files will first have to be processed.  
So lets go ahead and process them so we can get the final trained model, which is what we want and need for actually testing and using the trained model.

In the command below, notice the part `model.ckpt-12527`.
This refers to the latest performed snapshot.
So replace this number with the one from the snapshot in your training directory that have the highest number.  
Then with your modification for the command below, execute it in the terminal inside the first container where you performed the training process.
```
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path=/root/sharedfolder/data/ssd_mobilenet_v1_coco.config \
    --trained_checkpoint_prefix=/root/sharedfolder/training/model.ckpt-12527 \
    --output_directory=/root/sharedfolder/training/result_graph_model
```

That will give you a bunch of new files in a folder inside the training directory.

### 4. Copying the trained results
Now, for the last step, lets copy the resulting trained model into the folder named `trainingResults`, so it's nice and simple to figure out.
Inside the container or from your host machine, execute the following command:
```
cp training/result_graph_model/frozen_inference_graph.pb trainingResults/
```

And that's it! You're done with the training process, and have the resulting file now neatly placed inside the directory `trainingResults`.

## Time to try it out!
Time to use your trained model and see if your hard work has payed off..

The entire purpose of this project were to make object detection easy with java.
As such, there's a java application built to work with the trained model created using this guide.  
So make sure you have Java installed and lets go!

I expect you to have cloned the repository to your local machine.
If not, well, what are you waiting for?! Get it done!

Open the java program with your favorite IDE. 
Inside the folder `src/test/resources/objectDetection`, you will find a readme informing you what files will be needed.  
Small hint. 
One file now resides in the `data` directory.
The other in the `trainingResults` directory.

Also remember to add a test image to the folder `src/test/resources/random-test-images`.

Then under `src/test/java` there's a test file called `ManualTests`. Modify it as needed and drop a debug stop point at the bottom of the method named `runObjectDetection()`, launch it with your debugger, and check out the results.

**That's all there is to it.**  
**Enjoy!**

## The end.
