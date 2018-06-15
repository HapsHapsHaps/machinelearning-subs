package dk.hapshapshaps.machinelearning.objectdetection;

import dk.hapshapshaps.machinelearning.objectdetection.models.ObjectRecognition;

import java.awt.image.BufferedImage;
import java.util.ArrayList;

public interface ObjectDetector extends AutoCloseable {

    /**
     * Returns all detected objects on the input image, that the trained model can find and recognize.
     * @param image The image to perform the objectdetection on.
     * @return All found detections in the input image.
     */
    ArrayList<ObjectRecognition> classifyImage(BufferedImage image);
}
