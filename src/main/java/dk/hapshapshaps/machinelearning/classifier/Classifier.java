package dk.hapshapshaps.machinelearning.classifier;

import dk.hapshapshaps.machinelearning.classifier.models.ClassifyRecognition;

import java.awt.image.BufferedImage;
import java.io.IOException;

public interface Classifier extends AutoCloseable {

    /**
     * Find out what known classification the image most likely resembles
     * @param image the image to classify according to model.
     * @return highest rated classification.
     * @throws IOException error preparing image for classification.
     */
    ClassifyRecognition classifyImage(BufferedImage image) throws IOException;
}
