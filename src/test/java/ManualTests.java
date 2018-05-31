
import dk.hapshapshaps.machinelearning.classifier.CustomClassifier;
import dk.hapshapshaps.machinelearning.classifier.models.ClassifyRecognition;
import dk.hapshapshaps.machinelearning.objectdetection.CustomObjectDetector;
import dk.hapshapshaps.machinelearning.objectdetection.ObjectDetector;
import dk.hapshapshaps.machinelearning.objectdetection.models.Box;
import dk.hapshapshaps.machinelearning.objectdetection.models.ObjectRecognition;
import dk.hapshapshaps.machinelearning.objectdetection.models.RectFloats;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class ManualTests {
    final static Logger log = LoggerFactory.getLogger(ManualTests.class);
    final static String TEST_IMAGE_DIR = "random-test-images";
    final static String TEST_IMAGE01 = "image-6.jpg";

    final static String OBJECT_DETECTION_DIR = "objectDetection";
    final static String CLASSIFICATION_DIR = "classification";

    static File imageFile01;

    @BeforeAll
    public static void setup() {
        Path path = Paths.get(TEST_IMAGE_DIR, TEST_IMAGE01);
        URL image01 = Thread.currentThread().getContextClassLoader().getResource(path.toString());
        imageFile01 = new File(image01.getFile());
    }

    @Test
    @Disabled("For manual debugging only.")
    public void runObjectDetection() throws IOException {

        File modelFile = getObjectDetectionModel();

        File labelFile = getObjectDetectionLabels();

        BufferedImage image = ImageIO.read(imageFile01);

        ObjectDetector objectDetector = new CustomObjectDetector(modelFile, labelFile);

        ArrayList<ObjectRecognition> objectRecognitions = objectDetector.classifyImage(image);

        List<Box> boxes = new ArrayList<>();
        for (ObjectRecognition objectRecognition : objectRecognitions) {
            if(objectRecognition.getConfidence() > 0.05f) {
                RectFloats location = objectRecognition.getLocation();
                int x = (int) location.getX();
                int y = (int) location.getY();
                int width = (int) location.getWidth() - x;
                int height = (int) location.getHeight() - y;

                boxes.add(new Box(x, y, width, height));
            }
        }

        BufferedImage boxedImage = Box.drawBoxes(image, boxes);

        String s = "";
    }

    @Test
    @Disabled("For manual debugging only.")
    public void runClassification() throws IOException {
        File modelFile = getClassificationModel();
        File labelFile = getClassificationLabels();

        BufferedImage image = ImageIO.read(imageFile01);

        CustomClassifier classifier = new CustomClassifier(modelFile, labelFile);

        ClassifyRecognition recognition = classifier.classifyImage(image);

        String s = "";
    }

    private static File getObjectDetectionModel() {
        final String graphName = "frozen_inference_graph.pb";
        return getFile(OBJECT_DETECTION_DIR, graphName);
    }

    private static File getObjectDetectionLabels() {
        final String labelsName = "object-detection.pbtxt";
        return getFile(OBJECT_DETECTION_DIR, labelsName);
    }

    private static File getClassificationModel() {
        final String graphName = "output_graph.pb";
        return getFile(CLASSIFICATION_DIR, graphName);
    }

    private static File getClassificationLabels() {
        final String labelsName = "output_labels.txt";
        return getFile(CLASSIFICATION_DIR, labelsName);
    }

    private static File getFile(String dir, String fileName) {
        Path path = Paths.get(dir, fileName);
        URL resource = Thread.currentThread().getContextClassLoader().getResource(path.toString());

        if(resource == null) {
            throw new RuntimeException("A needed TEST file haven't been added. The missing file is: " + path.toString());
        }

        return new File(resource.getFile());
    }
}
