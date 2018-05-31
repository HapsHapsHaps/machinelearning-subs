
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

    final static String modelsDir = "objectDetection";
    final static String graphName = "frozen_inference_graph.pb";
    final static String labelsName = "object-detection.pbtxt";

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


        Path modelPath = Paths.get(modelsDir, graphName);
        URL modelResource = Thread.currentThread().getContextClassLoader().getResource(modelPath.toString());
        File modelFile = new File(modelResource.getFile());

        Path labelsPath = Paths.get(modelsDir, labelsName);
        URL labelsResource = Thread.currentThread().getContextClassLoader().getResource(labelsPath.toString());
        File labelFile = new File(labelsResource.getFile());

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
        File modelFile = new File("/home/jacob/andet/training/docker-training-shared/classification/subs/trained-files/output_graph.pb");
        File labelFile = new File("/home/jacob/andet/training/docker-training-shared/classification/subs/trained-files/output_labels.txt");

//        BufferedImage image = ImageIO.read(imageFile01);

        CustomClassifier classifier = new CustomClassifier(modelFile, labelFile);

        ClassifyRecognition recognition = classifier.classifyImage(imageFile01);

        String s = "";
    }
}
