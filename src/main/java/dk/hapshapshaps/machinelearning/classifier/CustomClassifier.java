package dk.hapshapshaps.machinelearning.classifier;

import dk.hapshapshaps.machinelearning.DependencyHandler;
import dk.hapshapshaps.machinelearning.classifier.models.ClassifyRecognition;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import com.sun.imageio.plugins.png.PNGMetadata;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.metadata.IIOInvalidTreeException;
import javax.imageio.metadata.IIOMetadata;
import javax.imageio.metadata.IIOMetadataNode;
import javax.imageio.stream.ImageOutputStream;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class CustomClassifier implements Classifier {

    private final Graph graph;
    private final List<String> labels;

    public CustomClassifier(File graphFile, File labelFile) throws IOException {
        DependencyHandler.loadDependencies();

        byte[] graphBytes = Files.readAllBytes(graphFile.toPath());
        this.graph = loadGraph(graphBytes);
        this.labels = Files.readAllLines(labelFile.toPath());
    }

    /**
     * Find out what known classification the image most likely resembles
     * @param image the image to classify according to model.
     * @return highest rated classification.
     * @throws IOException error preparing image for classification.
     */
    @Override
    public ClassifyRecognition classifyImage(BufferedImage image) throws IOException {
        Tensor<Float> imageTensor = normalizeImage(image);

        float[] graphResults = executeGraph(imageTensor);

        int labelIndex = bestProbabilityIndex(graphResults);

        String label = labels.get(labelIndex);

        float confidence = graphResults[labelIndex];

        ClassifyRecognition recognition = new ClassifyRecognition(labelIndex, label, confidence);

        imageTensor.close();

        return recognition;
    }

    private Tensor<Float> normalizeImage(BufferedImage image) throws IOException {

//        byte[] imageBytes = new byte[0];
//        try {
//            imageBytes = Files.readAllBytes(image.toPath());
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//        byte[] imageBytes = ((DataBufferByte) image.getData().getDataBuffer()).getData();

        byte[] imageBytes = imageToTypedByteArray(image);

        try (Graph graph = new Graph()) {
            GraphBuilder builder = new GraphBuilder(graph);

            // The image will be resized to fit trained model with the following specifications.
            final int Height = 299;
            final int Width = 299;
            final float mean = 0f;
            final float scale = 255f;

            final Output<String> input = builder.constant("input", imageBytes);

            final Output<Float> resizedImage = builder.resizeBilinear(
                    builder.expandDims(
                            builder.cast(builder.decodeJpeg(input, 3), Float.class),
                            builder.constant("make_batch", 0)),
                    builder.constant("size", new int[] {Height, Width}));

            final Output<Float> output =
                    builder.div(
                            builder.sub(resizedImage, builder.constant("mean", mean)),
                            builder.constant("scale", scale));
            try (Session session = new Session(graph)) {
                return session.runner().fetch(output.op().name()).run().get(0).expect(Float.class); // casts Tensor<?> to Tensor<Float>.
            }
        }
    }

    /**
     * Executes graph on the given preprocessed image
     * @param image preprocessed image
     * @return output tensor returned by tensorFlow
     */
    private float[] executeGraph(final Tensor<Float> image) {
        try (Session s = new Session(graph);
             Tensor<Float> result =
                     s.runner()
                             .feed("Mul", image)
                             .fetch("final_result")
                             .run()
                             .get(0)
                             .expect(Float.class)) {
            final long[] rshape = result.shape();
            if (result.numDimensions() != 2 || rshape[0] != 1) {
                throw new RuntimeException(
                        String.format(
                                "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                Arrays.toString(rshape)));
            }
            int nlabels = (int) rshape[1];
            return result.copyTo(new float[1][nlabels])[0];
        }
    }

    /**
     * Converts BufferedImage into a byte array with image type meta data included.
     * @param image
     * @return
     * @throws IOException a cache file might have been needed, adding meta data failed, og something different.
     */
    private byte[] imageToTypedByteArray(BufferedImage image) throws IOException {
        PNGMetadata metadata = new PNGMetadata();
        setDPI(metadata); // Defines DPI information for the PNG image format.

        String formatName = "png";

        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();

        for (Iterator< ImageWriter > iw = ImageIO.getImageWritersByFormatName(formatName); iw.hasNext(); ) {
            ImageWriter writer = iw.next();

            try (final ImageOutputStream stream = ImageIO.createImageOutputStream(byteArrayOutputStream)) {
                writer.setOutput(stream);
                ImageWriteParam writeParam = writer.getDefaultWriteParam();
                writer.write(metadata, new IIOImage(image, null, metadata), writeParam);
            } finally {
                writer.dispose();
            }
        }

        byte[] byteArray = byteArrayOutputStream.toByteArray();

        byteArrayOutputStream.close();

        return byteArray;
    }

    /**
     * Defines the dpi values for the PNG file type.
     * @param metadata the instance to add the PNG DPI data to.
     * @throws IIOInvalidTreeException if the tree cannot be parsed successfully using the rules of the given format.
     */
    private static void setDPI(IIOMetadata metadata) throws IIOInvalidTreeException {

        // for PNG, it's dots per millimeter

        double dotsPerMilli = 1.0 * 300 / 10 / 2.541f;
        IIOMetadataNode horiz = new IIOMetadataNode("HorizontalPixelSize");
        horiz.setAttribute("value", Double.toString(dotsPerMilli));

        IIOMetadataNode vert = new IIOMetadataNode("VerticalPixelSize");
        vert.setAttribute("value", Double.toString(dotsPerMilli));

        IIOMetadataNode dim = new IIOMetadataNode("Dimension");
        dim.appendChild(horiz);
        dim.appendChild(vert);

        IIOMetadataNode root = new IIOMetadataNode("javax_imageio_1.0");
        root.appendChild(dim);

        metadata.mergeTree("javax_imageio_1.0", root);
    }

    private Graph loadGraph(byte[] graphBytes) {
        Graph graph = new Graph();
        graph.importGraphDef(graphBytes);
        return graph;
    }

    private static int bestProbabilityIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }

    @Override
    public void close() throws Exception {
        this.graph.close();
    }
}
