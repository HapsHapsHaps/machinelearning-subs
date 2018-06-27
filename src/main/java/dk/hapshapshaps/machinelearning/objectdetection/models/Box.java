package dk.hapshapshaps.machinelearning.objectdetection.models;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.List;

public class Box {
    public final int x;
    public final int y;
    public final int width;
    public final int height;

    public Box(int x, int y, int width, int height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    public static BufferedImage drawBoxes(BufferedImage image, List<Box> boxes) {
        Graphics2D graph = image.createGraphics();
        graph.setColor(Color.green);

        for (Box box : boxes) {
            graph.drawRect(box.x, box.y, box.width, box.height);
        }

        graph.dispose();
        return image;
    }

    public static BufferedImage drawBox(BufferedImage image, int x, int y, int width, int height) {
        Graphics2D graph = image.createGraphics();
        graph.setColor(Color.GREEN);
//        graph.fill(new Rectangle(x, y, width, height));
        graph.drawRect(x, y, width, height);
//        graph.setStroke();
        graph.dispose();
        return image;
    }
}
