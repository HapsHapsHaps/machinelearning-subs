package dk.hapshapshaps.machinelearning;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URL;
import java.util.concurrent.atomic.AtomicBoolean;

public class DependencyHandler {
    private final static Logger log = LoggerFactory.getLogger(DependencyHandler.class);
    private static final String DEPENDENCY_01_FILENAME = "libtensorflow_framework.so";
    private static final String DEPENDENCY_02_FILENAME = "libtensorflow_jni.so";

    private static AtomicBoolean loaded = new AtomicBoolean(false);

    /**
     * If available and not already loaded, it loads the custom compiled TensorFlow machine code libraries.
     */
    public static void loadDependencies() {
        if( ! isLoaded()) {
            if(dependenciesExistsInResources()) {
                loadTensorFlowDependency();
            }
            DependencyHandler.loaded.set(true);
        }
    }

    private static void loadTensorFlowDependency() {

        URL resource = DependencyHandler.class.getClassLoader().getResource(DEPENDENCY_02_FILENAME);
        String path = resource.getFile();

        try {
            System.load(path);
            log.info("Using supplied custom compiled TensorFlow machine code library");
        } catch (UnsatisfiedLinkError e) {
            log.error("Failed to load custom TensorFlow machine code library. Falling back to official variants. \n" + e);
        }
    }

    private static boolean isLoaded() {
        return loaded.get();
    }

    private static boolean dependenciesExistsInResources() {
        boolean dependency01 = dependencyExistsInResources(DEPENDENCY_01_FILENAME);
        boolean dependency02 = dependencyExistsInResources(DEPENDENCY_02_FILENAME);
        return dependency01 && dependency02;
    }

    private static boolean dependencyExistsInResources(String dependencyName) {
        URL resource = DependencyHandler.class.getClassLoader().getResource(dependencyName);
        return resource != null;
    }
}
