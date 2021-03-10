package org.briarheart.neuralnet.util.resource;

import lombok.NonNull;
import lombok.RequiredArgsConstructor;

import java.net.URL;

/**
 * @author Roman Chigvintsev
 */
@RequiredArgsConstructor
public class ClassPathResource implements Resource {
    @NonNull
    private final String path;

    @Override
    public URL getUrl() {
        ClassLoader classLoader = getClass().getClassLoader();
        return classLoader.getResource(path);
    }

    @Override
    public String toString() {
        return path;
    }
}
