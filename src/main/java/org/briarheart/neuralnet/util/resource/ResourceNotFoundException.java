package org.briarheart.neuralnet.util.resource;

import java.io.IOException;

/**
 * @author Roman Chigvintsev
 */
public class ResourceNotFoundException extends IOException {
    public ResourceNotFoundException(Resource resource) {
        super(resource.toString());
    }
}
