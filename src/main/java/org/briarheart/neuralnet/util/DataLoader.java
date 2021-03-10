package org.briarheart.neuralnet.util;

import org.briarheart.neuralnet.util.resource.Resource;

import java.io.IOException;

/**
 * @author Roman Chigvintsev
 */
public interface DataLoader {
    double[][] load(Resource resource) throws IOException;
}
