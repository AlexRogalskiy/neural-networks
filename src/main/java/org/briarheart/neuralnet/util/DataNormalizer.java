package org.briarheart.neuralnet.util;

/**
 * @author Roman Chigvintsev
 */
public interface DataNormalizer {
    double[][] normalize(double[][] data);

    double[][] denormalize(double[][] data, double[][] normalizedData);
}
