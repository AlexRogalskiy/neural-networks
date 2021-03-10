package org.briarheart.neuralnet.util;

/**
 * @author Roman Chigvintsev
 */
@FunctionalInterface
public interface DoubleToDoubleFunction {
    double apply(double value);
}
