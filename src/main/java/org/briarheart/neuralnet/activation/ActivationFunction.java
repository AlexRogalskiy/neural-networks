package org.briarheart.neuralnet.activation;

import org.briarheart.neuralnet.util.DoubleToDoubleFunction;

/**
 * @author Roman Chigvintsev
 */
public interface ActivationFunction extends DoubleToDoubleFunction {
    DoubleToDoubleFunction getDerivative();

    ActivationFunction LINEAR = new ActivationFunction() {
        @Override
        public double apply(double value) {
            return value;
        }

        @Override
        public DoubleToDoubleFunction getDerivative() {
            return value -> 1.0;
        }
    };

    ActivationFunction HARD_LIMITING_THRESHOLD = new ActivationFunction() {
        @Override
        public double apply(double value) {
            return value >= 0.0 ? 1.0 : 0.0;
        }

        @Override
        public DoubleToDoubleFunction getDerivative() {
            return null;
        }
    };


    ActivationFunction HYPERBOLIC_TANGENT = new ActivationFunction() {
        @Override
        public double apply(double value) {
            return Math.tanh(value);
        }

        @Override
        public DoubleToDoubleFunction getDerivative() {
            return value -> 1.0 / Math.pow(Math.cosh(value), 2.0);
        }
    };

    ActivationFunction SIGMOID = new ActivationFunction() {
        @Override
        public double apply(double value) {
            return 1.0 / (1.0 + Math.exp(-value));
        }

        @Override
        public DoubleToDoubleFunction getDerivative() {
            return value -> value * (1.0 - value);
        }
    };
}
