package org.briarheart.neuralnet.activation;

import java.util.function.DoubleUnaryOperator;

/**
 * @author Roman Chigvintsev
 */
public interface ActivationFunction extends DoubleUnaryOperator {
    double apply(double value);

    DoubleUnaryOperator getDerivative();

    @Override
    default double applyAsDouble(double operand) {
        return apply(operand);
    }

    ActivationFunction LINEAR = new ActivationFunction() {
        @Override
        public double apply(double value) {
            return value;
        }

        @Override
        public DoubleUnaryOperator getDerivative() {
            return value -> 1.0;
        }
    };

    ActivationFunction HARD_LIMITING_THRESHOLD = new ActivationFunction() {
        @Override
        public double apply(double value) {
            return value >= 0.0 ? 1.0 : 0.0;
        }

        @Override
        public DoubleUnaryOperator getDerivative() {
            return null;
        }
    };


    ActivationFunction HYPERBOLIC_TANGENT = new ActivationFunction() {
        @Override
        public double apply(double value) {
            return Math.tanh(value);
        }

        @Override
        public DoubleUnaryOperator getDerivative() {
            return value -> 1.0 / Math.pow(Math.cosh(value), 2.0);
        }
    };

    ActivationFunction SIGMOID = new ActivationFunction() {
        @Override
        public double apply(double value) {
            return 1.0 / (1.0 + Math.exp(-value));
        }

        @Override
        public DoubleUnaryOperator getDerivative() {
            return value -> value * (1.0 - value);
        }
    };
}
