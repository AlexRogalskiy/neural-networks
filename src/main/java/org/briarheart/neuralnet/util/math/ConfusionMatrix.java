package org.briarheart.neuralnet.util.math;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

/**
 * @author Roman Chigvintsev
 */
public class ConfusionMatrix extends Matrix {
    public ConfusionMatrix(double[][] data, double marginError) {
        super(2, 2);
        checkNotNull(data, "Data must not be null");

        int truePositive = 0, falsePositive = 0, trueNegative = 0, falseNegative = 0;
        Size size = getSize();
        for (double[] row : data) {
            checkArgument(row.length == size.getY(), "Length of data row must be equal to " + size.getY());
            double[] buf = new double[row.length];
            for (int col = 0; col < row.length; col++) {
                buf[col] = row[col] <= marginError ? -1.0 : 1.0;
            }
            if (buf[0] == 1.0 && buf[1] == 1.0) {
                truePositive++;
            } else if (buf[0] == -1.0 && buf[1] == -1.0) {
                trueNegative++;
            } else if (buf[0] == -1.0 && buf[1] == 1.0) {
                falsePositive++;
            } else if (buf[0] == 1.0 && buf[1] == -1.0) {
                falseNegative++;
            }
        }

        set(0, 0, truePositive);
        set(0, 1, falsePositive);
        set(1, 0, falseNegative);
        set(1, 1, trueNegative);
    }

    public double getSensitivity() {
        double truePositive = get(0, 0);
        double falseNegative = get(1, 0);
        return truePositive / (truePositive + falseNegative);
    }

    public double getSpecificity() {
        double trueNegative = get(1, 1);
        double falsePositive = get(0, 1);
        return trueNegative / (falsePositive + trueNegative);
    }

    public double getAccuracy() {
        double truePositive = get(0, 0);
        double trueNegative = get(1, 1);
        double falseNegative = get(1, 0);
        double falsePositive = get(0, 1);
        return (truePositive + trueNegative) / (truePositive + falseNegative + falsePositive + trueNegative);
    }
}
