package org.briarheart.neuralnet.util;

import com.google.common.base.Preconditions;

/**
 * @author Roman Chigvintsev
 */
public class MinMaxDataNormalizer implements DataNormalizer {
    @Override
    public double[][] normalize(double[][] data) {
        Preconditions.checkNotNull(data, "Data to be normalized must not be null");

        int rows = data.length;
        int cols = data[0].length;
        double[][] normalizedData = new double[rows][cols];
        for (int col = 0; col < cols; col++) {
            double maxValue = findMaxColumnValue(data, col);
            for (int row = 0; row < rows; row++) {
                normalizedData[row][col] = data[row][col] / Math.abs(maxValue);
            }
        }
        return normalizedData;
    }

    @Override
    public double[][] denormalize(double[][] data, double[][] normalizedData) {
        Preconditions.checkNotNull(normalizedData, "Data must not be null");
        Preconditions.checkNotNull(normalizedData, "Data to be denormalized must not be null");

        int rows = normalizedData.length;
        int cols = normalizedData[0].length;
        double[][] denormalizedData = new double[rows][cols];
        for (int col = 0; col < cols; col++) {
            double maxValue = findMaxColumnValue(data, col);
            for (int row = 0; row < rows; row++) {
                denormalizedData[row][col] = normalizedData[row][col] * Math.abs(maxValue);
            }
        }
        return denormalizedData;
    }

    private double findMaxColumnValue(double[][] data, int col) {
        double maxValue = data[0][col];
        for (int i = 1; i < data.length; i++) {
            double[] row = data[i];
            if (Double.compare(row[col], maxValue) > 0) {
                maxValue = row[col];
            }
        }
        return maxValue;
    }
}
