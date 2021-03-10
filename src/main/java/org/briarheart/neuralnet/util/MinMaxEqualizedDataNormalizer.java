package org.briarheart.neuralnet.util;

import com.google.common.base.Preconditions;

/**
 * @author Roman Chigvintsev
 */
public class MinMaxEqualizedDataNormalizer implements DataNormalizer {
    @Override
    public double[][] normalize(double[][] data) {
        Preconditions.checkNotNull(data, "Data to be normalized must not be null");

        int rows = data.length;
        int cols = data[0].length;
        double[][] normalizedData = new double[rows][cols];

        for (int col = 0; col < cols; col++) {
            double[] minAndMax = findMinAndMaxColumnValues(data, col);
            double minValue = minAndMax[0];
            double maxValue = minAndMax[1];
            for (int row = 0; row < rows; row++) {
                normalizedData[row][col] = (data[row][col] - minValue) / (maxValue - minValue);
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
            double[] minAndMax = findMinAndMaxColumnValues(data, col);
            double minValue = minAndMax[0];
            double maxValue = minAndMax[1];
            for (int row = 0; row < rows; row++) {
                denormalizedData[row][col] = normalizedData[row][col] * (maxValue - minValue) + minValue;
            }
        }
        return denormalizedData;
    }

    private double[] findMinAndMaxColumnValues(double[][] data, int col) {
        double minValue = data[0][col];
        double maxValue = data[0][col];

        for (int i = 1; i < data.length; i++) {
            double[] row = data[i];
            if (Double.compare(row[col], minValue) < 0) {
                minValue = row[col];
            }
            if (Double.compare(row[col], maxValue) > 0) {
                maxValue = row[col];
            }
        }

        return new double[]{minValue, maxValue};
    }
}
