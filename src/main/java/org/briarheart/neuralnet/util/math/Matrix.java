package org.briarheart.neuralnet.util.math;

import com.google.common.base.Preconditions;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import org.briarheart.neuralnet.util.Arrays;

/**
 * @author Roman Chigvintsev
 */
public class Matrix {
    private final double[][] matrix;
    @Getter
    private final transient Size size;

    @Setter
    private Double determinant;

    public Matrix(int numberOfRows, int numberOfColumns) {
        this(new Size(numberOfRows, numberOfColumns));
    }

    public Matrix(Matrix a) {
        Preconditions.checkNotNull(a, "Matrix must not be null");
        this.matrix = Arrays.copyOf(a.matrix);
        this.size = a.size;
        this.determinant = a.determinant;
    }

    private Matrix(Size size) {
        this.matrix = new double[size.x][size.y];
        this.size = size;
    }

    public Matrix add(Matrix m) {
        return add(this, m);
    }

    public static Matrix add(Matrix a, Matrix b) {
        Preconditions.checkNotNull(a, "Matrix \"a\" must not be null");
        Preconditions.checkNotNull(b, "Matrix \"b\" must not be null");
        Preconditions.checkArgument(a.size.equals(b.size), "Matrices must have the same number of rows and columns");

        Matrix result = new Matrix(a.size);
        for (int i = 0; i < a.size.x; i++) {
            for (int j = 0; j < a.size.y; j++) {
                result.set(i, j, a.get(i, j) + b.get(i, j));
            }
        }
        return result;
    }

    public Matrix transpose() {
        return transpose(this);
    }

    public static Matrix transpose(Matrix a) {
        Preconditions.checkNotNull(a, "Matrix must not be null");

        Matrix result = new Matrix(a.size.y, a.size.x);
        for (int i = 0; i < a.size.x; i++) {
            for (int j = 0; j < a.size.y; j++) {
                result.set(j, i, a.get(i, j));
            }
        }
        return result;
    }

    public Matrix multiply(Matrix a) {
        return multiply(this, a);
    }

    public static Matrix multiply(Matrix a, Matrix b) {
        Preconditions.checkNotNull(a, "Matrix \"a\" must not be null");
        Preconditions.checkNotNull(b, "Matrix \"b\" must not be null");
        Preconditions.checkArgument(a.size.y == b.size.x,
                "Number of columns of the first matrix must match number of rows of the second matrix");

        Matrix result = new Matrix(a.size.x, b.size.y);
        for (int i = 0; i < a.size.x; i++) {
            for (int j = 0; j < b.size.y; j++) {
                double value = 0;
                for (int k = 0; k < b.size.x; k++) {
                    value += a.get(i, k) * b.get(k, j);
                }
                result.set(i, j, value);
            }
        }
        return result;
    }

    public Matrix multiplyScalar(double v) {
        return multiplyScalar(this, v);
    }

    public static Matrix multiplyScalar(Matrix a, double v) {
        Preconditions.checkNotNull(a, "Matrix must not be null");

        Matrix result = new Matrix(a.size);
        for (int i = 0; i < a.size.x; i++) {
            for (int j = 0; j < a.size.y; j++) {
                result.set(i, j, a.get(i, j) * v);
            }
        }
        return result;
    }

    public double multiplyDiagonal() {
        double result = 1;
        for (int i = 0; i < size.y; i++) {
            result *= get(i, i);
        }
        return result;
    }

    public Matrix[] getLuDecomposition() {
        Matrix[] result = new Matrix[3];
        Matrix lu = new Matrix(this);
        Matrix l = new Matrix(lu.size);
        Matrix pSign = new Matrix(new IdentityMatrix(lu.size.y));
        int permutations = 0;
        l.set(0, 0, 1.0);
        for (int i = 1; i < lu.size.x; i++) {
            l.set(i, i, 1.0);
            for (int j = 0; j < i; j++) {
                double multiplier = -lu.get(i, j) / lu.get(j, j);
                lu.sumRowByRow(i, j, multiplier);
                if (j == i - 1) {
                    double value = lu.get(i, i);
                    if (value == 0) {
                        int rowIndex = lu.findRowWithNonZeroValue(i, i + 1, size.x - 1);
                        lu.permutateRow(i, rowIndex);
                        permutations++;
                    }
                }
                l.set(i, j, -multiplier);
            }
        }
        Matrix u = new Matrix(lu);
        if (permutations % 2 == 1) {
            pSign.set(0, 0, -1);
        }
        result[0] = l;
        result[1] = u;
        result[2] = pSign;
        return result;
    }

    public void sumRowByRow(int row, int rowSum, double multiplier) {
        Preconditions.checkArgument(row < size.x && rowSum < size.x, "Row index is out of bounds");
        for (int i = 0; i < size.y; i++) {
            set(row, i, get(row, i) + get(rowSum, i) * multiplier);
        }
    }

    public double getDeterminant() {
        return getDeterminant(this);
    }

    public static double getDeterminant(Matrix m) {
        Preconditions.checkNotNull(m, "Matrix must not be null");
        Preconditions.checkArgument(m.size.x == m.size.y, "Only square matrices can have determinant");

        if (m.determinant != null) {
            return m.determinant;
        }
        if (m.size.y == 1) {
            return m.get(0, 0);
        }
        if (m.size.y == 2) {
            return (m.get(0, 0) * m.get(1, 1)) - (m.get(1, 0) * m.get(0, 1));
        }
        Matrix[] lu = m.getLuDecomposition();
        return lu[1].multiply(lu[2]).multiplyDiagonal();
    }

    public Matrix inverse() {
        return inverse(this);
    }

    public static Matrix inverse(Matrix m) {
        Preconditions.checkNotNull(m, "Matrix must not be null");
        Preconditions.checkArgument(m.getDeterminant() != 0, "Matrix is not inversible");

        Matrix[] lu = m.getLuDecomposition();
        Matrix l = lu[0];
        Matrix u = lu[1];

        Matrix z = new Matrix(m.size);
        Matrix result = new Matrix(m.size);

        for (int j = 0; j < m.size.x; j++) {
            for (int i = 0; i < m.size.x; i++) {
                double value = 0.0;
                if (i == j) {
                    value = 1.0;
                }
                for (int k = j; k < i; k++) {
                    value -= l.get(i, k) * z.get(k, j);
                }
                z.set(i, j, value);
            }
        }

        for (int j = 0; j < m.size.x; j++) {
            for (int i = m.size.x - 1; i >= 0; i--) {
                double value = z.get(i, j);
                for (int k = i + 1; k < m.size.x; k++) {
                    value -= u.get(i, k) * result.get(k, j);
                }
                value /= u.get(i, i);
                result.set(i, j, value);
            }
        }

        return result;
    }

    public double get(int row, int col) {
        Preconditions.checkArgument(row < size.x, "Row index is out of bounds");
        Preconditions.checkArgument(col < size.y, "Column index is out of bounds");
        return matrix[row][col];
    }

    public void set(int row, int col, double value) {
        Preconditions.checkArgument(row < size.x, "Row index is out of bounds");
        Preconditions.checkArgument(col < size.y, "Column index is out of bounds");
        matrix[row][col] = value;
        determinant = null;
    }

    @Override
    public String toString() {
        return toJson(true);
    }

    public String toJson(boolean prettyPrint) {
        GsonBuilder gsonBuilder = new GsonBuilder();
        if (prettyPrint) {
            gsonBuilder.setPrettyPrinting();
        }
        Gson gson = gsonBuilder.create();
        return gson.toJson(this);
    }

    private int findRowWithNonZeroValue(int atColumn, int fromRow, int toRow) {
        for (int i = fromRow; i <= toRow; i++) {
            if (get(i, atColumn) != 0) {
                return i;
            }
        }
        return -1;
    }

    private void permutateRow(int r1, int r2) {
        double[] r1Values = copyRow(r1);
        double[] r2Values = copyRow(r2);
        copyRow(r1, r2Values);
        copyRow(r2, r1Values);
    }

    private double[] copyRow(int rowIndex) {
        double[] result = new double[size.y];
        System.arraycopy(matrix[rowIndex], 0, result, 0, size.y);
        return result;
    }

    public void copyRow(int rowIndex, double[] values) {
        System.arraycopy(values, 0, matrix[rowIndex], 0, size.y);
    }

    @Getter
    @RequiredArgsConstructor
    @EqualsAndHashCode
    public static class Size {
        private final int x;
        private final int y;
    }
}
