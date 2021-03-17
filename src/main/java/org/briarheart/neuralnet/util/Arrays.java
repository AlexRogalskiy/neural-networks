package org.briarheart.neuralnet.util;

import com.google.common.base.Preconditions;

import java.util.Random;
import java.util.function.IntBinaryOperator;
import java.util.function.IntSupplier;

/**
 * @author Roman Chigvintsev
 */
public class Arrays {
    private Arrays() {
        //no instance
    }

    /**
     * Copies the specified array.
     *
     * @param source array to be copied
     * @return a copy of the original array (must not be {@code null})
     * @throws NullPointerException if array to be copied is null
     */
    public static double[][] copyOf(double[][] source) {
        Preconditions.checkNotNull(source, "Source array must not be null");
        if (source.length == 0) {
            return source;
        }

        double[][] copy = new double[source.length][];
        for (int i = 0; i < source.length; i++) {
            double[] element = source[i];
            if (element == null || element.length == 0) {
                copy[i] = element;
            } else {
                copy[i] = java.util.Arrays.copyOf(element, element.length);
            }
        }
        return copy;
    }

    /**
     * Returns index of minimum element in the given array.
     *
     * @param a array in which index of minimum element is to be determined (must not be {@code null})
     * @return index of minimum element in the given array; if there is more than one minimum element in the given
     * array this method returns index of the first one; if the given array is a zero-length array this method
     * returns -1
     */
    public static int findMinimum(double[] a) {
        Preconditions.checkNotNull(a, "Array must not be null");
        if (a.length == 0) {
            return -1;
        }

        int result = 0;
        int i = 0;
        double candidate = a[i++];
        for (; i < a.length; i++) {
            double next = a[i];
            if (next < candidate) {
                candidate = next;
                result = i;
            }
        }
        return result;
    }

    /**
     * Returns index of maximum element in the given array.
     *
     * @param a array in which index of maximum element is to be determined (must not be {@code null})
     * @return index of maximum element in the given array; if there is more than one maximum element in the given
     * array this method returns index of the first one; if the given array is a zero-length array this method
     * returns -1
     */
    public static int findMaximum(double[] a) {
        Preconditions.checkNotNull(a, "Array must not be null");
        if (a.length == 0) {
            return -1;
        }

        int result = 0;
        int i = 0;
        double candidate = a[i++];
        for (; i < a.length; i++) {
            double next = a[i];
            if (next > candidate) {
                candidate = next;
                result = i;
            }
        }
        return result;
    }

    public static double[] flatten(double[][] a) {
        Preconditions.checkNotNull(a, "Array to be flattened must not be null");
        if (a.length == 0) {
            return new double[0];
        }
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            if (a[i] != null && a[i].length > 0) {
                Preconditions.checkArgument(a[i].length == 1, "Array cannot be flattened");
                result[i] = a[i][0];
            }
        }
        return result;
    }

    public static void fill(int[] a, IntSupplier supplier, IntBinaryOperator accumulator) {
        Preconditions.checkNotNull(a, "Array to be filled must not be null");
        Preconditions.checkNotNull(supplier, "Supplier must not be null");
        Preconditions.checkNotNull(accumulator, "Accumulator must not be null");

        if (a.length == 0) {
            return;
        }

        int initialValue = supplier.getAsInt();
        a[0] = initialValue;
        for (int i = 1; i < a.length; i++) {
            a[i] = accumulator.applyAsInt(i, a[i - 1]);
        }
    }

    public static void shuffle(int[] a) {
        Preconditions.checkNotNull(a, "Array to be shuffled must not be null");

        if (a.length == 0) {
            return;
        }

        Random random = new Random();
        for (int i = 0; i < a.length; i++) {
            int r = i + random.nextInt(a.length - i);
            int tmp = a[i];
            a[i] = a[r];
            a[r] = tmp;
        }
    }
}
