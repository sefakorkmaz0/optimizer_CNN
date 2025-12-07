package tests;

import engine.Tensor;

public class TensorTest {
    public static void main(String[] args) {
        System.out.println("Running Tensor Tests...");

        testIndexing();
        testAddition();
        testDotProduct();

        System.out.println("All tests passed!");
    }

    private static void testIndexing() {
        System.out.print("Testing Indexing... ");
        Tensor t = new Tensor(2, 3, 3); // 2x3x3
        t.set(0, 1, 1, 5.0);
        t.set(1, 2, 2, 10.0);

        if (t.get(0, 1, 1) != 5.0)
            throw new RuntimeException("Get/Set failed");
        if (t.get(1, 2, 2) != 10.0)
            throw new RuntimeException("Get/Set failed");
        System.out.println("OK");
    }

    private static void testAddition() {
        System.out.print("Testing Addition... ");
        Tensor t1 = new Tensor(1, 2, 2);
        t1.fill(1.0);

        Tensor t2 = new Tensor(1, 2, 2);
        t2.fill(2.0);

        t1.add(t2);

        for (double v : t1.data) {
            if (v != 3.0)
                throw new RuntimeException("Addition failed");
        }
        System.out.println("OK");
    }

    private static void testDotProduct() {
        System.out.print("Testing Dot Product... ");
        Tensor t1 = new Tensor(1, 1, 3);
        t1.data[0] = 1;
        t1.data[1] = 2;
        t1.data[2] = 3;

        Tensor t2 = new Tensor(1, 1, 3);
        t2.data[0] = 4;
        t2.data[1] = 5;
        t2.data[2] = 6;

        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        double result = t1.dot(t2);

        if (result != 32.0)
            throw new RuntimeException("Dot product failed: " + result);
        System.out.println("OK");
    }
}
