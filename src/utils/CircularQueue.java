package utils;

/**
 * A fixed-size Circular Queue implementation for double values.
 * Used to track the moving average of training loss efficiently.
 * 
 * Time Complexity:
 * - add(): O(1)
 * - getAverage(): O(1) (maintains a running sum)
 * 
 * Space Complexity: O(K) where K is the capacity.
 */
public class CircularQueue {
    private final double[] buffer;
    private final int capacity;
    private int head;
    private int tail;
    private int count;
    private double sum;

    public CircularQueue(int capacity) {
        this.capacity = capacity;
        this.buffer = new double[capacity];
        this.head = 0;
        this.tail = 0;
        this.count = 0;
        this.sum = 0.0;
    }

    /**
     * Adds a new value to the queue.
     * If the queue is full, it overwrites the oldest value (at head).
     */
    public void add(double value) {
        if (count == capacity) {
            // Queue is full, remove the oldest value from sum
            sum -= buffer[head];
            head = (head + 1) % capacity;
            count--;
        }

        buffer[tail] = value;
        sum += value;
        tail = (tail + 1) % capacity;
        count++;
    }

    /**
     * Returns the average of all elements currently in the queue.
     * Returns 0 if empty.
     */
    public double getAverage() {
        if (count == 0)
            return 0.0;
        return sum / count;
    }

    public int size() {
        return count;
    }

    public boolean isFull() {
        return count == capacity;
    }

    public boolean isEmpty() {
        return count == 0;
    }

    @Override
    public String toString() {
        return "CircularQueue[size=" + count + ", avg=" + getAverage() + "]";
    }
}
