Below is an in-depth set of Java notes covering the language's fundamentals, object-oriented programming (OOP), core features, and advanced topics. These notes are designed to provide a comprehensive understanding of Java, from syntax to practical application, building on the OOP concepts previously discussed but expanding into broader language details.

---

### **1. Introduction to Java**
- **What is Java?**
  - A high-level, object-oriented, platform-independent programming language developed by Sun Microsystems (now owned by Oracle) in 1995.
  - Famous for its "Write Once, Run Anywhere" (WORA) capability due to the Java Virtual Machine (JVM).

- **Key Features**:
  - **Platform Independence**: Compiled bytecode runs on any JVM, regardless of OS.
  - **Object-Oriented**: Built around classes and objects.
  - **Robust**: Strong type-checking, exception handling, and garbage collection.
  - **Secure**: Sandboxed execution, no direct memory access.
  - **Multithreaded**: Built-in support for concurrent programming.

- **Java Editions**:
  - **Java SE (Standard Edition)**: Core libraries and APIs.
  - **Java EE (Enterprise Edition)**: For large-scale, distributed applications.
  - **Java ME (Micro Edition)**: For mobile and embedded devices.

- **Compilation and Execution**:
  - Source code (`.java`) → Compiled by `javac` → Bytecode (`.class`) → Executed by JVM (`java` command).

---

### **2. Java Basics**

#### **2.1 Syntax and Structure**
- **Basic Program**:
  ```java
  public class HelloWorld {
      public static void main(String[] args) {
          System.out.println("Hello, World!");
      }
  }
  ```
  - `public class`: Must match the filename (e.g., `HelloWorld.java`).
  - `main`: Entry point of the program.
  - `static`: Allows method to be called without an object.
  - `String[] args`: Command-line arguments.

- **Comments**:
  - Single-line: `// Comment`
  - Multi-line: `/* Comment */`
  - Javadoc: `/** Documentation */`

#### **2.2 Variables and Data Types**
- **Primitive Types**:
  - `byte`: 8-bit (-128 to 127)
  - `short`: 16-bit (-32,768 to 32,767)
  - `int`: 32-bit (-2^31 to 2^31-1)
  - `long`: 64-bit (-2^63 to 2^63-1), suffix `L` (e.g., `100L`)
  - `float`: 32-bit floating-point, suffix `f` (e.g., `3.14f`)
  - `double`: 64-bit floating-point (e.g., `3.14`)
  - `char`: 16-bit Unicode character (e.g., `'A'`)
  - `boolean`: `true` or `false`
- **Declaration**:
  ```java
  int age = 25;
  double salary = 50000.50;
  ```

- **Reference Types**:
  - Objects (e.g., `String`, `ArrayList`), stored as references to memory locations.
  - Example: `String name = "John";`

#### **2.3 Operators**
- **Arithmetic**: `+`, `-`, `*`, `/`, `%` (modulus)
- **Relational**: `==`, `!=`, `>`, `<`, `>=`, `<=`
- **Logical**: `&&` (AND), `||` (OR), `!` (NOT)
- **Bitwise**: `&`, `|`, `^` (XOR), `~`, `<<`, `>>`
- **Assignment**: `=`, `+=`, `-=`, etc.
- **Ternary**: `condition ? value1 : value2`
  ```java
  int max = (a > b) ? a : b;
  ```

#### **2.4 Control Flow**
- **If-Else**:
  ```java
  int x = 10;
  if (x > 0) {
      System.out.println("Positive");
  } else if (x == 0) {
      System.out.println("Zero");
  } else {
      System.out.println("Negative");
  }
  ```
- **Switch**:
  ```java
  int day = 2;
  switch (day) {
      case 1: System.out.println("Monday"); break;
      case 2: System.out.println("Tuesday"); break;
      default: System.out.println("Unknown");
  }
  ```
- **Loops**:
  - **For**:
    ```java
    for (int i = 0; i < 5; i++) {
        System.out.println(i);
    }
    ```
  - **Enhanced For** (for-each):
    ```java
    int[] numbers = {1, 2, 3};
    for (int num : numbers) {
        System.out.println(num);
    }
    ```
  - **While**:
    ```java
    int i = 0;
    while (i < 5) {
        System.out.println(i);
        i++;
    }
    ```
  - **Do-While**:
    ```java
    int i = 0;
    do {
        System.out.println(i);
        i++;
    } while (i < 5);
    ```

---

### **3. Object-Oriented Programming (OOP) in Java**
(Expanding on earlier OOP notes with additional details)

#### **3.1 Classes and Objects**
- **Class Declaration**:
  ```java
  class Person {
      String name;
      int age;

      Person(String name, int age) {
          this.name = name;
          this.age = age;
      }

      void introduce() {
          System.out.println("Hi, I'm " + name + ", " + age + " years old.");
      }
  }
  ```
- **Object Creation**:
  ```java
  Person person = new Person("Alice", 30);
  person.introduce();
  ```

#### **3.2 Encapsulation**
- Use `private` fields with `public` getters/setters:
  ```java
  class BankAccount {
      private double balance;

      public void deposit(double amount) {
          if (amount > 0) balance += amount;
      }

      public double getBalance() {
          return balance;
      }
  }
  ```

#### **3.3 Inheritance**
- **Syntax**: `class Child extends Parent`
- **Superclass Access**: Use `super`:
  ```java
  class Vehicle {
      String type = "Generic";
  }

  class Car extends Vehicle {
      String model;
      Car(String model) {
          this.model = model;
          System.out.println(type + " " + model);
      }
  }
  ```

#### **3.4 Polymorphism**
- **Method Overloading**:
  ```java
  class MathUtils {
      int add(int a, int b) { return a + b; }
      double add(double a, double b) { return a + b; }
  }
  ```
- **Method Overriding**:
  ```java
  class Animal {
      void sound() { System.out.println("Generic sound"); }
  }
  class Dog extends Animal {
      @Override
      void sound() { System.out.println("Woof"); }
  }
  ```

#### **3.5 Abstraction**
- **Abstract Class**:
  ```java
  abstract class Shape {
      abstract double area();
  }
  class Rectangle extends Shape {
      double width, height;
      Rectangle(double w, double h) { width = w; height = h; }
      double area() { return width * height; }
  }
  ```
- **Interface**:
  ```java
  interface Printable {
      void print();
  }
  class Document implements Printable {
      public void print() { System.out.println("Printing..."); }
  }
  ```

---

### **4. Java Core Features**

#### **4.1 Arrays**
- **Declaration and Initialization**:
  ```java
  int[] numbers = new int[5]; // Array of size 5
  int[] values = {1, 2, 3, 4, 5}; // Initialized array
  ```
- **Access**:
  ```java
  System.out.println(numbers[0]); // First element
  ```

#### **4.2 Strings**
- **Immutable**: Cannot be changed after creation.
- **Common Methods**:
  - `length()`, `charAt(int index)`, `substring(int start, int end)`, `toLowerCase()`, `toUpperCase()`, `equals()`
- **Example**:
  ```java
  String str = "Hello";
  System.out.println(str.length()); // 5
  System.out.println(str.substring(1, 3)); // "el"
  ```

#### **4.3 Exception Handling**
- **Syntax**:
  ```java
  try {
      int result = 10 / 0;
  } catch (ArithmeticException e) {
      System.out.println("Error: " + e.getMessage());
  } finally {
      System.out.println("This always executes.");
  }
  ```
- **Types**:
  - `Exception` (checked), `RuntimeException` (unchecked).
- **Throwing Exceptions**:
  ```java
  throw new IllegalArgumentException("Invalid input");
  ```

#### **4.4 Collections Framework**
- **Purpose**: Dynamic data structures.
- **Common Classes**:
  - `ArrayList`: Dynamic array.
  - `HashMap`: Key-value pairs.
  - `HashSet`: Unique elements.
- **Example**:
  ```java
  import java.util.ArrayList;
  ArrayList<String> list = new ArrayList<>();
  list.add("Apple");
  list.add("Banana");
  System.out.println(list.get(0)); // Apple
  ```

#### **4.5 Multithreading**
- **Thread Creation**:
  - Extend `Thread`:
    ```java
    class MyThread extends Thread {
        public void run() {
            System.out.println("Thread running");
        }
    }
    MyThread t = new MyThread();
    t.start();
    ```
  - Implement `Runnable`:
    ```java
    class MyRunnable implements Runnable {
        public void run() {
            System.out.println("Runnable running");
        }
    }
    Thread t = new Thread(new MyRunnable());
    t.start();
    ```
- **Synchronization**:
  ```java
  synchronized void criticalSection() {
      // Thread-safe code
  }
  ```

---

### **5. Advanced Java Topics**

#### **5.1 Generics**
- **Purpose**: Type safety and reusability.
- **Example**:
  ```java
  class Box<T> {
      T item;
      Box(T item) { this.item = item; }
      T getItem() { return item; }
  }
  Box<Integer> intBox = new Box<>(123);
  ```

#### **5.2 Lambda Expressions (Java 8+)**
- **Syntax**: `(parameters) -> expression`
- **Example**:
  ```java
  import java.util.Arrays;
  String[] names = {"Alice", "Bob"};
  Arrays.sort(names, (a, b) -> a.compareTo(b));
  ```

#### **5.3 Streams (Java 8+)**
- **Purpose**: Process collections functionally.
- **Example**:
  ```java
  import java.util.List;
  import java.util.Arrays;
  List<Integer> numbers = Arrays.asList(1, 2, 3, 4);
  numbers.stream()
         .filter(n -> n % 2 == 0)
         .forEach(System.out::println); // Prints 2, 4
  ```

#### **5.4 File I/O**
- **Reading/Writing Files**:
  ```java
  import java.io.*;
  try (BufferedWriter writer = new BufferedWriter(new FileWriter("file.txt"))) {
      writer.write("Hello, Java!");
  } catch (IOException e) {
      e.printStackTrace();
  }
  ```

---

### **6. Best Practices**
- **Naming Conventions**: `ClassName`, `methodName`, `variableName`, `CONSTANT_NAME`.
- **Code Organization**: Use packages (e.g., `com.example.project`).
- **Error Handling**: Always catch specific exceptions.
- **Memory Management**: Rely on garbage collection, avoid unnecessary object creation.
- **Documentation**: Use Javadoc for public APIs.

---

### **7. Example Program**
Here’s a complete example combining multiple concepts:
```java
import java.util.ArrayList;

class Student {
    private String name;
    private int grade;

    public Student(String name, int grade) {
        this.name = name;
        this.grade = grade;
    }

    public int getGrade() { return grade; }
    public String getName() { return name; }

    @Override
    public String toString() {
        return name + ": " + grade;
    }
}

public class Main {
    public static void main(String[] args) {
        ArrayList<Student> students = new ArrayList<>();
        students.add(new Student("Alice", 85));
        students.add(new Student("Bob", 92));

        // Using Stream to filter and print
        students.stream()
                .filter(s -> s.getGrade() > 90)
                .forEach(System.out::println); // Prints: Bob: 92
    }
}
```

---

These notes cover Java from the ground up. Let me know if you’d like deeper exploration into specific areas like JDBC, Spring, or JavaFX!
