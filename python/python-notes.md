Below is an in-depth set of Python notes covering the language's fundamentals, core features, object-oriented programming (OOP), and advanced topics. These notes are designed to provide a comprehensive understanding of Python, a versatile, high-level, interpreted language known for its readability and wide applicability in data science, web development, automation, and more.

---

### **1. Introduction to Python**
- **What is Python?**
  - A general-purpose, interpreted, dynamically-typed programming language created by Guido van Rossum in 1991.
  - Emphasizes code readability with its use of indentation and simple syntax.

- **Key Features**:
  - **Interpreted**: Executes code line-by-line (no compilation step).
  - **Dynamically Typed**: No need to declare variable types.
  - **Multi-Paradigm**: Supports procedural, object-oriented, and functional programming.
  - **Extensive Standard Library**: Rich set of modules and packages.
  - **Cross-Platform**: Runs on Windows, macOS, Linux, etc.

- **Python Versions**:
  - **Python 2**: Legacy, end-of-life in 2020.
  - **Python 3**: Current version (e.g., 3.11 as of 2025), with ongoing updates.

- **Running Python**:
  - Interactive mode: `python` or `python3` in terminal.
  - Script mode: `python3 script.py`.

---

### **2. Python Basics**

#### **2.1 Syntax and Structure**
- **Basic Program**:
  ```python
  print("Hello, World!")
  ```
  - No semicolons or braces; indentation defines code blocks.

- **Comments**:
  - Single-line: `# Comment`
  - Multi-line: `""" Comment """`

#### **2.2 Variables and Data Types**
- **Variables**:
  - Dynamically typed, assigned with `=`.
  ```python
  x = 10        # Integer
  name = "Alice" # String
  pi = 3.14     # Float
  ```
- **Basic Data Types**:
  - `int`: Whole numbers (e.g., `42`)
  - `float`: Decimal numbers (e.g., `3.14`)
  - `str`: Strings (e.g., `"hello"`)
  - `bool`: Boolean (`True`, `False`)
  - `complex`: Complex numbers (e.g., `3 + 4j`)
- **Type Checking**:
  ```python
  print(type(x))  # <class 'int'>
  ```

#### **2.3 Operators**
- **Arithmetic**: `+`, `-`, `*`, `/`, `//` (floor division), `%` (modulus), `**` (exponentiation)
  ```python
  print(5 / 2)   # 2.5
  print(5 // 2)  # 2
  print(2 ** 3)  # 8
  ```
- **Comparison**: `==`, `!=`, `>`, `<`, `>=`, `<=`
- **Logical**: `and`, `or`, `not`
- **Assignment**: `=`, `+=`, `-=`, etc.

#### **2.4 Control Flow**
- **If-Else**:
  ```python
  x = 10
  if x > 0:
      print("Positive")
  elif x == 0:
      print("Zero")
  else:
      print("Negative")
  ```
- **Loops**:
  - **For**:
    ```python
    for i in range(5):  # 0 to 4
        print(i)
    ```
  - **While**:
    ```python
    i = 0
    while i < 5:
        print(i)
        i += 1
    ```
  - **Break and Continue**:
    ```python
    for i in range(10):
        if i == 5:
            break  # Exit loop
        if i % 2 == 0:
            continue  # Skip even numbers
        print(i)
    ```

---

### **3. Data Structures**

#### **3.1 Lists**
- Ordered, mutable collections.
  ```python
  fruits = ["apple", "banana", "cherry"]
  fruits.append("orange")  # Add item
  fruits[1] = "grape"     # Modify
  print(fruits)           # ['apple', 'grape', 'cherry', 'orange']
  ```

#### **3.2 Tuples**
- Ordered, immutable collections.
  ```python
  point = (3, 4)
  print(point[0])  # 3
  # point[0] = 5  # Error: tuples are immutable
  ```

#### **3.3 Dictionaries**
- Key-value pairs, unordered, mutable.
  ```python
  student = {"name": "Alice", "age": 20}
  student["grade"] = "A"  # Add key-value
  print(student["name"])  # Alice
  ```

#### **3.4 Sets**
- Unordered, mutable collections of unique elements.
  ```python
  numbers = {1, 2, 3, 3}
  print(numbers)  # {1, 2, 3}
  numbers.add(4)
  print(numbers)  # {1, 2, 3, 4}
  ```

---

### **4. Functions**
- **Definition**:
  ```python
  def greet(name):
      return f"Hello, {name}!"
  print(greet("Alice"))  # Hello, Alice!
  ```
- **Default Arguments**:
  ```python
  def greet(name="Guest"):
      return f"Hello, {name}!"
  print(greet())  # Hello, Guest!
  ```
- **Keyword Arguments**:
  ```python
  def describe_person(name, age):
      return f"{name} is {age} years old."
  print(describe_person(age=25, name="Bob"))
  ```
- **Lambda Functions**:
  ```python
  add = lambda x, y: x + y
  print(add(3, 4))  # 7
  ```

---

### **5. Object-Oriented Programming (OOP) in Python**

#### **5.1 Classes and Objects**
- **Syntax**:
  ```python
  class Dog:
      def __init__(self, name, age):  # Constructor
          self.name = name
          self.age = age

      def bark(self):
          print(f"{self.name} says Woof!")
  
  dog = Dog("Rex", 5)
  dog.bark()  # Rex says Woof!
  ```
- **`self`**: Refers to the instance of the class.

#### **5.2 Encapsulation**
- **Private Attributes**: Use `_` (convention) or `__` (name mangling).
  ```python
  class BankAccount:
      def __init__(self, balance):
          self.__balance = balance  # Private

      def deposit(self, amount):
          if amount > 0:
              self.__balance += amount

      def get_balance(self):
          return self.__balance

  account = BankAccount(100)
  account.deposit(50)
  print(account.get_balance())  # 150
  ```

#### **5.3 Inheritance**
- **Syntax**:
  ```python
  class Animal:
      def __init__(self, name):
          self.name = name

      def speak(self):
          pass

  class Cat(Animal):
      def speak(self):
          return f"{self.name} says Meow!"

  cat = Cat("Whiskers")
  print(cat.speak())  # Whiskers says Meow!
  ```

#### **5.4 Polymorphism**
- **Method Overriding**:
  ```python
  class Bird(Animal):
      def speak(self):
          return f"{self.name} says Chirp!"

  animals = [Cat("Whiskers"), Bird("Tweety")]
  for animal in animals:
      print(animal.speak())  # Polymorphic behavior
  ```

#### **5.5 Abstraction**
- Use `abc` module for abstract base classes:
  ```python
  from abc import ABC, abstractmethod

  class Shape(ABC):
      @abstractmethod
      def area(self):
          pass

  class Rectangle(Shape):
      def __init__(self, width, height):
          self.width = width
          self.height = height

      def area(self):
          return self.width * self.height

  rect = Rectangle(3, 4)
  print(rect.area())  # 12
  ```

---

### **6. Advanced Python Features**

#### **6.1 List Comprehensions**
- Concise way to create lists:
  ```python
  squares = [x**2 for x in range(5)]
  print(squares)  # [0, 1, 4, 9, 16]
  ```

#### **6.2 Generators**
- Lazy evaluation for memory efficiency:
  ```python
  def fibonacci(n):
      a, b = 0, 1
      for _ in range(n):
          yield a
          a, b = b, a + b

  print(list(fibonacci(5)))  # [0, 1, 1, 2, 3]
  ```

#### **6.3 Decorators**
- Modify function behavior:
  ```python
  def log(func):
      def wrapper(*args):
          print(f"Calling {func.__name__}")
          return func(*args)
      return wrapper

  @log
  def add(a, b):
      return a + b

  print(add(2, 3))  # Calling add, then 5
  ```

#### **6.4 Exception Handling**
- **Syntax**:
  ```python
  try:
      result = 10 / 0
  except ZeroDivisionError as e:
      print(f"Error: {e}")
  else:
      print("No error")
  finally:
      print("Done")
  ```
- **Raising Exceptions**:
  ```python
  raise ValueError("Invalid input")
  ```

#### **6.5 Modules and Packages**
- **Importing**:
  ```python
  import math
  print(math.sqrt(16))  # 4.0
  ```
- **Custom Module** (e.g., `mymodule.py`):
  ```python
  # mymodule.py
  def say_hello():
      print("Hello from module!")
  ```
  ```python
  # main.py
  import mymodule
  mymodule.say_hello()
  ```

---

### **7. File Handling**
- **Reading/Writing**:
  ```python
  with open("file.txt", "w") as f:
      f.write("Hello, Python!")

  with open("file.txt", "r") as f:
      print(f.read())  # Hello, Python!
  ```

---

### **8. Libraries and Ecosystem**
- **Common Libraries**:
  - `numpy`: Numerical computing.
  - `pandas`: Data analysis.
  - `matplotlib`: Plotting.
  - `requests`: HTTP requests.
- **Example** (using `requests`):
  ```python
  import requests
  response = requests.get("https://api.github.com")
  print(response.status_code)  # 200
  ```

---

### **9. Best Practices**
- **PEP 8**: Follow style guide (e.g., 4-space indentation, snake_case for variables).
- **Docstrings**: Use `"""Docstring"""` for documentation.
- **Type Hints** (Python 3.5+):
  ```python
  def add(a: int, b: int) -> int:
      return a + b
  ```
- **Virtual Environments**: Use `venv` to isolate dependencies:
  ```bash
  python3 -m venv myenv
  source myenv/bin/activate  # Linux/macOS
  ```

---

### **10. Example Program**
Here’s a complete example combining multiple concepts:
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14 * self.radius ** 2

def main():
    shapes = [Circle(5), Circle(3)]
    areas = [shape.area() for shape in shapes]
    print(f"Areas: {areas}")  # Areas: [78.5, 28.26]

if __name__ == "__main__":
    main()
```

---

These notes provide a thorough foundation in Python. Let me know if you’d like deeper exploration into topics like data science with Python, web frameworks (e.g., Flask/Django), or concurrency!
