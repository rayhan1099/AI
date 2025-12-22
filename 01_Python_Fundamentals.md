# Python Fundamentals for AI Engineers

## 📖 Table of Contents
1. [Why Python for AI?](#why-python-for-ai)
2. [Python Basics](#python-basics)
3. [Data Structures](#data-structures)
4. [Functions and Modules](#functions-and-modules)
5. [Object-Oriented Programming](#object-oriented-programming)
6. [File Handling](#file-handling)
7. [Error Handling](#error-handling)
8. [List Comprehensions & Generators](#list-comprehensions--generators)
9. [Practice Exercises](#practice-exercises)

---

## Why Python for AI?

Python is the #1 language for AI/ML because:
- **Simple syntax** - Easy to learn and read
- **Rich ecosystem** - Thousands of ML libraries
- **Great community** - Extensive documentation and support
- **Flexibility** - Works for research and production

---

## Python Basics

### Variables and Data Types

```python
# Numbers
x = 10          # Integer
y = 3.14        # Float
z = 2 + 3j      # Complex

# Strings
name = "AI Engineer"
message = 'Hello, World!'
multiline = """This is a
multiline string"""

# Boolean
is_active = True
is_complete = False

# Type checking
print(type(x))  # <class 'int'>
print(type(name))  # <class 'str'>
```

### Type Conversion

```python
# Convert between types
num_str = "123"
num_int = int(num_str)      # 123
num_float = float(num_str)  # 123.0

# String formatting (important for ML logging)
name = "Model"
accuracy = 0.95
print(f"{name} accuracy: {accuracy:.2%}")  # Model accuracy: 95.00%
print(f"{name} accuracy: {accuracy:.4f}")  # Model accuracy: 0.9500
```

### Operators

```python
# Arithmetic
a, b = 10, 3
print(a + b)   # 13
print(a - b)   # 7
print(a * b)   # 30
print(a / b)   # 3.333...
print(a // b)  # 3 (floor division)
print(a % b)   # 1 (modulo)
print(a ** b)  # 1000 (exponentiation)

# Comparison
print(a == b)  # False
print(a != b)  # True
print(a > b)   # True
print(a <= b)  # False

# Logical
x, y = True, False
print(x and y)  # False
print(x or y)   # True
print(not x)    # False
```

---

## Data Structures

### Lists - Most Important for ML

```python
# Creating lists
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
empty = []

# Accessing elements
print(numbers[0])      # 1 (first element)
print(numbers[-1])     # 5 (last element)
print(numbers[1:3])    # [2, 3] (slicing)

# List operations
numbers.append(6)           # Add to end
numbers.insert(0, 0)        # Insert at index
numbers.extend([7, 8])      # Add multiple
numbers.remove(3)           # Remove value
popped = numbers.pop()      # Remove and return last
popped = numbers.pop(0)     # Remove and return at index

# List methods (very useful in ML)
data = [1, 2, 3, 4, 5]
print(len(data))            # 5
print(sum(data))            # 15
print(max(data))            # 5
print(min(data))            # 1
print(sorted(data, reverse=True))  # [5, 4, 3, 2, 1]

# List comprehension (Pythonic way)
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
evens = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]
```

### Tuples - Immutable Sequences

```python
# Creating tuples
point = (3, 4)
coordinates = (1, 2, 3)
single = (5,)  # Note the comma!

# Unpacking (very useful in ML)
x, y = point
a, b, c = coordinates

# Use cases: Returning multiple values from functions
def get_stats(data):
    return (min(data), max(data), sum(data) / len(data))

min_val, max_val, avg_val = get_stats([1, 2, 3, 4, 5])
```

### Dictionaries - Key-Value Pairs

```python
# Creating dictionaries
person = {
    "name": "John",
    "age": 30,
    "city": "NYC"
}

# Accessing values
print(person["name"])           # John
print(person.get("age"))        # 30
print(person.get("email", "N/A"))  # N/A (default if key doesn't exist)

# Adding/Updating
person["email"] = "john@example.com"
person["age"] = 31

# Dictionary methods
print(person.keys())    # dict_keys(['name', 'age', 'city', 'email'])
print(person.values())  # dict_values(['John', 31, 'NYC', 'john@example.com'])
print(person.items())   # dict_items([('name', 'John'), ...])

# Dictionary comprehension
squares_dict = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Very useful in ML for hyperparameters
hyperparams = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100
}
```

### Sets - Unique Elements

```python
# Creating sets
unique_numbers = {1, 2, 3, 4, 5}
another_set = set([1, 2, 2, 3, 3])  # {1, 2, 3} (duplicates removed)

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

print(set1 | set2)   # Union: {1, 2, 3, 4, 5, 6}
print(set1 & set2)   # Intersection: {3, 4}
print(set1 - set2)   # Difference: {1, 2}
print(set1 ^ set2)   # Symmetric difference: {1, 2, 5, 6}

# Useful in ML for finding unique classes
classes = set([1, 2, 1, 3, 2, 1])  # {1, 2, 3}
```

---

## Functions and Modules

### Defining Functions

```python
# Basic function
def greet(name):
    return f"Hello, {name}!"

# Function with default parameters
def power(x, n=2):
    return x ** n

print(power(3))    # 9 (uses default n=2)
print(power(3, 3)) # 27

# Function with multiple return values
def calculate_stats(numbers):
    return {
        "sum": sum(numbers),
        "mean": sum(numbers) / len(numbers),
        "max": max(numbers),
        "min": min(numbers)
    }

# Variable arguments
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3, 4, 5))  # 15

# Keyword arguments
def create_model(**kwargs):
    model_config = {
        "learning_rate": kwargs.get("lr", 0.001),
        "batch_size": kwargs.get("batch_size", 32),
        "epochs": kwargs.get("epochs", 10)
    }
    return model_config

config = create_model(lr=0.01, epochs=100)
```

### Lambda Functions - Quick Functions

```python
# Lambda syntax: lambda arguments: expression
square = lambda x: x ** 2
print(square(5))  # 25

# Very useful with map, filter, reduce
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))  # [1, 4, 9, 16, 25]
evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4]

# Sorting with lambda
points = [(1, 2), (3, 1), (2, 3)]
sorted_by_y = sorted(points, key=lambda p: p[1])  # [(3, 1), (1, 2), (2, 3)]
```

### Modules and Imports

```python
# Import entire module
import math
print(math.sqrt(16))  # 4.0

# Import specific function
from math import sqrt, pi
print(sqrt(16))  # 4.0
print(pi)        # 3.14159...

# Import with alias (common in ML)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import all (not recommended)
from math import *  # Avoid this in production code
```

### Creating Your Own Module

```python
# Save as my_ml_utils.py
def normalize(data):
    """Normalize data to 0-1 range"""
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

def standardize(data):
    """Standardize data (mean=0, std=1)"""
    mean = sum(data) / len(data)
    variance = sum((x - mean)**2 for x in data) / len(data)
    std = variance ** 0.5
    return [(x - mean) / std for x in data]

# Use it
# from my_ml_utils import normalize, standardize
```

---

## Object-Oriented Programming

### Classes and Objects

```python
# Basic class
class Model:
    def __init__(self, name, accuracy):
        self.name = name
        self.accuracy = accuracy
        self.trained = False
    
    def train(self):
        self.trained = True
        print(f"{self.name} is now trained!")
    
    def predict(self, data):
        if not self.trained:
            raise ValueError("Model must be trained first!")
        return f"Predicting on {data}"

# Create instance
model = Model("Neural Network", 0.95)
model.train()
result = model.predict([1, 2, 3])
```

### Inheritance

```python
# Base class
class BaseModel:
    def __init__(self, name):
        self.name = name
        self.trained = False
    
    def train(self):
        self.trained = True
    
    def predict(self, data):
        raise NotImplementedError("Subclass must implement")

# Derived class
class LinearModel(BaseModel):
    def __init__(self, name, learning_rate=0.01):
        super().__init__(name)
        self.learning_rate = learning_rate
    
    def predict(self, data):
        if not self.trained:
            raise ValueError("Model not trained")
        return [x * 2 for x in data]  # Simple prediction

# Usage
model = LinearModel("Linear Regression", learning_rate=0.001)
model.train()
predictions = model.predict([1, 2, 3])
```

### Special Methods (Magic Methods)

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Vector(x={self.x}, y={self.y})"
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    def __len__(self):
        return int((self.x**2 + self.y**2)**0.5)

v1 = Vector(3, 4)
v2 = Vector(1, 2)
print(v1 + v2)      # Vector(4, 6)
print(v1 * 2)       # Vector(6, 8)
print(len(v1))      # 5
```

---

## File Handling

### Reading and Writing Files

```python
# Writing to file
with open("data.txt", "w") as f:
    f.write("Hello, World!\n")
    f.write("This is line 2\n")

# Reading from file
with open("data.txt", "r") as f:
    content = f.read()        # Read entire file
    # OR
    lines = f.readlines()     # Read as list of lines
    # OR
    for line in f:            # Read line by line
        print(line.strip())

# Working with CSV (before using pandas)
import csv

# Writing CSV
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Age", "City"])
    writer.writerow(["John", 30, "NYC"])
    writer.writerow(["Jane", 25, "LA"])

# Reading CSV
with open("data.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# JSON files (very common in ML for configs)
import json

# Writing JSON
config = {
    "model_name": "neural_net",
    "learning_rate": 0.001,
    "epochs": 100
}
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

# Reading JSON
with open("config.json", "r") as f:
    config = json.load(f)
```

---

## Error Handling

### Try-Except Blocks

```python
# Basic error handling
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Multiple exceptions
try:
    value = int(input("Enter a number: "))
    result = 10 / value
except ValueError:
    print("Invalid input! Must be a number.")
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"An error occurred: {e}")

# Try-Except-Else-Finally
try:
    file = open("data.txt", "r")
    content = file.read()
except FileNotFoundError:
    print("File not found!")
else:
    print("File read successfully!")
finally:
    file.close()  # Always executes

# Raising exceptions
def validate_accuracy(accuracy):
    if not 0 <= accuracy <= 1:
        raise ValueError("Accuracy must be between 0 and 1")
    return accuracy

# Custom exceptions
class ModelNotTrainedError(Exception):
    pass

def predict(model):
    if not model.trained:
        raise ModelNotTrainedError("Model must be trained first")
    return model.predict()
```

---

## List Comprehensions & Generators

### List Comprehensions

```python
# Basic comprehension
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(10) if x % 2 == 0]

# Nested comprehensions
matrix = [[i*j for j in range(3)] for i in range(3)]
# [[0, 0, 0], [0, 1, 2], [0, 2, 4]]

# Dictionary comprehension
squares_dict = {x: x**2 for x in range(5)}

# Set comprehension
unique_squares = {x**2 for x in range(-5, 6)}
```

### Generators - Memory Efficient

```python
# Generator function
def squares_generator(n):
    for i in range(n):
        yield i**2

# Use generator
for square in squares_generator(10):
    print(square)

# Generator expression
squares = (x**2 for x in range(10))  # Note: parentheses, not brackets
print(list(squares))  # Convert to list if needed

# Memory efficient for large datasets
def read_large_file(filename):
    with open(filename, "r") as f:
        for line in f:
            yield line.strip()  # Process one line at a time
```

---

## Control Flow

### If-Else Statements

```python
# Basic if-else
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"

# Ternary operator
status = "Active" if score >= 70 else "Inactive"
```

### Loops

```python
# For loop
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# For loop with enumerate (very useful in ML)
items = ["apple", "banana", "cherry"]
for index, item in enumerate(items):
    print(f"{index}: {item}")

# For loop with zip (combine multiple lists)
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# While loop
count = 0
while count < 5:
    print(count)
    count += 1

# Break and continue
for i in range(10):
    if i == 3:
        continue  # Skip this iteration
    if i == 7:
        break      # Exit loop
    print(i)
```

---

## Practice Exercises

### Exercise 1: Basic Calculator
```python
# Create a function that takes two numbers and an operation
# Returns the result
def calculator(a, b, operation):
    # Your code here
    pass
```

### Exercise 2: Data Statistics
```python
# Create a function that takes a list of numbers
# Returns: mean, median, mode, standard deviation
def calculate_statistics(data):
    # Your code here
    pass
```

### Exercise 3: Text Processing
```python
# Create a function that processes text:
# - Removes punctuation
# - Converts to lowercase
# - Returns word count and unique words
def process_text(text):
    # Your code here
    pass
```

### Exercise 4: Simple ML Model Class
```python
# Create a Model class with:
# - __init__: takes model name and parameters
# - train: marks model as trained
# - predict: returns predictions (can be dummy for now)
# - evaluate: returns accuracy score
class SimpleModel:
    # Your code here
    pass
```

---

## Key Takeaways

1. **Master list comprehensions** - They're everywhere in ML code
2. **Understand dictionaries** - Used for configs, hyperparameters
3. **Learn lambda functions** - Useful with map, filter, reduce
4. **Practice file I/O** - You'll read/write data constantly
5. **Error handling is crucial** - ML code fails often, handle gracefully

---

## Next Steps

Once you're comfortable with these concepts, move to:
- **[02_NumPy_Pandas_Basics.md](02_NumPy_Pandas_Basics.md)** - Data manipulation for ML

---

**Practice makes perfect! Code along with every example and complete the exercises.**

