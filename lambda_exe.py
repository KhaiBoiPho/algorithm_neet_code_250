# 1. Use lambda with map()
numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))
print(doubled)

# Use lambda with filter()
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)

# Use lambda with sorted()
pairs = [(1, 'apple'), (3, 'banana'), (2, 'cherry')]
sorted_pairs = sorted(pairs, key=lambda x: x[0])
print(sorted_pairs)

# Using lambda with sorted() to sort by multiple criteria
employees = [("Alice", 30), ("Bob", 25), ("Charlie", 35)]
sorted_employees = sorted(employees, key=lambda x: (x[1], x[0]))
print(sorted_employees)

# Lambda with condition (Ternary operator)
is_even = lambda x: "Even" if x % 2 == 0 else "Odd"
print(is_even(4))
print(is_even(5))

# Lambda with min()/max() function
strings = ["apple", "banana", "cherry"]
longest = max(strings, key=lambda x : len(x))
print(longest)

# Lambda with enumerate()
numbers = [10, 20, 30, 40]
indices = list(map(lambda x: x[0], filter(lambda x: x[1] > 20, enumerate(numbers))))
print(indices)

# exe 1:
pairs = [("a", 1), ("b", 2), ("c", 3)]
dict_from_pairs = {k: v for k, v in map(lambda x: (x[0], x[1]), pairs)}
print(dict_from_pairs)