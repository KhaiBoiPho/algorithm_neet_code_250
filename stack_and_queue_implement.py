# Stack
class Stack:
    def __init__(self, capacity):
        self.stack = []
        self.capacity = capacity
    
    def push(self, item):
        if self.isFull():
            raise OverflowError("Stack is full")
        self.stack.append(item)
    
    def pop(self):
        if self.isEmpty():
            raise IndexError("Pop from empty stack")
        return self.stack.pop()
    
    def top(self):
        if self.isEmpty():
            raise IndexError("Peek from empty stack")
        return self.stack[-1]
    
    def isEmpty(self):
        return len(self.stack) == 0
    
    def isFull(self):
        return len(self.stack) == self.capacity
    
    def clear(self):
        self.stack.clear()


# Queue
class Queue:
    def __init__(self, capacity):
        self.queue = []
        self.capacity = capacity
    
    def enqueue(self, item):
        if self.isFull():
            raise OverflowError("Queue is full")
        self.queue.append(item)
    
    def dequeue(self):
        if self.isEmpty():
            raise IndexError("Dequeue from empty queue")
        return self.queue.pop(0)
    
    def front(self):
        if self.isEmpty():
            raise IndexError("Peek from empty queue")
        return self.queue[0]
    
    def isEmpty(self):
        return len(self.queue) == 0
    
    def isFull(self):
        return len(self.capacity) == self.capacity
    
    def size(self):
        return len(self.queue)
    
    def clear(self):
        self.queue.clear()