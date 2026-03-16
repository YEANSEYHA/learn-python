# Classes - Practice
# Examples:
# class Dog:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
#     def bark(self):
#         print(f"{self.name} says Woof!")
#
# dog1 = Dog("Rex", 3)
# dog1.bark()

# ============================================
# Problem 1: Bank Account
# ============================================
# Create a class called `BankAccount` with:
#   - __init__ takes owner (name) and balance (default 0)
#   - deposit(amount) → adds to balance
#   - withdraw(amount) → subtracts from balance (only if enough money)
#   - show() → prints "Owner: ___, Balance: ___"
#
# Test it:
#   acc = BankAccount("Matin", 100)
#   acc.deposit(50)
#   acc.withdraw(30)
#   acc.show()   # should print "Owner: Matin, Balance: 120"

class BankAccount:
    def __init__(self, name, balance=0):
        self.name = name
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
        else:
            print("Not enough funds!")

    def show(self):
        print(f"Owner: {self.name}, Balance: {self.balance}")

acc = BankAccount("Matin", 100)
acc.deposit(50)
acc.withdraw(30)
acc.show()

# ============================================
# Problem 2: Dataset
# ============================================
# In ML, you work with datasets constantly. Build one!
# Create a class called `Dataset` with:
#   - __init__ takes a name (string) and starts with an empty list called samples
#   - add(sample) → appends a sample to the list
#   - size() → returns the number of samples
#   - summary() → prints "Dataset: ___, Samples: ___"
#
# Test it:
#   ds = Dataset("MNIST")
#   ds.add([0, 0, 1])
#   ds.add([1, 0, 0])
#   ds.add([0, 1, 0])
#   print(ds.size())    # 3
#   ds.summary()         # "Dataset: MNIST, Samples: 3"

class Dataset:
    def __init__(self, name, samples=None):
        self.name = name
        self.samples = samples if samples else []

    def add(self, sample):
        self.samples.append(sample)

    def size(self):
        return len(self.samples)

    def summary(self):
        print(f"Dataset: {self.name}, Samples: {self.size()}")

ds = Dataset("MNIST")
ds.add([0, 0, 1])
ds.add([1, 0, 0])
ds.add([0, 1, 0])
print(ds.size())
ds.summary()

# ============================================
# Problem 3: Inheritance — SmartAccount
# ============================================
# Inheritance lets one class build on another. In ML, you'll
# write classes that inherit from nn.Module, adding your own layers.
#
# Create a class `SmartAccount` that inherits from BankAccount:
#   - Adds a history list (starts empty) in __init__
#   - Override deposit → same behavior + appends "Deposited ___" to history
#   - Override withdraw → same behavior + appends "Withdrew ___" to history
#   - Add print_history() → prints each entry in history
#
# Hint: use super().__init__(name, balance) to call the parent's __init__
#       use super().deposit(amount) to call the parent's deposit
#
# Test it:
#   sa = SmartAccount("Matin", 200)
#   sa.deposit(50)
#   sa.withdraw(100)
#   sa.withdraw(300)
#   sa.show()             # "Owner: Matin, Balance: 150"
#   sa.print_history()    # "Deposited 50"  then  "Withdrew 100"

class SmartAccount(BankAccount):
    def __init__(self, name, balance=0):
        super().__init__(name, balance)
        self.history = []

    def deposit(self, amount):
        super().deposit(amount)
        self.history.append(f"Deposited {amount}")

    def withdraw(self, amount):
        if amount <= self.balance:
            super().withdraw(amount)
            self.history.append(f"Withdrew {amount}")
        else:
            print("Not enough funds!")

    def print_history(self):
        for entry in self.history:
            print(entry)

sa = SmartAccount("Matin", 200)
sa.deposit(50)
sa.withdraw(100)
sa.withdraw(300)
sa.show()
sa.print_history()
