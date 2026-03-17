# String Methods - Practice
# Examples:
# "hello".upper()          → "HELLO"
# "  hi  ".strip()         → "hi"
# "a,b,c".split(",")       → ["a", "b", "c"]
# ",".join(["a", "b"])     → "a,b"
# "hello".replace("l","r") → "herro"

# ============================================
# Problem 1: Clean Labels
# ============================================
# You received messy labels from a dataset.
# Clean each label: strip whitespace and lowercase it.
#
# labels = ["  Cat ", "DOG  ", " biRd", "  FISH "]
#
# Print the cleaned list → ["cat", "dog", "bird", "fish"]
# Hint: use a list comprehension with .strip() and .lower()

labels = ["  Cat ", "DOG  ", " biRd", "  FISH "]
for label in labels:
    print(label.strip().lower())

# ============================================
# Problem 2: Parse a CSV Line
# ============================================
# ML datasets often come as CSV files (comma-separated values).
# Given this line from a CSV file, split it into a list.
#
# line = "0.5,1.2,3.7,0.8,2.1"
#
# Split it, convert each value to a float, and print the list.
# Expected output → [0.5, 1.2, 3.7, 0.8, 2.1]
# Hint: split(",") gives strings, use float() to convert
line = "0.5,1.2,3.7,0.8,2.1"

datas = line.split(",")
newData = []
print(datas)
for data in datas:
    print(data)
    newData.append(float(data))

print("New Datas",newData)

# ============================================
# Problem 3: Build a File Path
# ============================================
# In ML, you build file paths for loading images/data.
# Given these variables:
#
# folder = "datasets"
# subfolder = "train"
# filename = "img_001.png"
#
# Use join() to build the path: "datasets/train/img_001.png"
# Then replace "train" with "test" to get: "datasets/test/img_001.png"
# Print both paths.
# Hint: "/".join([...]) and .replace()

folder = "datasets"
subfolder = "train"
filename = "img_001.png"

train_path = "/".join([folder, subfolder, filename])
test_path = train_path.replace("train", "test")

print(train_path)
print(test_path)

# ============================================
# Problem 4: Training Log Formatter
# ============================================
# During ML training, you print logs every epoch.
# Given these variables:
#
# epoch = 3
# total_epochs = 10
# loss = 0.04217
# accuracy = 0.9341
#
# Use an f-string to print:
# "Epoch 03/10 | Loss: 0.0422 | Accuracy: 93.41%"
#
# Hints:
#   - f"{epoch:02d}" → zero-padded to 2 digits
#   - f"{loss:.4f}" → 4 decimal places
#   - f"{accuracy:.2%}" → percentage with 2 decimals


epoch = 3
total_epochs = 10
loss = 0.04217
accuracy = 0.9341

result = print(f"Epoch {epoch:02d}/{total_epochs} | Loss: {loss:.4f} | Accuracy: {accuracy:.2%}")




