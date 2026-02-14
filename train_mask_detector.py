import os
import matplotlib.pyplot as plt

# Path to your dataset
DATASET_DIR = r"C:\Mask Detection\CODE\Face-Mask-Detection-master\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# Count the number of images per category
category_counts = {}
for category in CATEGORIES:
    path = os.path.join(DATASET_DIR, category)
    category_counts[category] = len(os.listdir(path))

# Plot the distribution
plt.figure(figsize=(6, 4))
plt.bar(category_counts.keys(), category_counts.values(), color=['green', 'red'])
plt.title('Dataset Distribution')
plt.xlabel('Category')
plt.ylabel('Number of Images')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
