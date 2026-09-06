import os
import pickle

filenames = []

for root, dirs, files in os.walk("data"):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(root, file)
            filenames.append(path.replace("\\", "/"))

with open("filenames.pkl", "wb") as f:
    pickle.dump(filenames, f)

print("Total images:", len(filenames))
print("First 10 paths:")
print(*filenames[:10], sep="\n")