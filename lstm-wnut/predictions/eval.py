import os
for file in os.listdir("./"):
    if file.endswith(".txt"):
        print(os.path.join("./", file))