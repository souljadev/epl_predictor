import os

root = r"C:\Users\brend\OneDrive\Desktop\aiprojects\soccer_agent_local"

for base, dirs, files in os.walk(root):
    for f in files:
        if f.lower() == "compare_models.py":
            print("FOUND:", os.path.join(base, f))
