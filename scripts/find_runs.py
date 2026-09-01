import os

def main():
    root = "/gscratch/ssuresh/results/paper"
    print("Listing directories under:", root)
    if not os.path.exists(root):
        print(f"Directory {root} does not exist!")
        return

    for item in sorted(os.listdir(root)):
        path = os.path.join(root, item)
        if os.path.isdir(path):
            print(f" - {item}")
            # list subdirs
            subdirs = [s for s in os.listdir(path) if os.path.isdir(os.path.join(path, s))]
            print(f"   Subdirs: {subdirs}")

if __name__ == "__main__":
    main()
