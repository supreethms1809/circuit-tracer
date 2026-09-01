import os

def main():
    root = "/gscratch/ssuresh/results/paper"
    print("Searching for spline_feature_match_gpt2_small checkpoints in", root)
    if not os.path.exists(root):
        print(f"Directory {root} does not exist!")
        return

    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "spline_feature_match_gpt2_small" in dirpath:
            for f in filenames:
                if f == "metadata.safetensors":
                    found.append(os.path.join(dirpath, f))

    print(f"Found {len(found)} metadata.safetensors files:")
    for f in sorted(found):
        print(" -", f)

if __name__ == "__main__":
    main()
