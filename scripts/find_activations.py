import os

def main():
    root = "/gscratch/ssuresh"
    print("Searching for activations under:", root)
    if not os.path.exists(root):
        print(f"Directory {root} does not exist!")
        return

    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        # We look for mlp_inputs_val.npy or mlp_inputs.npy
        for f in filenames:
            if "mlp_inputs" in f or "mlp_outputs" in f:
                found.append(os.path.join(dirpath, f))
                if len(found) >= 30:
                    break
        if len(found) >= 30:
            break

    print(f"Found {len(found)} activation files:")
    for f in sorted(found):
        print(" -", f)

if __name__ == "__main__":
    main()
