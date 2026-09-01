import os

def main():
    root = "/gscratch/ssuresh/results/paper/paper_v2_gpt2_small_dt4096_2b/runs"
    print("Searching for checkpoints under:", root)
    if not os.path.exists(root):
        print(f"Directory {root} does not exist!")
        return

    for dirpath, dirnames, filenames in os.walk(root):
        # Check if metadata.safetensors or model.safetensors exists
        if "metadata.safetensors" in filenames or "model.safetensors" in filenames:
            print("Found Checkpoint:", dirpath)

if __name__ == "__main__":
    main()
