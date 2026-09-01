import os
import datetime

def main():
    root = "/gscratch/ssuresh/results/paper/paper_gpt2_small/runs/spline_feature_match_gpt2_small/seed_101/evaluation"
    print("Checking contents of evaluation dir:", root)
    if not os.path.exists(root):
        print("Evaluation directory does not exist yet!")
        return

    for item in sorted(os.listdir(root)):
        path = os.path.join(root, item)
        is_dir = os.path.isdir(path)
        if is_dir:
            print(f"[DIR] {item}")
            # list files inside
            for sub in sorted(os.listdir(path)):
                subpath = os.path.join(path, sub)
                mtime = os.path.getmtime(subpath)
                mtime_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                print(f"  [FILE] {item}/{sub} (modified: {mtime_str}, size: {os.path.getsize(subpath)} bytes)")
        else:
            mtime = os.path.getmtime(path)
            mtime_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"[FILE] {item} (modified: {mtime_str}, size: {os.path.getsize(path)} bytes)")

if __name__ == "__main__":
    main()
