#!/usr/bin/env bash
# Bundle paper_final/macag artifacts for off-cluster analysis.
#
#   MODE=json   logit_gap finals + graph-metadata (no ckpt, no shard logs) ~1GB
#   MODE=full   all three CLT trees under OUTROOT (includes ~140GB graph JSONs)
#
#   MODE=json bash scripts/pack_macag_paper_final.sh
#   MODE=full DEST=/gscratch/$USER/paper_final/bundles bash scripts/pack_macag_paper_final.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTROOT="${OUTROOT:-/gscratch/${USER}/paper_final/macag}"
DEST="${DEST:-/gscratch/${USER}/paper_final/bundles}"
MODE="${MODE:-json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PROMPTS="${PROMPTS:-$ROOT/macag/data/mib_benchmark_prompts.json}"

mkdir -p "$DEST"
MANIFEST="$DEST/MANIFEST_${STAMP}.json"
LIST="$DEST/filelist_${STAMP}.txt"

python3 - "$MANIFEST" "$OUTROOT" "$MODE" "$PROMPTS" "$ROOT" <<'PY'
import json, os, sys, hashlib, subprocess
from pathlib import Path

manifest_path, outroot, mode, prompts, repo_root = sys.argv[1:6]
outroot = Path(outroot)
clts = ["llama32-524k", "gemma2-426k", "gemma2-2.5M"]
files: list[str] = []

def add(p: Path) -> None:
    if p.is_file():
        files.append(str(p))

for extra in ["status.retry.txt", "assignment.json"]:
    add(outroot / extra)

for clt in clts:
    base = outroot / clt
    if not base.is_dir():
        continue
    for prompt_dir in sorted(base.glob("mib_*")):
        if mode == "json":
            lg = prompt_dir / "logit_gap"
            if lg.is_dir():
                for f in sorted(lg.glob("*.json")):
                    if f.name.endswith(".ckpt.json"):
                        continue
                    add(f)
            gm = prompt_dir / "graphs" / "graph-metadata.json"
            add(gm)
        else:
            for f in sorted(prompt_dir.rglob("*")):
                if f.is_file() and ".gpu_slots" not in f.parts:
                    add(f)

if Path(prompts).is_file():
    files.append(str(Path(prompts).resolve()))

list_path = manifest_path.replace("MANIFEST_", "filelist_").replace(".json", ".txt")

def count_game2(clt: str) -> int:
    d = outroot / clt
    if not d.is_dir():
        return 0
    n = 0
    for p in d.glob("mib_*"):
        if (p / "logit_gap" / "macag_game2.json").is_file():
            n += 1
    return n

head = subprocess.check_output(["git", "-C", repo_root, "rev-parse", "HEAD"], text=True).strip()
digest = hashlib.sha256(Path(prompts).read_bytes()).hexdigest() if Path(prompts).is_file() else None

payload = {
    "created_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "mode": mode,
    "outroot": str(outroot),
    "code_version": head,
    "prompts_sha256": digest,
    "game2_complete": {c: count_game2(c) for c in clts},
    "filelist": os.path.basename(list_path),
}
files.append(str(Path(manifest_path).resolve()))
payload["file_count"] = len(files)
Path(manifest_path).write_text(json.dumps(payload, indent=2) + "\n")
with open(list_path, "w") as lf:
    for f in files:
        lf.write(f + "\n")
print(f"listed {len(files)} files -> {list_path}")
PY

ARCHIVE="$DEST/macag_paper_final_${MODE}_${STAMP}.tar"
echo ">>> writing $ARCHIVE"
tar -cf "$ARCHIVE" --files-from="$LIST"
echo ">>> archive: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
echo ">>> optional: zstd -T0 -19 -o ${ARCHIVE}.zst ${ARCHIVE}"
