# Development Environment Setup

## Files to Carry Over to GH200

### Required (committed to repo)
These are already in the repo — just clone and you're set:
- `CLAUDE.md` — Project context for Claude Code (loaded automatically when Claude Code runs in this directory)
- `TASKS.md` — Phased implementation checklist
- `experiments/configs/gpt2_small.yaml` — Training config
- `.claude/settings.local.json` — Claude Code permission allowlist (optional, Claude will re-prompt)

### Not Required to Copy
These are local to your machine and will be auto-created:
- `~/.claude/settings.json` — Global Claude Code settings (currently empty `{}`)
- `~/.claude/projects/` — Session history, auto-created per project
- `~/.claude/todos/` — Todo state, auto-created

---

## Option 1: Claude Code via SSH Terminal

### Install Claude Code on GH200
```bash
# Install Node.js (required for Claude Code)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Claude Code
npm install -g @anthropic-ai/claude-code

# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."
# Or add to ~/.bashrc for persistence:
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
```

### Usage
```bash
# SSH into GH200
ssh user@gh200-node

# Navigate to project
cd ~/circuit-tracer

# Start Claude Code — it auto-reads CLAUDE.md for project context
claude

# Or run a one-shot command
claude "run the tests and show me the results"
```

### Tips for SSH Usage
- Claude Code works fully in terminal — no GUI needed
- `CLAUDE.md` at the repo root is automatically loaded as project context
- Use `/help` inside Claude Code to see all commands
- Use `/clear` to reset conversation context
- Session history persists in `~/.claude/projects/` on the GH200

---

## Option 2: Cursor with Remote SSH to GH200

### Prerequisites
1. Install [Cursor](https://cursor.com) on your local machine
2. Install the **Remote - SSH** extension in Cursor (same as VS Code's)

### Setup SSH Config (local machine)
Add to `~/.ssh/config`:
```
Host gh200
    HostName <gh200-ip-or-hostname>
    User <your-username>
    IdentityFile ~/.ssh/id_rsa
    ForwardAgent yes
```

### Connect from Cursor
1. Open Cursor
2. `Cmd+Shift+P` → "Remote-SSH: Connect to Host" → select `gh200`
3. Open the project folder: `~/circuit-tracer`
4. Cursor's terminal will run on the GH200 (GPU access works)
5. Cursor's AI features (Cmd+K, Ctrl+L) work with the remote files

### Cursor + Claude Code Together
You can use both simultaneously:
- **Cursor** for file editing, code navigation, inline AI suggestions
- **Claude Code** in Cursor's integrated terminal for multi-file tasks, running tests, training

In Cursor's terminal:
```bash
# Claude Code runs in the integrated terminal with full GPU access
claude
```

### Cursor Settings for Python Remote
Add to Cursor's `settings.json` (Cmd+Shift+P → "Open Remote Settings"):
```json
{
    "python.defaultInterpreterPath": "/path/to/your/python",
    "python.analysis.typeCheckingMode": "basic"
}
```

---

## Option 3: VS Code with Claude Code Extension

Same as Cursor but using VS Code:
1. Install VS Code + Remote-SSH extension
2. Install the [Claude Code extension](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code) from the marketplace
3. Connect to GH200 via Remote-SSH
4. Claude Code runs as a sidebar panel inside VS Code

---

## Environment Setup on GH200 (All Options)

After SSH-ing or connecting remotely:

```bash
# Clone your fork
git clone git@github.com:supreethms1809/circuit-tracer.git
cd circuit-tracer
git checkout claude/silly-feistel

# Create conda environment (recommended)
conda create -n kan-clt python=3.12 -y
conda activate kan-clt

# Install PyTorch with CUDA (GH200 = Hopper = sm_90)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install project + dependencies
pip install -e ".[dev]"
pip install git+https://github.com/Blealtan/efficient-kan.git
pip install datasets pyyaml

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# Verify Spline-CLT
pytest tests/test_kan_encoder.py tests/test_kan_transcoder.py tests/test_attribution.py -v
```

## Quick Reference: What Goes Where

| Item | Location | Carried Over? |
|------|----------|---------------|
| `CLAUDE.md` | Repo root | Yes (in git) |
| `TASKS.md` | Repo root | Yes (in git) |
| `.claude/settings.local.json` | Repo `.claude/` | Optional (re-created by Claude Code) |
| `~/.claude/settings.json` | Home dir | No (empty, auto-created) |
| `~/.claude/projects/` | Home dir | No (session history, auto-created) |
| Anthropic API key | `~/.bashrc` or env var | Must set manually on GH200 |
| SSH keys | `~/.ssh/` | Must set up for git push on GH200 |
| Conda env | System | Must create on GH200 |
