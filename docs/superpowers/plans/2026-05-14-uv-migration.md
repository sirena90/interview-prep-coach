# uv Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `requirements.txt` + `pip` with `pyproject.toml` + `uv.lock` as the sole package-management setup.

**Architecture:** Create `pyproject.toml` declaring runtime deps and a `dev` dependency group containing ruff. Generate `uv.lock` via `uv sync`. Remove `requirements.txt` and update all doc references to use `uv` commands.

**Tech Stack:** uv 0.7+, Python ≥3.10

---

## File Map

| Action | Path |
|---|---|
| Create | `pyproject.toml` |
| Create (generated) | `uv.lock` |
| Delete | `requirements.txt` |
| Update | `CLAUDE.md` |
| Update | `README.md` |
| Update | `ARCHITECTURE.md` |

---

### Task 1: Create pyproject.toml

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Create pyproject.toml**

Create `/pyproject.toml` with this exact content:

```toml
[project]
name = "interview-prep-coach"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "streamlit>=1.28.0",
    "anthropic>=0.39.0",
    "chromadb>=0.5.0",
    "pydantic>=2.0.0",
    "pypdf>=4.0.0",
    "python-dotenv>=1.0.0",
]

[dependency-groups]
dev = ["ruff"]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pyproject.toml"
```

---

### Task 2: Generate lockfile and verify install

**Files:**
- Create (generated): `uv.lock`

- [ ] **Step 1: Sync dependencies**

```bash
uv sync --group dev
```

Expected: uv resolves and installs all packages, creates `.venv/` and `uv.lock`. No errors.

- [ ] **Step 2: Smoke-test the install**

```bash
uv run python -c "import streamlit, anthropic, chromadb, pydantic, pypdf, dotenv; print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Smoke-test ruff**

```bash
uv run ruff --version
```

Expected: prints ruff version, e.g. `ruff 0.x.y`

- [ ] **Step 4: Commit lockfile**

```bash
git add uv.lock
git commit -m "build: add uv.lock"
```

---

### Task 3: Delete requirements.txt

**Files:**
- Delete: `requirements.txt`

- [ ] **Step 1: Delete the file**

```bash
git rm requirements.txt
```

- [ ] **Step 2: Commit**

```bash
git commit -m "build: remove requirements.txt"
```

---

### Task 4: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Replace the Commands section**

Find this block in `CLAUDE.md`:

```markdown
## Commands

```bash
# Setup
cp .env.example .env          # add ANTHROPIC_API_KEY
pip install -r requirements.txt

# Run
streamlit run app.py           # opens at http://localhost:8501

# Tests (none yet — directory doesn't exist)
pytest tests/
```
```

Replace it with:

```markdown
## Commands

```bash
# Setup
cp .env.example .env          # add ANTHROPIC_API_KEY
uv sync --group dev           # installs runtime + ruff

# Run
uv run streamlit run app.py   # opens at http://localhost:8501

# Lint
uv run ruff check .
uv run ruff format .

# Tests (none yet — directory doesn't exist)
uv run pytest tests/
```
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for uv"
```

---

### Task 5: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the Dependencies subsection**

Find this block in `README.md`:

```markdown
### 1. Dependencies

```bash
cd interview-prep-coach/simple

# Create a virtualenv (recommended)
python -m venv venv
source venv/bin/activate    # Mac / Linux
# venv\Scripts\activate     # Windows

# Install
pip install -r requirements.txt
```
```

Replace it with:

```markdown
### 1. Dependencies

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (creates .venv automatically)
uv sync --group dev
```
```

- [ ] **Step 2: Replace the Development section**

Find this block near the bottom of `README.md`:

```markdown
## Development

Run the tests (once they exist):

```bash
pytest tests/
```
```

Replace it with:

```markdown
## Development

```bash
# Lint and format
uv run ruff check .
uv run ruff format .

# Run tests (once they exist)
uv run pytest tests/
```
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README for uv"
```

---

### Task 6: Update ARCHITECTURE.md

**Files:**
- Modify: `ARCHITECTURE.md`

- [ ] **Step 1: Replace the Running locally section**

Find this block in `ARCHITECTURE.md` (section 8):

```markdown
```bash
# 1. Clone the repo, cd into simple/
cd interview-prep-coach/simple

# 2. Create a virtualenv (recommended)
python -m venv venv
source venv/bin/activate    # Mac / Linux
# venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up .env with the Anthropic API key
cp .env.example .env
# edit .env and paste your ANTHROPIC_API_KEY

# 5. Run Streamlit
streamlit run app.py
```
```

Replace it with:

```markdown
```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync --group dev

# 3. Set up .env with the Anthropic API key
cp .env.example .env
# edit .env and paste your ANTHROPIC_API_KEY

# 4. Run Streamlit
uv run streamlit run app.py
```
```

- [ ] **Step 2: Commit**

```bash
git add ARCHITECTURE.md
git commit -m "docs: update ARCHITECTURE.md for uv"
```

---

### Task 7: Final verification

- [ ] **Step 1: Confirm no pip/venv references remain in docs**

```bash
grep -rn "pip install\|python -m venv\|requirements.txt" README.md ARCHITECTURE.md CLAUDE.md
```

Expected: no output (zero matches).

- [ ] **Step 2: Confirm uv.lock and pyproject.toml are tracked**

```bash
git status
```

Expected: clean working tree.

- [ ] **Step 3: Full install from scratch**

```bash
rm -rf .venv
uv sync --group dev
uv run python -c "import streamlit, anthropic, chromadb, pydantic, pypdf, dotenv; print('OK')"
```

Expected: `OK` — proves the lockfile is self-contained.
